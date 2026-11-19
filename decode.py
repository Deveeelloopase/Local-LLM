#!/usr/bin/env python3
"""bench_ab.py — find out WHY decode tok/s differs between the benchmark and the server.

Changes ONE thing at a time, reads llama.cpp's own load log (offload, backend,
graph splits) and the GPU busy counter, and prints a verdict.  Read-only.

    systemctl stop local-llm-cloud            # the model needs the machine to itself
    python bench_ab.py ~/models/Qwen3-8B-Q4_K_M.gguf
    python bench_ab.py ~/models/Qwen3-4B-Q4_K_M.gguf --quick
"""
import argparse, glob, gc, os, re, statistics, sys, tempfile, threading, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--ctx", type=int, default=4096)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--tokens", type=int, default=40); ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--bw", type=float, default=19.2, help="memory bus peak GB/s (single-channel DDR4-2400 = 19.2)")
ap.add_argument("--quick", action="store_true", help="fewer runs/tokens")
a = ap.parse_args()
if a.quick: a.tokens, a.runs = 24, 2

import numpy as np
from llama_cpp import Llama
PROMPT = "Write a short paragraph about why the sky is blue."
LONG = ("The quick brown fox jumps over the lazy dog. " * 60)[:2400]
size_gb = Path(a.model).stat().st_size / 1e9

# ── machine facts (informational) ────────────────────────────────────────────
def rd(p, d=""):
    try: return Path(p).read_text().strip()
    except Exception: return d
print("═" * 74); print(f" decode-speed A/B · {Path(a.model).name} · {size_gb:.2f} GB"); print("═" * 74)
cpu = next((l.split(":")[1].strip() for l in rd("/proc/cpuinfo").splitlines() if l.startswith("model name")), "?")
gov = rd("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "n/a")
card = next(iter(sorted(glob.glob("/sys/class/drm/card[0-9]/device"))), "")
busy_f = f"{card}/gpu_busy_percent" if card and os.path.exists(f"{card}/gpu_busy_percent") else ""
print(f"  CPU      {cpu}  (governor: {gov})")
print(f"  GPU path {card or 'n/a'}   dpm level: {rd(card + '/power_dpm_force_performance_level', 'n/a') if card else 'n/a'}")
envs = {k: v for k, v in os.environ.items() if re.match(r"(HSA|HIP|ROC|GGML|VK|AMD_VULKAN|OMP_)", k)}
print(f"  env      {envs or 'none of HSA/HIP/GGML/VK/OMP set'}")
print(f"  bus      {a.bw:.1f} GB/s peak  →  hard ceiling for this file: {a.bw / size_gb:.2f} tok/s; "
      f"realistic (70%): {0.70 * a.bw / size_gb:.2f}; (62%): {0.62 * a.bw / size_gb:.2f}")
if os.geteuid() == 0:
    import subprocess
    try:
        dm = subprocess.run(["dmidecode", "-t", "memory"], capture_output=True, text=True, timeout=10).stdout
        mods = [b for b in dm.split("Memory Device") if re.search(r"Size:\s+\d+\s*[MG]B", b)]
        spd = re.findall(r"Configured (?:Memory|Clock) Speed:\s*(\d+)", dm)
        print(f"  RAM      {len(mods)} populated module(s), configured speed {sorted(set(spd))} MT/s "
              f"→ {'SINGLE channel' if len(mods) == 1 else str(len(mods)) + ' modules'}")
    except Exception: pass

# ── helpers ──────────────────────────────────────────────────────────────────
class Busy(threading.Thread):
    def __init__(s): super().__init__(daemon=True); s.v = []; s.stop = False
    def run(s):
        while not s.stop:
            if busy_f:
                try: s.v.append(float(rd(busy_f, "0")))
                except Exception: pass
            time.sleep(0.25)

def parse(log):
    g = lambda p: (re.search(p, log) or [None, None])[1]
    be = ("Vulkan" if "ggml_vulkan" in log else "ROCm/HIP" if re.search(r"ggml_cuda_init|ROCm", log)
          else "CUDA" if "CUDA" in log else "CPU-only")
    off = re.search(r"offloaded (\d+)/(\d+) layers", log)
    kvdev = re.findall(r"(\S+)\s+KV buffer size", log)
    return dict(backend=be, off=f"{off[1]}/{off[2]}" if off else "?",
                splits=(g(r"graph splits\s*=\s*([^\n]+)") or "?").strip()[:28],
                kv=",".join(sorted(set(kvdev))) or "?")

def load(**kw):
    """Load with llama.cpp logging ON and capture it (C stderr is fd 2)."""
    tf = tempfile.TemporaryFile(mode="w+b"); old = os.dup(2); os.dup2(tf.fileno(), 2)
    try: llm = Llama(model_path=a.model, n_ctx=a.ctx, n_threads=a.threads, n_threads_batch=a.threads,
                      use_mmap=True, verbose=True, **kw)
    finally:
        sys.stderr.flush(); os.dup2(old, 2); os.close(old)
    tf.seek(0); log = tf.read().decode("utf-8", "replace"); tf.close()
    return llm, log

def gen_tps(llm, toks, n, runs, **gkw):
    out = []
    for _ in range(runs):
        llm.reset(); first = None; c = 0
        for _t in llm.generate(toks, temp=0.0, reset=True, **gkw):
            now = time.perf_counter(); first = first or now; c += 1
            if c >= n: break
        out.append((c - 1) / max(time.perf_counter() - first, 1e-6))
    return statistics.median(out)

def warm_ttft(llm, toks):
    llm.reset(); t0 = time.perf_counter()
    next(iter(llm.generate(toks, temp=0.0, reset=True))); return time.perf_counter() - t0

def run_variant(name, kw, draft=False):
    gc.collect()
    if draft:                    # must be a constructor arg: it also forces logits_all=True inside llama-cpp-python
        from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
        kw = {**kw, "draft_model": LlamaPromptLookupDecoding(num_pred_tokens=2)}
    llm, log = load(**kw); info = parse(log)
    toks = llm.tokenize(PROMPT.encode(), add_bos=False, special=True)
    for _ in llm.generate(toks, temp=0.0, reset=True):       # warm-up: kernels, page-in (excluded)
        break
    llm.reset()
    b = Busy(); b.start()
    tps = gen_tps(llm, toks, a.tokens, a.runs)
    b.stop = True; b.join(1)
    busy = f"{statistics.mean(b.v):.0f}%" if b.v else "n/a"
    return llm, dict(name=name, tps=tps, busy=busy, **info)

# server-path decode loop: raw ctx.decode + numpy argmax, exactly what the batched engine does per step
def raw_loop(llm, steps=40):
    from llama_cpp._internals import LlamaBatch
    ctx = llm._ctx; V = llm._n_vocab
    toks = llm.tokenize(PROMPT.encode(), add_bos=False, special=True)
    ctx.kv_cache_clear()
    batch = LlamaBatch(n_tokens=len(toks), embd=0, n_seq_max=1)
    batch.set_batch(toks, 0, False); ctx.decode(batch)
    pos = len(toks); t_dec = t_smp = 0.0; tok = 0
    for i in range(steps):
        lg = np.ctypeslib.as_array(ctx.get_logits(), shape=(V,))
        t0 = time.perf_counter(); tok = int(np.argmax(lg.astype(np.float32, copy=True))); t1 = time.perf_counter()
        b1 = LlamaBatch(n_tokens=1, embd=0, n_seq_max=1); b1.set_batch([tok], pos, False)
        t2 = time.perf_counter(); ctx.decode(b1); ctx.get_logits(); t3 = time.perf_counter()
        pos += 1
        if i >= 4: t_smp += t1 - t0; t_dec += t3 - t2        # skip first 4 steps
    n = steps - 4
    return n / (t_dec + t_smp), t_dec / n * 1000, t_smp / n * 1000

# ── variants: ONE change at a time ───────────────────────────────────────────
BASE = dict(n_gpu_layers=36, flash_attn=True, n_batch=512, n_ubatch=512)
KVQ = dict(type_k=8, type_v=8)          # GGML_TYPE_Q8_0 — what the server sets (kv=q8_0)
V = [("A  bench config (f16 KV, fa, ngl=36, batch 512)", BASE, False),
     ("B  A + ngl=99 (also offload the output layer)", {**BASE, "n_gpu_layers": 99}, False),
     ("C  A + KV cache q8_0   ← server setting", {**BASE, **KVQ}, False),
     ("D  C + n_batch 2048    ← server setting", {**BASE, **KVQ, "n_batch": 2048}, False),
     ("E  D + prompt-lookup draft model ← server default", {**BASE, **KVQ, "n_batch": 2048}, True)]
if a.quick: V = [V[0], V[2], V[3]]
rows = []; raw = None
print(f"\n  {'variant':<52}{'tok/s':>7}{'GPU busy':>10}  offload  splits")
for name, kw, draft in V:
    try:
        llm, r = run_variant(name, kw, draft); rows.append(r)
        print(f"  {name:<52}{r['tps']:>7.2f}{r['busy']:>10}  {r['off']:<7}  {r['splits']}", flush=True)
        if name.startswith("D") or (a.quick and name.startswith("D")):
            try: raw = raw_loop(llm)
            except Exception as e: raw = ("err", str(e)[:80], 0)
    except Exception as e:
        print(f"  {name:<52}  failed: {str(e)[:60]}", flush=True)
    finally:
        try: del llm
        except Exception: pass
        gc.collect()

# ── verdict ──────────────────────────────────────────────────────────────────
print("\n" + "─" * 74)
if rows:
    base = rows[0]; eff = lambda t: t * size_gb
    print(f"  backend: {base['backend']}   offloaded: {base['off']}   KV on: {base['kv']}")
    print(f"  baseline {base['tps']:.2f} tok/s = {eff(base['tps']):.1f} GB/s effective "
          f"({eff(base['tps']) / a.bw * 100:.0f}% of the {a.bw:.1f} GB/s bus)")
    for r in rows[1:]:
        d = (r["tps"] / base["tps"] - 1) * 100
        tag = "no effect" if abs(d) < 5 else ("FASTER" if d > 0 else "SLOWER")
        print(f"  {r['name'][:1]} vs A: {d:+5.0f}%  {tag}")
    parent = {"B": "A", "C": "A", "D": "C", "E": "D"}      # each variant changes ONE thing vs its parent
    by = {r["name"][0]: r for r in rows}
    culprits = []
    for r in rows[1:]:
        p = by.get(parent.get(r["name"][0], "A"))
        if p and r["tps"] < 0.85 * p["tps"]:
            culprits.append((r["name"][0], r["name"][3:].split("←")[0].strip(), (r["tps"] / p["tps"] - 1) * 100))
    if culprits:
        print()
        for k, what, d in culprits:
            print(f"  ► '{k}' costs {abs(d):.0f}% of decode speed: {what}")
    elif len(rows) > 1:
        print("\n  ► None of the server settings tested here slows decode → the loss is outside the model settings "
              "(environment / contention / the server process itself).")
if raw and raw[0] != "err":
    tps, dec, smp = raw
    print(f"  server-style raw decode loop (config D): {tps:.2f} tok/s  = decode {dec:.0f} ms + sampling {smp:.1f} ms per token")
elif raw:
    print(f"  raw decode loop skipped: {raw[1]}")
print("─" * 74)
