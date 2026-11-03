"""
Worker child process. Owns one Llama(), wraps it in a minimal InferenceEngine,
serves requests over TCP loopback to the parent.

Invoked by the parent as:
    python -m local_llm_core.engine.worker.main

All configuration arrives via environment variables set by the parent:
    LLM_WORKER_PORT        127.0.0.1:<port>   — parent is listening
    LLM_WORKER_SECRET      hex secret, one per worker
    LLM_WORKER_CONFIG      base64(json) of {
                              "model_id": str,
                              "model_path": str,
                              "load_kwargs": {...},
                              "engine_config": {...},
                              "runtime": {...}
                           }

Reuses InferenceEngine so all guard logic (loop guard, solicitation guard,
chunked prefill, EWMA prefill tracker, stall detection) is identical to the
single-process path.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import signal
import sys
import time
from typing import Any, Dict

from local_llm_core.accel.protocol import read_frame, write_frame

log = logging.getLogger(__name__)


def _reconstruct_system_info(sys_info_dict: Dict[str, Any]) -> Any:
    """P1-6c: rebuild a SystemInfo from the parent's serialized dict.

    Used to skip the ~3 s `HardwareDetector.detect_all()` call on every
    warm respawn. `detect_all()` re-runs nvidia-smi, rocm-smi, sysfs
    walks, a RAM bandwidth triad benchmark, and — worst of all on this
    box — the whole llama-cpp-python backend probe. All of that was
    already done in the parent; the worker just needs the answer.

    Returns a SystemInfo on success, or None if the dict is missing
    required keys. Caller falls back to `detect_all()` in that case.

    Non-critical fields (temperature, power draw, memory_type enum) are
    filled with safe defaults — they are read only for logging and for
    `vram_total_mb` computation, neither of which affects inference
    correctness. The drift-sensitive parts (ram_total_mb, gpus[*].vram_
    total_mb, cpu_cores, ram_channels, measured bandwidth, gtt total)
    are preserved verbatim.
    """
    try:
        from local_llm_core.hal.hardware_detector import (
            SystemInfo, GPUInfo, GPUVendor, MemoryType,
        )
    except Exception as exc:
        log.warning("Cannot import HardwareDetector types for reconstruction: %s", exc)
        return None

    try:
        _vendor_map = {v.name.lower(): v for v in GPUVendor}
        gpus_data = sys_info_dict.get("gpus") or []
        gpus: list = []
        for i, g in enumerate(gpus_data):
            _v_str = str(g.get("vendor", "unknown")).lower()
            _vendor = _vendor_map.get(_v_str, GPUVendor.UNKNOWN)
            gpus.append(GPUInfo(
                vendor=_vendor,
                name=g.get("name", "unknown"),
                vram_total_mb=int(g.get("vram_total_mb", 0) or 0),
                vram_free_mb=int(g.get("vram_free_mb", 0) or 0),
                memory_type=MemoryType.UNKNOWN,
                driver_version=g.get("driver_version", "unknown"),
                compute_capability=g.get("compute_capability"),
                cuda_cores=g.get("cuda_cores"),
                backend=g.get("backend", "unknown"),
                pcie_gen=g.get("pcie_gen"),
                pcie_lanes=g.get("pcie_lanes"),
                bus_id=g.get("bus_id"),
                temperature_c=None,
                power_draw_w=None,
                power_limit_w=None,
                tdp_w=None,
                index=int(g.get("index", i) or i),
                supported=bool(g.get("supported", True)),
            ))

        return SystemInfo(
            cpu_model=sys_info_dict.get("cpu_model", "unknown"),
            cpu_cores=int(sys_info_dict.get("cpu_cores", 1) or 1),
            cpu_threads=int(sys_info_dict.get("cpu_threads", 1) or 1),
            ram_total_mb=int(sys_info_dict.get("ram_total_mb", 0) or 0),
            ram_free_mb=int(sys_info_dict.get("ram_free_mb", 0) or 0),
            os_name=sys_info_dict.get("os_name", "linux"),
            kernel_version=sys_info_dict.get("kernel_version", "unknown"),
            arch=sys_info_dict.get("arch", "x86_64"),
            gpus=gpus,
            primary_gpu=gpus[0] if gpus else None,
            has_cuda=bool(sys_info_dict.get("has_cuda", False)),
            has_rocm=bool(sys_info_dict.get("has_rocm", False)),
            has_opencl=bool(sys_info_dict.get("has_opencl", False)),
            has_vulkan=bool(sys_info_dict.get("has_vulkan", False)),
            has_numba=False,
            has_numpy=True,
            available_backends=list(sys_info_dict.get("available_backends") or ["cpu"]),
            recommended_backend=sys_info_dict.get("recommended_backend", "cpu"),
            ram_channels=int(sys_info_dict.get("ram_channels", 0) or 0),
            ram_bandwidth_gbps_theoretical=float(
                sys_info_dict.get("ram_bandwidth_gbps_theoretical", 0.0) or 0.0
            ),
            ram_bandwidth_gbps_measured=float(
                sys_info_dict.get("ram_bandwidth_gbps_measured", 0.0) or 0.0
            ),
            llama_cpp_backends=list(sys_info_dict.get("llama_cpp_backends") or []),
            llama_cpp_gpu_offload=bool(sys_info_dict.get("llama_cpp_gpu_offload", False)),
            vram_gtt_total_mb=int(sys_info_dict.get("vram_gtt_total_mb", 0) or 0),
        )
    except Exception as exc:
        log.warning("Failed to reconstruct SystemInfo from dict: %s", exc)
        return None


async def _load_engine(cfg: Dict[str, Any]) -> Any:
    from local_llm_core.engine.gguf.model_loader import ModelConfig, ModelLoader
    from local_llm_core.engine.inference.inference_engine import (
        InferenceEngine, InferenceRequest,
    )
    from local_llm_core.memory.vram_manager import VRAMManager
    from local_llm_core.memory.ram_optimizer import RAMOptimizer
    from local_llm_core.memory.swap_handler import SwapHandler
    from local_llm_core.memory.hybrid_manager import HybridMemoryManager
    from local_llm_core.hal.hardware_detector import HardwareDetector
    import psutil

    # ── reconstruct SystemInfo from the parent when available ────
    # detect_all() costs ~3 s per warm respawn on this box (rocm-smi,
    # sysfs walks, RAM bandwidth triad, llama-cpp backend probe). The
    # parent already ran it; the worker does not need to re-measure to
    # serve one model. Skipping it is what turns a ~15 s warm respawn
    # into a ~5 s one.
    #
    # api_server.py currently passes `hardware_profile` but not
    # `system_info_dict`; the fallback path below still works, and the
    # profile override (next block) eliminates the resulting drift. To
    # activate the full 3 s saving, add ONE line to api_server.py's
    # `_load_spec` dict: `"system_info_dict": <serialized SystemInfo>`.
    sys_info = None
    sys_info_dict = cfg.get("system_info_dict")
    if sys_info_dict:
        sys_info = _reconstruct_system_info(sys_info_dict)
        if sys_info is not None:
            log.info(
                "Reconstructed SystemInfo from parent — skipped detect_all()"
            )

    if sys_info is None:
        det = HardwareDetector(config=cfg.get("hardware_config") or {})
        sys_info = det.detect_all()
        log.info(
            "Ran hardware detection (~3s) — pass `system_info_dict` in "
            "_load_spec to skip this on warm respawns"
        )

    mem_cfg = cfg.get("memory_config") or {}
    total_vram = max(
        sum(g.vram_total_mb for g in sys_info.gpus if getattr(g, "supported", True)) * 1024 * 1024,
        1,
    )
    ram_target = float(mem_cfg.get("ram_utilization_target", 0.75))
    vram_target = float(mem_cfg.get("vram_utilization_target", 0.95))

    vram_mgr = VRAMManager(
        total_vram_bytes=total_vram,
        config={"memory": {"vram_target_utilization": vram_target}},
    )
    ram_mgr = RAMOptimizer(
        max_ram_mb=int(sys_info.ram_total_mb * ram_target),
        compression=bool(mem_cfg.get("compression_enabled", True)),
    )
    swap_mgr = SwapHandler(config={"memory": {
        "ssd_swap_path": mem_cfg.get("ssd_swap_path", "/var/lib/local-llm/swap"),
        "ssd_swap_max_gb": mem_cfg.get("ssd_swap_max_gb", 32),
    }})
    hybrid = HybridMemoryManager(
        vram_bytes=total_vram,
        ram_bytes=sys_info.ram_total_mb * 1024 * 1024,
        config={"memory": {}},
    )

    loader = ModelLoader(
        vram_mgr, ram_mgr, swap_mgr, sys_info.gpus,
        cfg, hybrid=hybrid, system_info=sys_info,
    )

    # ── override loader._profile with the parent's pre-computed one ──
    # Confirmed drift in the committed log: parent ran apu_2gb with
    # target_tps=3.0 / n_batch=2048 / n_parallel=2 / ctx=4096, but workers
    # ran target_tps=8.0 / n_batch=1024 / n_parallel=1 / ctx=8192 — same
    # tier name, different settings. Root cause: `hardware_profiles.overrides.apu_2gb`
    # from config.yaml is applied by ModelLoader's detect_profile() via
    # `self._cfg.get("hardware_profiles", {}).get("overrides", {})`, but
    # `_load_spec` (the worker's `cfg`) does not include `hardware_profiles`
    # — so the worker computes the base tier without the operator's overrides.
    #
    # Replacing `loader._profile` AFTER ModelLoader init and BEFORE
    # `loader.load(mc)` gives the worker the exact profile the parent used.
    # `_load_sync` re-reads self._profile at load time (confirmed: all
    # `self._profile.*` reads in `_load_sync` happen after this point), so
    # the replacement is authoritative for KV plan, thread counts, offload,
    # and target TPS.
    _parent_profile_dict = cfg.get("hardware_profile") or {}
    if _parent_profile_dict:
        try:
            from local_llm_core.engine.hardware.hardware_profiles import HardwareProfile
            _parent_profile = HardwareProfile(**_parent_profile_dict)
            _worker_profile = loader._profile
            # Log drift explicitly so operators can see WHY the override
            # was needed, and confirm the fix corrected it.
            _drift_fields = [
                _f for _f in (
                    "target_tps", "n_batch", "n_ubatch", "context_length",
                    "n_parallel", "kv_cache_quant", "use_mlock",
                )
                if getattr(_worker_profile, _f, None)
                   != getattr(_parent_profile, _f, None)
            ]
            if _drift_fields:
                log.warning(
                    "Profile drift detected and corrected by parent override: %s",
                    ", ".join(
                        f"{_f}={getattr(_worker_profile, _f)!r}->{getattr(_parent_profile, _f)!r}"
                        for _f in _drift_fields
                    ),
                )
            loader._profile = _parent_profile
        except Exception as _exc:
            log.warning(
                "Failed to apply parent hardware_profile override: %s "
                "(worker will use its locally computed profile — drift may persist)",
                _exc,
            )

    model_id = cfg["model_id"]
    model_path = cfg["model_path"]
    mc = ModelConfig(
        model_path=model_path,
        model_id=model_id,
        **({k: v for k, v in (cfg.get("load_kwargs") or {}).items()
            if k in {
                "quantization", "context_length", "num_gpu_layers",
                "n_threads", "use_mmap", "use_mlock", "rope_scaling", "gpu_split",
            }}),
    )
    await loader.load(mc)

    # ── BitNet engine flag from config (not hardcoded default) ───
    # InferenceEngine.__init__ defaults bitnet_enabled to True. Without
    # this passthrough every worker ran `_auto_register_bitnet_from_state`
    # on boot — constructing a BitNetLoader, emitting "BitNetLoader v1.0.0
    # init", and logging the "state file not found" warning — while
    # config.yaml has bitnet.enabled=false. Every worker respawn repeated
    # all of it. Now the worker honours the parent's decision exactly.
    engine_cfg = dict(cfg.get("engine_config") or {})
    _bitnet_enabled = bool((cfg.get("bitnet_config") or {}).get("enabled", False))

    # ── Continuous batching is single_process-only ─────────────────────
    # BatchedInferenceEngine drives one Llama across N concurrent sequences.
    # In worker-subprocess mode, the pool routes ONE request per worker at a
    # time (see SharedWeightProcessPool._dispatch_available); a second submit
    # cannot reach the worker until the first finishes, so batching would sit
    # idle. Run batching in the parent process instead — set
    #   concurrency.mode: single_process
    #   inference.continuous_batching.enabled: true
    # The parent then skips the pool and calls InferenceEngine.generate_stream
    # directly; InferenceEngine dispatches to BatchedInferenceEngine.
    # Fail-safe: if the operator sets both, disable batching HERE with a
    # clear log line rather than let the worker silently ignore it.
    if (engine_cfg.get("continuous_batching") or {}).get("enabled", False):
        log.warning(
            "Worker subprocess: continuous_batching.enabled=true is IGNORED. "
            "Batching requires concurrency.mode: single_process. "
            "Disable batching in worker mode OR set concurrency.mode back to "
            "single_process. Worker will use the single-request path."
        )
        _cb_cfg = dict(engine_cfg.get("continuous_batching") or {})
        _cb_cfg["enabled"] = False
        engine_cfg["continuous_batching"] = _cb_cfg

    log.info(
        "BitNet enabled=%s (from config.yaml bitnet.enabled)",
        _bitnet_enabled,
    )

    # Target TPS from the resolved profile, not from a config lookup —
    # after the profile override above, `loader._profile.target_tps` IS
    # the parent's target.
    _target_tps = (
        float(loader._profile.target_tps)
        if loader._profile is not None
        else float(engine_cfg.get("target_tps", 3.0))
    )

    engine = InferenceEngine(
        loader,
        config=engine_cfg,
        max_batch_size=1,              # single-slot worker
        target_tps=_target_tps,
        thermal_state=None,
        kernels_enabled=False,
        bitnet_enabled=_bitnet_enabled,
    )
    engine.start_background_tasks()
    return engine


def _deserialize_request(payload: dict) -> Any:
    from local_llm_core.engine.inference.inference_engine import InferenceRequest
    return InferenceRequest(
        request_id=payload["request_id"],
        model_id=payload["model_id"],
        prompt=payload["prompt"],
        max_tokens=int(payload.get("max_tokens", 0) or 0),
        temperature=float(payload.get("temperature", 0.7)),
        top_p=float(payload.get("top_p", 0.95)),
        top_k=int(payload.get("top_k", 40)),
        stop=list(payload.get("stop") or []),
        stream=True,
        timeout_s=float(payload.get("timeout_s", 0.0) or 0.0),
        # parent decides via payload. Default True means think
        # tags are stripped before tokens leave the worker — matches the
        # single-process engine's behaviour exactly.
        strip_think=bool(payload.get("strip_think", True)),
        root_id=payload.get("root_id") or "",
        grammar=payload.get("grammar"),
        response_format=payload.get("response_format"),
    )

async def _heartbeat_loop(
    writer: asyncio.StreamWriter,
    write_lock: asyncio.Lock,
    interval_s: float,
) -> None:
    """Liveness signal emitted on the worker's own event loop.

    Why this exists: `supervisor.py:119` kills a worker that hasn't sent
    any frame within `heartbeat_timeout_s`. An idle worker sends no frames
    (`ping` is answered but never sent), so it was killed ~5 s after every
    successful handshake. The committed log records this kill 27 times in
    19 minutes, and 33 distinct worker pids in the same window.

    Running on the event loop means a frozen loop (the class of bug that
    killed the whole supervisor monitor) is detected by the parent: the
    heartbeat stops, the timeout fires, the worker is replaced.

    The payload carries `seq` (monotonic per-process counter) and `ts`
    (wall clock) so the parent can distinguish "heartbeats never
    arrived" from "heartbeats stopped arriving" from "heartbeat
    timestamps have drifted" (clock jump, suspend/resume).

    Serialisation: every outbound frame on the worker goes through the
    same `write_lock`, so heartbeat bytes can never interleave a request
    frame mid-header.
    """
    seq = 0
    while True:
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return
        seq += 1
        try:
            async with write_lock:
                await write_frame(
                    writer,
                    {"t": "hb", "ts": time.time(), "seq": seq},
                    b"",
                )
        except (ConnectionResetError, BrokenPipeError, RuntimeError):
            # Parent socket went away; the serve loop will see EOF.
            return
        except Exception:
            return

async def _serve_requests(
    engine: Any,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    write_lock: asyncio.Lock,
) -> None:
    """Main request loop. Serial: one request in flight, no local queue.

    Every outbound frame — chunk, done, error, cancelled, ready, busy,
    pong, shutdown_ok — goes through `write_lock`. This is what makes
    the concurrent heartbeat task safe: two coroutines can call
    `write_frame` at the same time only if the lock is free between
    awaits, and `write_frame` (from `accel/protocol.py`) issues its
    three `writer.write(...)` calls synchronously before the first
    `await writer.drain()`. Without the lock, a heartbeat could splice
    its header bytes into the middle of a chunk's body.

    Request/terminal correlation: `chunk` and terminal frames carry the
    `rid` of the request they belong to. A stale terminal from a previous
    request cannot corrupt the current one because `_run_one` only writes
    frames tagged with its own `rid`, and the parent's `_dispatch`
    reassociates them by `_active.request_id` (see handle.py).
    """
    current_task: asyncio.Task | None = None
    current_rid: str | None = None

    async def _write(header: dict, body: bytes = b"") -> None:
        async with write_lock:
            await write_frame(writer, header, body)

    async def _run_one(req_payload: dict) -> None:
        nonlocal current_rid
        rid = req_payload["request_id"]
        current_rid = rid
        ir = _deserialize_request(req_payload)
        try:
            async for token in engine.generate_stream(ir):
                if token:
                    await _write(
                        {"t": "chunk", "rid": rid},
                        token.encode("utf-8", "replace"),
                    )
            await _write({"t": "done", "rid": rid}, b"")
        except asyncio.CancelledError:
            await _write({"t": "cancelled", "rid": rid}, b"")
            raise
        except Exception as exc:
            log.exception("worker generate error for %s", rid)
            await _write(
                {"t": "error", "rid": rid},
                str(exc)[:500].encode("utf-8", "replace"),
            )
        finally:
            current_rid = None
            prefix_ids = _resident_prefix_ids(engine, ir.model_id)
            # the `rid` lets the parent correlate this `ready` with
            # the exact request that just finished. Without it, a delayed
            # `ready` from a completed request A could arrive after the
            # pool has already dispatched request B to this worker —
            # handle._dispatch would then clear _busy while B is still
            # generating, and the pool would route yet another request to
            # a busy worker (which fails with "already busy").
            #
            # handle.py's rid-validating dispatch (broader P1-4 fix)
            # activates this check across every frame type. The `rid` here
            # is the missing half that makes it work end-to-end for
            # `ready` frames.
            await _write({
                "t": "ready",
                "rid": rid,
                "resident_prefix_len": len(prefix_ids),
                "resident_prefix_head": prefix_ids[:4096],
            }, b"")

    while True:
        try:
            header, body = await read_frame(reader)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            log.info("parent closed connection")
            return
        except Exception:
            log.exception("bad frame")
            return
        t = header.get("t")
        if t == "submit":
            if current_task is not None and not current_task.done():
                await _write({
                    "t": "error", "rid": header.get("rid", "?"),
                }, b"worker busy")
                continue
            payload = json.loads(body.decode("utf-8"))
            current_task = asyncio.create_task(_run_one(payload))
        elif t == "cancel":
            rid = header.get("rid")
            if current_rid == rid and current_task is not None:
                current_task.cancel()
        elif t == "shutdown":
            if current_task is not None and not current_task.done():
                current_task.cancel()
            return
        elif t == "ping":
            await _write({"t": "pong"}, b"")

def _resident_prefix_ids(engine: Any, model_id: str) -> list:
    """Best-effort: read the live KV prefix the worker holds after the last call."""
    try:
        model = engine._loaded_model_obj(model_id)
        llm = getattr(model, "llm", model)
        ids = getattr(llm, "_input_ids", None)
        if ids is None:
            return []
        return [int(x) for x in ids[:4096]]
    except Exception:
        return []

async def _amain() -> int:
    # ── P1-7: honour LLM_WORKER_LOG_LEVEL (was hardcoded INFO) ──────────
    # The parent sets LLM_WORKER_LOG_LEVEL from its own logging level.
    # The old worker ignored it and always ran basicConfig(INFO), so:
    #   * transformers_compat / hardware_detector / bitnet_loader /
    #     model_loader / inference_engine all dumped 25-35 INFO lines
    #     per boot (the 87%-of-log-lines noise the diagnosis targets),
    #   * every worker respawn re-emitted the same dump,
    #   * operators had no way to quiet it without editing source.
    #
    # New policy:
    #   * Read LLM_WORKER_LOG_LEVEL (default: WARNING).
    #   * Cap library loggers at WARNING — INFO from library modules is
    #     the noise we are removing. DEBUG is respected if the parent
    #     explicitly requested it.
    #   * __main__ lifecycle lines stay at INFO regardless, so "worker
    #     starting", "model loaded", "handshake ok" remain visible.
    _env_level_name = (
                              os.environ.get("LLM_WORKER_LOG_LEVEL", "WARNING") or "WARNING"
                      ).upper().strip() or "WARNING"
    _lib_level = getattr(logging, _env_level_name, logging.WARNING)
    # INFO is the cap for library verbosity unless DEBUG was explicitly
    # requested. This is the noise fix in one line.
    if _lib_level == logging.INFO:
        _lib_level = logging.WARNING
    logging.basicConfig(
        level=_lib_level,
        format="%(asctime)s [worker %(process)d] %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    # __main__ (this module) always emits at INFO — lifecycle lines are
    # how operators see worker readiness and death in the parent log.
    logging.getLogger("__main__").setLevel(logging.INFO)

    port = int(os.environ["LLM_WORKER_PORT"])
    secret = os.environ["LLM_WORKER_SECRET"]
    cfg = json.loads(base64.b64decode(os.environ["LLM_WORKER_CONFIG"]).decode("utf-8"))

    log.info("worker starting: model=%s port=%d", cfg.get("model_id"), port)

    engine = await _load_engine(cfg)
    log.info("worker: model loaded, connecting to parent")

    reader, writer = await asyncio.open_connection("127.0.0.1", port)

    # Single lock serialises every outbound frame on this connection.
    write_lock = asyncio.Lock()

    # Handshake (covered by the lock too, though nothing else is writing yet).
    async with write_lock:
        await write_frame(
            writer,
            {"t": "hello", "secret": secret, "pid": os.getpid()},
            b"",
        )
    hdr, _ = await read_frame(reader)
    if hdr.get("t") != "hello_ok":
        log.error("handshake failed: %r", hdr)
        return 2

    # Heartbeat interval comes from the parent's load_spec.runtime. Parent
    # supplies `heartbeat_interval_s` (default 1.0) and enforces a kill at
    # `heartbeat_timeout_s` (default 15.0) — the interval must be strictly
    # smaller; the supervisor clamps at 3× if misconfigured.
    hb_interval = float(
        (cfg.get("runtime") or {}).get("heartbeat_interval_s", 1.0)
    )
    if hb_interval <= 0:
        hb_interval = 1.0
    hb_task = asyncio.create_task(
        _heartbeat_loop(writer, write_lock, hb_interval),
        name="worker_hb",
    )

    log.info(
        "worker: handshake ok, entering serve loop (hb=%.2fs)",
        hb_interval,
    )
    try:
        await _serve_requests(engine, reader, writer, write_lock)
    finally:
        if hb_task and not hb_task.done():
            hb_task.cancel()
            try:
                await hb_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await engine.shutdown()
        except Exception:
            pass
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
    return 0

def main() -> None:
    try:
        rc = asyncio.run(_amain())
    except KeyboardInterrupt:
        rc = 0
    sys.exit(rc)


if __name__ == "__main__":
    main()
