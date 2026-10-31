from __future__ import annotations

"""
engine/bitnet/bitnet_cpp_backend.py — BitNet.cpp Process Backend
================================================================
v1.0.0 — Production-grade BitNet.cpp process management.

Architecture
────────────
Three backend tiers, tried in order:
  Tier 1  BitNetCppBackend        — llama-server HTTP (preferred, streaming)
  Tier 2  BitNetRunInferenceBackend — run_inference.py subprocess
  Tier 3  None → caller falls back to transformers

Key improvements in v1.0.0
────────────────────────────────────────
PYTHONPATH awareness (root cause fix):
  The original Popen() call inherited the current process's os.environ as-is.
  Two failure modes:
    (a) Stale PYTHONPATH: points to deleted/moved gguf-py → subprocess import errors
    (b) Missing PYTHONPATH: SPECTRE gguf-py path not present → TL1/TL2 missing in any
        Python subprocess (run_inference.py, convert-hf-to-gguf-bitnet.py called later)
  Fix: SubprocessEnvironmentBuilder constructs a CLEAN env dict for every Popen/
  create_subprocess_exec call:
    • C++ binaries (llama-server, llama-cli): strip Python vars that could confuse
      wrapper scripts; add correct PATH / LD_LIBRARY_PATH for GPU libs.
    • Python scripts (run_inference.py): inject SPECTRE gguf-py into PYTHONPATH so
      `import gguf` resolves to the local gguf-py with TL1/TL2 types.

Additional improvements:
  • Port conflict detection + auto-retry with next free port
  • Server health monitor (background thread, auto-restart on crash)
  • atexit + signal handler for clean subprocess teardown
  • Circuit breaker: stops hammering a repeatedly-crashing server
  • ATLAS thread count from state file / bitnet_state.json
  • Metrics: tokens/sec, latency, total calls tracked per instance
  • Structured logging with correlation IDs
  • httpx connection pool reuse across requests
  • SSE parser hardened against malformed frames
  • BitNetRunInferenceBackend: full PYTHONPATH + --gguf-path support
  • create_best_backend() now returns typed BackendResult with reason string
"""

import asyncio
import atexit
import json
import logging
import os
import pwd
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence, Tuple
)

log = logging.getLogger(__name__)

# ── Global registry of live subprocesses for atexit cleanup ──────────────────
_LIVE_PROCS: weakref.WeakSet = weakref.WeakSet()

def _cleanup_all_procs() -> None:
    """atexit handler: terminate all live subprocesses cleanly."""
    for obj in list(_LIVE_PROCS):
        try:
            proc = getattr(obj, "_proc", None)
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass

atexit.register(_cleanup_all_procs)


# =============================================================================
# Runtime configuration
# =============================================================================
@dataclass
class BitNetCppConfig:
    """Full runtime config forwarded to the bitnet.cpp backend."""
    max_tokens:          int   = 512
    temperature:         float = 0.7
    top_p:               float = 0.95
    top_k:               int   = 40
    repetition_penalty:  float = 1.1
    n_threads:           int   = 0       # 0 = auto-detect physical cores
    n_ctx:               int   = 4096    # context window
    # PRODUCTION FIX: n_batch raised from 512 to 2048. Batch size controls
    # the prompt-processing (prefill) phase — larger values allow the model
    # to process more tokens per GEMM call, significantly improving throughput
    # on multi-token prompts (all but the first message in a conversation).
    # 2048 matches the hardware profile's n_batch and is the sweet spot for
    # this APU: big enough to amortize kernel-launch overhead, small enough
    # not to exceed the APU's shared-memory L2 cache per batch.
    n_batch:             int   = 2048
    # PRODUCTION FIX: raised from 60s to 180s. The 60s timeout was firing
    # because llama-cli needs >60s to load the 1.1GB GGUF from NVMe when
    # RAM is at 87%+ (Qwen3-8B also loaded). 180s is still a firm ceiling
    # — genuinely stuck processes are caught here too.
    first_token_timeout: float = 180.0
    total_timeout:       float = 300.0   # total generation timeout
    stop:    Optional[List[str]] = None
    seed:    int   = -1                  # -1 = random
    mirostat:     int   = 0             # 0=off 1=mirostat 2=mirostat_v2
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1


# =============================================================================
# SubprocessEnvironmentBuilder — PYTHONPATH-aware env construction
# =============================================================================
class SubprocessEnvironmentBuilder:
    """
    Builds a clean, correct os.environ dict for every subprocess spawned by
    this module.  Prevents two critical failure modes:

      Stale PYTHONPATH  — old gguf-py path no longer on disk causes ImportError
                          inside run_inference.py and any Python child processes.
      Missing PYTHONPATH — SPECTRE gguf-py path never set → TL1/TL2 missing,
                           causing AttributeError in convert-hf-to-gguf-bitnet.py.

    Two environment profiles:
      for_cpp_binary()   — for llama-server / llama-cli (C++ binaries)
                           Removes PYTHON* vars; adds GPU lib paths to
                           LD_LIBRARY_PATH / DYLD_LIBRARY_PATH.
      for_python_script()— for run_inference.py and any Python subprocesses.
                           Injects SPECTRE gguf-py first in PYTHONPATH;
                           removes broken/stale entries from PYTHONPATH.
    """

    # Python-specific env vars that should NOT be passed to C++ binaries
    # (they can confuse wrapper shell scripts around the binary)
    _PYTHON_VARS_TO_STRIP_FOR_CPP = frozenset({
        "PYTHONSTARTUP", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE",
        "PYTHONOPTIMIZE", "PYTHONINSPECT", "PYTHONVERBOSE",
        "PYTHONHASHSEED",  # keep PYTHONPATH — some llama wrappers are shell scripts
    })

    # GPU library directory suffixes to add to LD_LIBRARY_PATH
    _GPU_LIB_DIRS = [
        "/opt/rocm/lib", "/opt/rocm/lib64",
        "/usr/local/cuda/lib64", "/usr/local/cuda/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
    ]

    @classmethod
    def _read_spectre_gguf_py_path(cls) -> Optional[str]:
        """
        Read the SPECTRE gguf-py path from multiple sources in priority order:
          1. SPECTRE_GGUF_PY_PATH env var (set by install.sh / .llm_env)
          2. bitnet_state.json spectre_gguf_py_path field
          3. PYTHONPATH entries that look like gguf-py directories
        """
        # Source 1: explicit env var
        ep = os.environ.get("SPECTRE_GGUF_PY_PATH", "").strip()
        if ep and Path(ep).is_dir() and (Path(ep) / "gguf" / "__init__.py").exists():
            return ep

        # Source 2: bitnet_state.json
        state = _load_bitnet_state()
        if state:
            sp = state.get("spectre_gguf_py_path", "").strip()
            if sp and Path(sp).is_dir() and (Path(sp) / "gguf" / "__init__.py").exists():
                return sp
            # Derive from bitnet_dir
            bdir = state.get("bitnet_dir", "")
            if bdir:
                derived = Path(bdir) / "3rdparty" / "llama.cpp" / "gguf-py"
                if derived.is_dir() and (derived / "gguf" / "__init__.py").exists():
                    return str(derived)

        # Source 3: scan existing PYTHONPATH entries
        for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
            entry = entry.strip()
            if not entry:
                continue
            p = Path(entry)
            if p.name == "gguf-py" and (p / "gguf" / "__init__.py").exists():
                return str(p)

        return None

    @classmethod
    def _validate_pythonpath_entries(cls, raw: str) -> List[str]:
        """
        Return only the PYTHONPATH entries that actually exist on disk.
        Removes deleted directories (stale entries from old venvs, moved repos).
        """
        validated = []
        for entry in raw.split(os.pathsep):
            entry = entry.strip()
            if not entry:
                continue
            if Path(entry).exists():
                validated.append(entry)
            else:
                log.debug(
                    "SubprocessEnvironmentBuilder: dropping stale PYTHONPATH entry: %s",
                    entry,
                )
        return validated

    @classmethod
    def _build_ld_library_path(cls, extra: Optional[List[str]] = None) -> str:
        """
        Build LD_LIBRARY_PATH / DYLD_LIBRARY_PATH by combining existing value
        with GPU library dirs that actually exist.
        """
        existing = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        dirs = [d for d in existing if d.strip()]
        for candidate in cls._GPU_LIB_DIRS + (extra or []):
            if Path(candidate).is_dir() and candidate not in dirs:
                dirs.append(candidate)
        return os.pathsep.join(dirs)

    @classmethod
    def for_cpp_binary(
        cls,
        extra_path_dirs: Optional[List[str]] = None,
        extra_ld_dirs: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Build environment for llama-server / llama-cli (C++ binaries).

        Strategy:
          • Copy current env
          • Strip vars that could confuse wrapper shell scripts
          • Add GPU library paths to LD_LIBRARY_PATH
          • Add bitnet build bin dirs to PATH
          • Validate (but keep) existing PYTHONPATH in case a wrapper script uses it
        """
        env = dict(os.environ)

        # Strip pure-Python vars not needed by C++ binaries
        for var in cls._PYTHON_VARS_TO_STRIP_FOR_CPP:
            env.pop(var, None)

        # Validate PYTHONPATH (strip stale entries but keep valid ones)
        raw_pp = env.get("PYTHONPATH", "")
        if raw_pp:
            clean = cls._validate_pythonpath_entries(raw_pp)
            if clean:
                env["PYTHONPATH"] = os.pathsep.join(clean)
            else:
                env.pop("PYTHONPATH", None)

        # Augment PATH with bitnet build dirs
        extra_path = extra_path_dirs or []
        if extra_path:
            existing_path = env.get("PATH", "")
            new_dirs = os.pathsep.join(str(d) for d in extra_path)
            env["PATH"] = f"{new_dirs}{os.pathsep}{existing_path}"

        # GPU library paths
        env["LD_LIBRARY_PATH"] = cls._build_ld_library_path(extra_ld_dirs)
        # macOS
        env["DYLD_LIBRARY_PATH"] = env.get("DYLD_LIBRARY_PATH", env["LD_LIBRARY_PATH"])

        # Prevent llama.cpp from opening interactive terminals
        env.setdefault("LLAMA_NO_PROMPT", "1")
        # Disable llama.cpp metal/GPU for BitNet (always CPU kernels)
        # These are already handled by --n-gpu-layers 0 but belt+suspenders
        env.setdefault("LLAMA_NO_METAL", "0")  # keep at 0; let flags control

        return env

    @classmethod
    def for_python_script(
        cls,
        spectre_gguf_py: Optional[str] = None,
        extra_path_dirs: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Build environment for run_inference.py and other Python subprocesses.

        Strategy:
          • Copy current env
          • Validate existing PYTHONPATH (remove stale/missing entries)
          • Inject SPECTRE gguf-py at front of PYTHONPATH (highest priority)
          • This ensures `import gguf` resolves to local gguf-py with TL1/TL2
        """
        env = dict(os.environ)

        # Resolve SPECTRE gguf-py path
        gguf_py = spectre_gguf_py or cls._read_spectre_gguf_py_path()

        # Build clean PYTHONPATH
        raw_pp = env.get("PYTHONPATH", "")
        existing_entries = cls._validate_pythonpath_entries(raw_pp) if raw_pp else []

        # Put SPECTRE gguf-py first (highest priority)
        new_entries: List[str] = []
        if gguf_py and gguf_py not in existing_entries:
            new_entries.append(gguf_py)
            log.debug(
                "SubprocessEnvironmentBuilder: injecting SPECTRE gguf-py: %s", gguf_py
            )
        new_entries.extend(existing_entries)

        if new_entries:
            env["PYTHONPATH"] = os.pathsep.join(new_entries)
        else:
            env.pop("PYTHONPATH", None)

        # Augment PATH if needed
        if extra_path_dirs:
            existing_path = env.get("PATH", "")
            env["PATH"] = os.pathsep.join(str(d) for d in extra_path_dirs) \
                + os.pathsep + existing_path

        # Unbuffer Python stdout/stderr for streaming output
        env["PYTHONUNBUFFERED"] = "1"

        return env


# =============================================================================
# State file helpers
# =============================================================================
def _resolve_real_home() -> Path:
    """
    Resolve the "real" user home directory, robust to running as root under
    systemd (User=root with no Environment="HOME=..." set — systemd then
    gives the process $HOME=/root, the root passwd-db entry, not the
    operator's actual home).

    PRODUCTION FIX: this exact fallback chain previously existed ONLY
    inside _get_state_file_path(), duplicated nowhere else. Meanwhile
    _DEFAULT_BINARY_SEARCH_PATHS (below) used a bare Path.home() with none
    of these fallbacks — so the state-file lookup was hardened against
    this problem but binary discovery, the thing that actually failed in
    production, was not. Extracted once here; both now share it, so a
    future third use-site can't drift out of sync with these two again.

    Priority: REAL_HOME env var (set by install.sh, or manually in a
    systemd unit's Environment=) > SUDO_USER's passwd-db home > Path.home().
    """
    home = Path.home()
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user and sudo_user != "root":
        try:
            home = Path(pwd.getpwnam(sudo_user).pw_dir)
        except KeyError:
            pass
    real_home = os.environ.get("REAL_HOME", "")
    if real_home and Path(real_home).is_dir():
        home = Path(real_home)
    return home


def _get_state_file_path() -> Path:
    """Return the path to bitnet_state.json (contract with bitnet_setup.sh)."""
    return _resolve_real_home() / ".config" / "local-llm" / "bitnet_state.json"


def _load_bitnet_state() -> Optional[Dict[str, Any]]:
    """Load and validate bitnet_state.json. Returns None if invalid/missing."""
    state_path = _get_state_file_path()
    if not state_path.is_file():
        log.debug("BitNet state file not found at %s", state_path)
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            return None
        if not state.get("verify_passed", False):
            log.debug("BitNet state: verify_passed=False — ignoring")
            return None
        # Normalise all path values
        path_keys = [
            "gguf_path", "llama_server_binary", "llama_cli_binary",
            "bitnet_dir", "python_bin", "venv_dir", "run_inference_script",
            "spectre_gguf_py_path",
        ]
        for key in path_keys:
            val = state.get(key)
            if val and isinstance(val, str):
                state[key] = str(Path(val).expanduser().resolve())
        log.info("Loaded BitNet state file: %s", state_path)
        return state
    except Exception as exc:
        log.debug("Failed to load BitNet state file: %s", exc)
        return None


# =============================================================================
# Binary and model discovery
# =============================================================================
def _default_binary_search_paths() -> List[Path]:
    """
    PRODUCTION FIX: was a module-level constant built at import time with a
    bare Path.home() — meaning it was computed once, before _resolve_real_home
    existed as a concept, and never benefited from the REAL_HOME/SUDO_USER
    fallback that _get_state_file_path() already had. Now a function so it
    always reflects the current (correctly-resolved) home, and shares the
    exact same resolution logic as the state file — one source of truth.
    """
    home = _resolve_real_home()
    return [
        Path("build/bin"),
        Path("build/Release/bin"),
        Path("build/Debug/bin"),
        Path("BitNet/build/bin"),
        home / "BitNet" / "build" / "bin",
        home / "AMB-OPT" / "BitNet" / "build" / "bin",
        home / ".local" / "bin",
        Path("/usr/local/bin"),
        Path("/usr/bin"),
        Path("/opt/BitNet/build/bin"),
    ]


# Kept for any external code importing this name directly; now a thin
# wrapper rather than a stale, import-time-frozen list.
_DEFAULT_BINARY_SEARCH_PATHS: List[Path] = _default_binary_search_paths()

_LLAMA_SERVER_NAMES = ["llama-server", "llama-server.exe"]
_LLAMA_CLI_NAMES    = ["llama-cli",    "llama-cli.exe", "main"]


def _find_binary(
    names: List[str],
    extra_paths: Optional[List[Path]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Search for a binary in PATH, state-file paths, and extra search paths.
    Priority: state file > extra_paths > PATH > default search list.
    """
    # Priority 0: state file direct path
    if state:
        for key in ("llama_server_binary", "llama_cli_binary"):
            val = state.get(key, "")
            if val and Path(val).is_file() and os.access(val, os.X_OK):
                p = Path(val)
                if p.name in names or p.stem in [n.split(".")[0] for n in names]:
                    return p

    # Priority 1: extra_paths (callers can inject bitnet_dir/build/bin here)
    for base in (extra_paths or []):
        for name in names:
            candidate = Path(base) / name
            if candidate.exists() and os.access(candidate, os.X_OK):
                return candidate

    # Priority 2: system PATH
    for name in names:
        result = shutil.which(name)
        if result:
            return Path(result)

    # Priority 3: default search list
    # PRODUCTION FIX: call the function (resolves home now) rather than the
    # frozen _DEFAULT_BINARY_SEARCH_PATHS constant (resolved once at import
    # time, before REAL_HOME may even have been read correctly in edge
    # cases such as test harnesses that set env vars after import).
    for base in _default_binary_search_paths():
        for name in names:
            candidate = base / name
            if candidate.exists() and os.access(candidate, os.X_OK):
                return candidate

    return None


def _find_gguf_model(model_dir: str) -> Optional[Path]:
    """
    Locate the best GGUF file for a model directory.
    Priority: i2_s → tl1 → tl2 → embed-quantized → any .gguf
    """
    p = Path(model_dir)
    if p.suffix == ".gguf" and p.is_file():
        return p
    if not p.is_dir():
        return None
    priority_patterns = [
        "*i2_s*embed*.gguf", "*i2_s*.gguf",
        "*tl1*embed*.gguf",  "*tl1*.gguf",
        "*tl2*embed*.gguf",  "*tl2*.gguf",
        "*.gguf",
    ]
    for pattern in priority_patterns:
        matches = sorted(p.glob(pattern), key=lambda f: f.stat().st_size, reverse=True)
        if matches:
            return matches[0]
    return None


def _find_free_port(preferred: int = 18080, max_tries: int = 20) -> int:
    """Find a free TCP port starting at preferred. Returns preferred+N."""
    for offset in range(max_tries):
        port = preferred + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    log.warning("Could not find free port near %d — using %d anyway", preferred, preferred)
    return preferred


def _auto_thread_count() -> int:
    """Return physical CPU core count, preferring psutil over os.cpu_count."""
    # Check ATLAS recommendation from state file first
    state = _load_bitnet_state()
    if state:
        atlas = state.get("atlas_threads")
        if atlas and isinstance(atlas, int) and atlas > 0:
            return atlas
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 4
    except ImportError:
        return os.cpu_count() or 4


# =============================================================================
# Metrics tracker
# =============================================================================
@dataclass
class BackendMetrics:
    """Lightweight per-instance performance metrics."""
    total_calls:       int   = 0
    total_tokens:      int   = 0
    total_errors:      int   = 0
    server_restarts:   int   = 0
    last_latency_ms:   float = 0.0
    avg_tokens_per_sec: float = 0.0
    _latency_window:   List[float] = field(default_factory=list)

    def record_generation(self, tokens: int, elapsed_sec: float) -> None:
        self.total_calls  += 1
        self.total_tokens += tokens
        ms = elapsed_sec * 1000
        self.last_latency_ms = ms
        self._latency_window.append(ms)
        if len(self._latency_window) > 100:
            self._latency_window.pop(0)
        if elapsed_sec > 0 and tokens > 0:
            tps = tokens / elapsed_sec
            # Exponential moving average
            alpha = 0.1
            self.avg_tokens_per_sec = (
                tps if self.avg_tokens_per_sec == 0
                else alpha * tps + (1 - alpha) * self.avg_tokens_per_sec
            )

    def record_error(self) -> None:
        self.total_errors += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls":        self.total_calls,
            "total_tokens":       self.total_tokens,
            "total_errors":       self.total_errors,
            "server_restarts":    self.server_restarts,
            "last_latency_ms":    round(self.last_latency_ms, 1),
            "avg_tokens_per_sec": round(self.avg_tokens_per_sec, 1),
        }


# =============================================================================
# Circuit breaker
# =============================================================================
class CircuitBreaker:
    """
    Stops hammering a repeatedly-crashing server.
    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (probe)
    """
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout:  float = 60.0,
    ) -> None:
        self._state = self.CLOSED
        self._failures = 0
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._open_since: float = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                if time.monotonic() - self._open_since >= self._recovery_timeout:
                    self._state = self.HALF_OPEN
                    log.info("Circuit breaker → HALF_OPEN (probing)")
            return self._state

    def is_open(self) -> bool:
        return self.state == self.OPEN

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = self.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                if self._state != self.OPEN:
                    log.warning(
                        "Circuit breaker OPEN after %d failures — "
                        "pausing for %.0fs",
                        self._failures, self._recovery_timeout,
                    )
                self._state = self.OPEN
                self._open_since = time.monotonic()

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = self.CLOSED


# =============================================================================
# BitNetCliBackend — Tier 0: direct llama-cli, one process per request
# =============================================================================
class BitNetCliBackend:
    """
    Invokes the `llama-cli` binary directly, once per generation request —
    no persistent server process, no run_inference.py indirection layer.

    PRODUCTION RATIONALE: this is now the PRIMARY BitNet backend tier,
    ahead of both llama-server and run_inference.py. Background:

    llama-server (a persistent daemon, formerly Tier 1) shares one mutable
    generation loop across every request (especially acute here under
    --parallel 1, a single shared slot). If that loop gets stuck — a real,
    currently-open class of upstream llama.cpp bug (server freezes/becomes
    unresponsive while the OS process stays alive, particularly on CPU-only
    hardware) — every subsequent request is blocked until something
    notices and restarts it. In practice, across repeated real-world tests
    of this exact setup, llama-server was the one component that
    repeatedly stalled, while every direct llama-cli invocation succeeded
    cleanly and predictably every single time.

    run_inference.py (formerly Tier 2, now Tier 1) is itself just a thin
    Python wrapper that shells out to llama-cli — confirmed directly
    against Microsoft's own source (github.com/microsoft/BitNet/blob/main/
    run_inference.py): it's an argparse layer over the same binary this
    class calls, adding a Python interpreter startup and a second
    subprocess hop for no behavioral benefit. This class talks to the
    exact same llama-cli binary directly.

    A one-process-per-request architecture trades persistent-server
    startup-amortization (no warm KV cache reuse across turns) for
    structural immunity to "stuck shared state" bugs: a hung process here
    affects exactly one request and is simply killed, with zero blast
    radius to anything else — there's no shared mutable generation loop to
    get stuck in the first place. This exact pattern (subprocess.run over
    llama-cli directly, no server) is also independently documented as a
    real-world BitNet production deployment approach.

    Supports the same two usage modes llama-cli itself documents:
      - single-shot text completion (-p PROMPT -n N, the default here —
        matches this server's own architecture, which already reconstructs
        the full multi-turn prompt text per request rather than relying on
        the binary to hold conversation state)
      - conversation mode (-cnv) is NOT used by this class; it's an
        interactive stdin/stdout REPL mode, architecturally a poor fit for
        a one-shot-process-per-request model. Multi-turn behavior is
        achieved the same way every backend in this codebase achieves it:
        the full reconstructed history is passed as one prompt string.
    """

    def __init__(
        self,
        cli_binary:  Path,
        gguf_model:  Path,
        host:        str = "",   # unused; kept for constructor-shape parity
        port:        int = 0,    # unused; kept for constructor-shape parity
        n_threads:   int = 0,
        n_ctx:       int = 4096,
        bitnet_dir:  Optional[Path] = None,
        max_restarts: int = 5,
    ) -> None:
        self.cli_binary  = Path(cli_binary)
        self.gguf_model  = Path(gguf_model)
        self.n_threads   = n_threads or _auto_thread_count()
        self._n_ctx      = n_ctx
        # bitnet_dir is used only to locate build/bin for LD_LIBRARY_PATH /
        # PATH extension (GPU libs, sibling tools) — same purpose as
        # BitNetCppBackend's equivalent, kept consistent across both tiers.
        self.bitnet_dir  = bitnet_dir or self.cli_binary.parent.parent.parent
        self._correlation_id = str(uuid.uuid4())[:8]
        self.metrics  = BackendMetrics()
        self.breaker  = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
        # Not a long-lived process — "started" tracks only the readiness
        # check below, not an actual running subprocess.
        self._verified = False

    @property
    def n_ctx(self) -> int:
        return self._n_ctx

    @n_ctx.setter
    def n_ctx(self, v: int) -> None:
        self._n_ctx = v

    async def start(self, wait_timeout: float = 10.0) -> bool:
        """
        No persistent process to start. Verifies the binary is executable
        and the model file exists, so a misconfiguration is caught
        immediately at load time with a clear error rather than on the
        first user-facing request.
        """
        if self._verified:
            return True
        if not self.cli_binary.is_file() or not os.access(self.cli_binary, os.X_OK):
            log.error(
                "[%s] llama-cli binary not found or not executable: %s",
                self._correlation_id, self.cli_binary,
            )
            return False
        if not self.gguf_model.is_file():
            log.error(
                "[%s] GGUF model file not found: %s",
                self._correlation_id, self.gguf_model,
            )
            return False
        self._verified = True
        return True

    async def shutdown(self) -> None:
        """No persistent process to stop — present for interface parity
        with the other backends (some callers check hasattr(..., 'shutdown'))."""
        self._verified = False

    # Confirmed EOS/EOT tokens for BitNet b1.58-2B-4T (Llama-3 tokenizer):
    # kv 23: EOS = '<|end_of_text|>' (128001)
    # kv 13: EOT = '<|eot_id|>'       (128009)
    # Source: llama_load_print_meta dump from the real binary.
    # These are passed unconditionally via -r so generation halts at the
    # model's natural stop boundary instead of continuing to produce fake
    # follow-on user/assistant turns (the bug that caused 4264 token
    # responses and 300s timeouts).
    _LLAMA3_STOP_TOKENS: List[str] = [
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|end_of_text|>",    # EOS alias
    ]

    # Token strings that should NEVER appear in client-visible output.
    # Llama-3 and BitNet-specific: produced by the model reaching special
    # vocabulary items and by llama-cli's own EOS-display label.
    #
    # SEMANTICS: truncate at the FIRST special token rather than replacing
    # each one independently. Everything after a stop/EOS/EOT token is
    # hallucinated continuation — fake user/assistant turns, repeated
    # context, etc. — that should be discarded in its entirety.
    # The [\s\S]* suffix consumes everything after the matched token.
    _STRIP_TOKENS_RE = re.compile(
        r"(<\|eot_id\|>"
        r"|<\|end_of_text\|>"
        r"|<\|begin_of_text\|>"
        r"|<\|start_header_id\|>[^<]*<\|end_header_id\|>"
        r"|<\|end_header_id\|>"
        r"|<\|end\|>"
        r"|<\|user\|>"
        r"|<\|assistant\|>"
        r"|<\|system\|>"
        r"|\[end of text\]"
        r")[\s\S]*",
        re.DOTALL,
    )

    def _build_cmd(self, prompt: str, cfg: BitNetCppConfig, stop: Optional[List[str]] = None) -> List[str]:
        """Build the llama-cli subprocess command.

        PRODUCTION FIX: previously passed NO stop sequences to llama-cli,
        relying entirely on client-side substring matching. llama-cli's own
        -r / --reverse-prompt flag instructs the binary itself to halt
        generation at that string — verified against the actual llama-cli
        --help output and the binary's own antiprompt implementation, which
        checks antiprompts after every token sampled in both interactive and
        non-interactive modes (the help text's phrase "return control in
        interactive mode" describes what happens AFTER the match, not a
        restriction on when matching occurs).

        Without -r flags, the BitNet model generates past its natural
        <|eot_id|> stop token, then continues to produce fake <|user|> /
        <|assistant|> turns indefinitely — observed directly as 4264-token
        responses and 300-second total_timeout exhaustion.

        PERFORMANCE FLAGS: this round adds several flags that directly
        improve throughput on this 4-core / 8-thread APU deployment,
        each verified against the actual llama-cli --help output:
          -tb N   threads-batch: use ALL logical threads during the
                  prefill/prompt phase (where memory bandwidth matters
                  most), separate from -t which governs auto-regressive
                  token generation (where physical cores matter).
          --prio 2  high process/thread priority — reduces OS scheduling
                  jitter between tokens, improving latency consistency.
          -b N    batch-size: matches cfg.n_batch (now 2048) for aligned
                  prompt processing chunk size.
          --poll 0  disable busy-polling — reduces CPU waste between
                  tokens on a CPU-only model where polling gains nothing.
        """
        n_threads_gen   = max(1, cfg.n_threads or _auto_thread_count())
        n_threads_batch = max(n_threads_gen, min(
            # Use all logical threads for prefill; never exceed 2× physical
            # (hyper-threading gains on memory-bound workloads are modest
            # and adding too many threads causes cache thrashing on small APUs)
            n_threads_gen * 2,
            (os.cpu_count() or n_threads_gen),
        ))
        cmd = [
            str(self.cli_binary),
            "-m", str(self.gguf_model),
            "-p", prompt,
            "-n", str(cfg.max_tokens),
            "-t", str(n_threads_gen),
            "-tb", str(n_threads_batch),     # performance: all logical threads for prefill
            "-b", str(cfg.n_batch),          # performance: larger batch for better prompt throughput
            "-c", str(cfg.n_ctx or self._n_ctx),
            "--temp", str(max(cfg.temperature, 1e-7)),
            "--top-p", str(cfg.top_p),
            "--top-k", str(cfg.top_k),
            "--repeat-penalty", str(cfg.repetition_penalty),
            "--no-display-prompt",
            "--simple-io",
            "--no-perf",
            "--prio", "2",                   # performance: high OS scheduling priority
            "--poll", "0",                   # performance: no busy-polling (CPU-only path)
        ]
        # Confirmed Llama-3 stop tokens — unconditional so BitNet never
        # generates phantom conversation turns regardless of what the
        # server-side caller passes.
        seen_stops: set = set()
        for tok in self._LLAMA3_STOP_TOKENS:
            if tok not in seen_stops:
                cmd += ["-r", tok]
                seen_stops.add(tok)
        # Caller-supplied stop strings (from cfg.stop and the explicit stop
        # argument) — passed to the binary so it can halt at them too rather
        # than only relying on our Python-level substring scan of stdout.
        for s in (list(stop or []) + list(cfg.stop or [])):
            if s and s not in seen_stops:
                cmd += ["-r", s]
                seen_stops.add(s)
        if cfg.seed is not None and cfg.seed != -1:
            cmd += ["-s", str(cfg.seed)]
        return cmd

    async def agenerate(
        self,
        prompt: str,
        cfg: Optional[BitNetCppConfig] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Run llama-cli as a fresh subprocess for this one request, streaming
        stdout incrementally as text is generated.

        PRODUCTION FIX: enforces the SAME real, bounded timeout discipline
        as BitNetCppBackend.agenerate() (idle-stall bound + cumulative
        cfg.total_timeout wall-clock bound, both via asyncio.wait_for on
        this engine's own event loop) — a hung or runaway subprocess here
        is bounded and killed exactly like a hung HTTP stream would be,
        rather than only relying on cfg.first_token_timeout once at the
        very start (the same gap this fixed in the server-tier backend
        applies equally here if not addressed explicitly).

        Client-side stop-string handling: llama-cli's own --reverse-prompt
        only applies in interactive mode (confirmed via its --help text:
        "halt generation at PROMPT, return control in interactive mode"),
        which this class deliberately does not use. Stop strings are
        therefore detected here, against the accumulated output, and the
        subprocess is terminated as soon as one is found.
        """
        if self.breaker.is_open():
            raise RuntimeError(
                f"[{self._correlation_id}] Circuit breaker OPEN — "
                "llama-cli has failed repeatedly"
            )
        if not self._verified:
            ok = await self.start()
            if not ok:
                raise RuntimeError(
                    f"[{self._correlation_id}] llama-cli backend failed verification"
                )

        if cfg is None:
            cfg = BitNetCppConfig()
        effective_stop = list(stop or cfg.stop or [])
        cmd = self._build_cmd(prompt, cfg, stop=effective_stop)

        extra_path_dirs: List[str] = []
        for sub in ("build/bin", "build/Release/bin"):
            d = Path(self.bitnet_dir) / sub
            if d.is_dir():
                extra_path_dirs.append(str(d))
        proc_env = SubprocessEnvironmentBuilder.for_cpp_binary(
            extra_path_dirs=extra_path_dirs or None,
        )

        log.debug("[%s] llama-cli subprocess: %s", self._correlation_id, " ".join(cmd))

        t_start = time.monotonic()
        full_text = ""
        yielded_len = 0
        idle_limit = max(1.0, cfg.first_token_timeout)
        stopped_early = False

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            # PRODUCTION FIX: stderr captured SEPARATELY from stdout (never
            # merged). llama.cpp's own convention — confirmed consistent
            # with a real-world production deployment of this exact
            # binary — is that model-loading diagnostics and perf-timing
            # text go to stderr; only the prompt/completion text goes to
            # stdout. Merging them would corrupt generated output with
            # log lines.
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
            cwd=str(self.bitnet_dir) if Path(self.bitnet_dir).is_dir() else None,
        )

        try:
            assert proc.stdout is not None
            while True:
                elapsed = time.monotonic() - t_start
                remaining_total = cfg.total_timeout - elapsed
                if remaining_total <= 0:
                    raise TimeoutError(
                        f"[{self._correlation_id}] llama-cli generation exceeded "
                        f"total_timeout={cfg.total_timeout:.0f}s "
                        f"({elapsed:.1f}s elapsed)."
                    )
                step_timeout = min(idle_limit, remaining_total)
                try:
                    chunk = await asyncio.wait_for(
                        proc.stdout.read(256), timeout=step_timeout
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(
                        f"[{self._correlation_id}] llama-cli produced no output for "
                        f"{step_timeout:.0f}s ({elapsed:.1f}s elapsed) — the process "
                        "is likely stuck."
                    ) from None
                if not chunk:
                    break  # EOF — process finished naturally

                full_text += chunk.decode("utf-8", errors="replace")

                # Stop-string check against the FULL accumulated text, so a
                # stop sequence split across two reads is still caught.
                cut_at = None
                for s in effective_stop:
                    if s:
                        idx = full_text.find(s, yielded_len)
                        if idx != -1 and (cut_at is None or idx < cut_at):
                            cut_at = idx
                if cut_at is not None:
                    if cut_at > yielded_len:
                        raw = full_text[yielded_len:cut_at]
                        clean = self._STRIP_TOKENS_RE.sub("", raw).rstrip()
                        if clean:
                            yield clean
                    yielded_len = cut_at
                    stopped_early = True
                    break

                if len(full_text) > yielded_len:
                    raw = full_text[yielded_len:]
                    # PRODUCTION FIX: strip llama-cli's own EOS-display
                    # label ("[end of text]") and any Llama-3/BitNet
                    # special tokens that leaked through despite the -r
                    # stop flags — belt-and-suspenders defense so nothing
                    # ever reaches the client regardless of binary version
                    # differences or edge cases in the antiprompt logic.
                    clean = self._STRIP_TOKENS_RE.sub("", raw).rstrip()
                    if clean:
                        yield clean
                    yielded_len = len(full_text)

            if not stopped_early:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                if proc.returncode not in (0, None):
                    stderr_tail = b""
                    if proc.stderr is not None:
                        try:
                            stderr_tail = await asyncio.wait_for(
                                proc.stderr.read(2048), timeout=1.0
                            )
                        except Exception:
                            pass
                    log.warning(
                        "[%s] llama-cli exited with code %s: %s",
                        self._correlation_id, proc.returncode,
                        stderr_tail.decode("utf-8", errors="replace")[-500:],
                    )

            self.breaker.record_success()
            elapsed = time.monotonic() - t_start
            self.metrics.record_generation(len(full_text.split()), elapsed)

        except Exception as exc:
            self.metrics.record_error()
            self.breaker.record_failure()
            log.error(
                "[%s] llama-cli agenerate error: %s",
                self._correlation_id, exc, exc_info=not isinstance(exc, TimeoutError),
            )
            raise
        finally:
            # PRODUCTION FIX: guarantee the subprocess is never leaked,
            # regardless of how this generator exits — normal completion,
            # a stop-string break, a timeout, or the caller cancelling the
            # async generator (e.g. a client disconnect mid-stream).
            if proc.returncode is None:
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2.0)
                    except Exception:
                        pass
                except ProcessLookupError:
                    pass
                except Exception:
                    pass

    def create_completion(
        self,
        prompt:      str,
        max_tokens:  int   = 512,
        temperature: float = 0.7,
        top_p:       float = 0.95,
        top_k:       int   = 40,
        stop:        Optional[List[str]] = None,
        stream:      bool  = True,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        """Synchronous generator (llama.cpp-compatible interface), kept for
        parity with the other backend classes in case any caller relies on
        this shape rather than the native async agenerate()."""
        if not stream:
            raise NotImplementedError(
                "Non-streaming not implemented. Use stream=True or agenerate()."
            )
        cfg = BitNetCppConfig(
            max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, stop=stop, n_ctx=self._n_ctx,
        )

        def _run() -> Iterator[Dict[str, Any]]:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def _collect() -> AsyncIterator[str]:
                    async for tok in self.agenerate(prompt, cfg, stop=stop):
                        yield tok
                gen = _collect()
                while True:
                    try:
                        tok = loop.run_until_complete(gen.__anext__())
                    except StopAsyncIteration:
                        break
                    yield {"choices": [{"text": tok, "finish_reason": None}]}
            finally:
                loop.close()

        return _run()


# =============================================================================
# BitNetCppBackend — Tier 2 (last resort): llama-server HTTP
# =============================================================================
class BitNetCppBackend:
    """
    Manages a `llama-server` subprocess and communicates via its
    OpenAI-compatible HTTP API.

    v8.0.0 additions:
      • SubprocessEnvironmentBuilder for clean env (PYTHONPATH fix)
      • Port conflict detection + auto-retry
      • Background health monitor with auto-restart
      • Circuit breaker
      • BackendMetrics
      • httpx connection pool reuse
      • Hardened SSE parser
      • atexit + SIGTERM cleanup
    """

    def __init__(
        self,
        server_binary:    Path,
        gguf_model:       Path,
        host:             str  = "127.0.0.1",
        port:             int  = 18080,
        n_threads:        int  = 0,
        n_ctx:            int  = 4096,
        n_batch:          int  = 512,
        n_gpu_layers:     int  = 0,          # BitNet: always 0 (CPU SIMD kernels)
        extra_args:       Optional[List[str]] = None,
        auto_find_port:   bool = True,
        health_interval:  float = 10.0,
        max_restarts:     int  = 3,
        circuit_breaker:  Optional[CircuitBreaker] = None,
        bitnet_dir:       Optional[str] = None,
        spectre_gguf_py:  Optional[str] = None,
    ) -> None:
        self.server_binary   = server_binary
        self.gguf_model      = gguf_model
        self.host            = host
        self.n_threads       = n_threads or _auto_thread_count()
        self._n_ctx          = n_ctx
        self.n_batch         = n_batch
        self.n_gpu_layers    = n_gpu_layers
        self.extra_args      = extra_args or []
        self.health_interval = health_interval
        self.max_restarts    = max_restarts
        self.bitnet_dir      = bitnet_dir
        self.spectre_gguf_py = spectre_gguf_py or \
            SubprocessEnvironmentBuilder._read_spectre_gguf_py_path()

        # Port: auto-find free port if requested
        self.port = _find_free_port(port) if auto_find_port else port
        if auto_find_port and self.port != port:
            log.info("Port %d in use — using port %d instead", port, self.port)

        self._base_url = f"http://{self.host}:{self.port}"
        self._proc: Optional[subprocess.Popen] = None
        self._started = False
        self._start_lock = threading.Lock()
        self._restart_count = 0
        self._correlation_id = str(uuid.uuid4())[:8]

        self.metrics  = BackendMetrics()
        self.breaker  = circuit_breaker or CircuitBreaker()
        self._http_client: Optional[Any] = None  # httpx.AsyncClient, lazy init
        self._http_client_loop: Optional[Any] = None  # PRODUCTION FIX: the asyncio
                                                         # loop it was created on —
                                                         # see _get_http_client()

        # Health monitor
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event() if hasattr(asyncio, 'Event') else None

        # Register for atexit cleanup
        _LIVE_PROCS.add(self)

        log.debug(
            "[%s] BitNetCppBackend created: binary=%s model=%s port=%d threads=%d",
            self._correlation_id,
            self.server_binary, self.gguf_model, self.port, self.n_threads,
        )

    # ── Class-level factory ───────────────────────────────────────────────────
    @classmethod
    def auto_detect(
        cls,
        model_dir:           str,
        host:                str  = "127.0.0.1",
        port:                int  = 18080,
        extra_binary_paths:  Optional[List[Path]] = None,
        auto_find_port:      bool = True,
        **kwargs: Any,
    ) -> Optional["BitNetCppBackend"]:
        """
        Return a ready-to-start BitNetCppBackend if everything is in place.

        Discovery priority:
          1. bitnet_state.json (written by bitnet_setup.sh) — zero config
          2. Original search paths (PATH, build dirs, system paths)
        """
        state = _load_bitnet_state()
        binary: Optional[Path] = None
        gguf:   Optional[Path] = None
        bitnet_dir: Optional[str] = None
        spectre_gguf_py: Optional[str] = None

        # ── State-driven discovery ────────────────────────────────────────────
        if state:
            # Binary
            srv_path = state.get("llama_server_binary", "")
            if srv_path and Path(srv_path).is_file() and os.access(srv_path, os.X_OK):
                binary = Path(srv_path)
                log.info("[auto_detect] llama-server from state file: %s", binary)

            # GGUF model
            gguf_p = state.get("gguf_path", "")
            if gguf_p and Path(gguf_p).is_file():
                gguf = Path(gguf_p)
                log.info("[auto_detect] GGUF from state file: %s", gguf)

            # Augment extra paths with state-file build dirs
            bdir = state.get("bitnet_dir", "")
            if bdir:
                bitnet_dir = bdir
                for sub in ["build/bin", "build/Release/bin"]:
                    ep = Path(bdir) / sub
                    if ep.is_dir():
                        extra_binary_paths = (extra_binary_paths or []) + [ep]

            # SPECTRE gguf-py
            sp = state.get("spectre_gguf_py_path", "")
            if sp and Path(sp).is_dir():
                spectre_gguf_py = sp

        # ── Fallback: original search logic ──────────────────────────────────
        if binary is None:
            binary = _find_binary(_LLAMA_SERVER_NAMES, extra_binary_paths, state)
        if gguf is None:
            gguf = _find_gguf_model(model_dir)

        if binary is None:
            log.info(
                "BitNet.cpp llama-server binary not found. Build it once "
                "(one-time, ~2 min): clone https://github.com/microsoft/BitNet, "
                "then inside that directory run: "
                "python setup_env.py --hf-repo microsoft/bitnet-b1.58-2B-4T-gguf -q i2_s"
            )
            return None

        if gguf is None:
            log.info(
                "No GGUF model found in '%s'. Point config.yaml at your "
                "pre-converted .gguf file (e.g. ggml-model-i2_s.gguf), or "
                "convert one yourself with BitNet.cpp's convert script.",
                model_dir,
            )
            return None

        log.info(
            "[auto_detect] BitNet.cpp backend ready — binary=%s  model=%s",
            binary, gguf,
        )
        return cls(
            binary, gguf,
            host=host,
            port=port,
            auto_find_port=auto_find_port,
            bitnet_dir=bitnet_dir,
            spectre_gguf_py=spectre_gguf_py,
            **kwargs,
        )

    # ── Server lifecycle ──────────────────────────────────────────────────────
    async def start(self, wait_timeout: float = 30.0) -> bool:
        """
        Launch llama-server as a subprocess with a clean, PYTHONPATH-aware env.
        Returns True on success.
        """
        with self._start_lock:
            if self._started and self._proc and self._proc.poll() is None:
                return True

            if self.breaker.is_open():
                log.warning(
                    "[%s] Circuit breaker OPEN — refusing to start server",
                    self._correlation_id,
                )
                return False

            cmd = self._build_server_cmd()
            log.info(
                "[%s] Starting BitNet server: %s",
                self._correlation_id,
                " ".join(str(c) for c in cmd),
            )

            # Build clean subprocess environment — C++ binary profile
            # This is the PYTHONPATH fix: we don't blindly inherit os.environ.
            # We clean stale entries and add the correct GPU lib paths.
            extra_path_dirs: List[str] = []
            if self.bitnet_dir:
                for sub in ["build/bin", "build/Release/bin"]:
                    d = Path(self.bitnet_dir) / sub
                    if d.is_dir():
                        extra_path_dirs.append(str(d))

            proc_env = SubprocessEnvironmentBuilder.for_cpp_binary(
                extra_path_dirs=extra_path_dirs or None,
            )

            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=proc_env,   # ← explicit clean env (PYTHONPATH fix)
                    cwd=str(self.bitnet_dir) if self.bitnet_dir else None,
                )
                _LIVE_PROCS.add(self)

                ok = await self._wait_for_ready(wait_timeout)
                if ok:
                    self._started = True
                    self.breaker.record_success()
                    log.info(
                        "[%s] BitNet server ready at %s",
                        self._correlation_id, self._base_url,
                    )
                    # Start background health monitor
                    self._start_health_monitor()
                else:
                    log.error(
                        "[%s] BitNet server failed to start within %.0fs",
                        self._correlation_id, wait_timeout,
                    )
                    stderr_out = ""
                    if self._proc.stderr:
                        try:
                            self._proc.stderr.seek(0)
                            stderr_out = self._proc.stderr.read(500)
                        except Exception:
                            pass
                    if stderr_out:
                        log.error("[%s] Server stderr: %s", self._correlation_id, stderr_out)
                    self._proc.terminate()
                    self.breaker.record_failure()
                return ok

            except FileNotFoundError:
                log.error(
                    "[%s] llama-server binary not found: %s",
                    self._correlation_id, self.server_binary,
                )
                return False
            except Exception as exc:
                log.error(
                    "[%s] Failed to start BitNet server: %s",
                    self._correlation_id, exc, exc_info=True,
                )
                self.breaker.record_failure()
                return False

    async def stop(self) -> None:
        """Terminate llama-server subprocess gracefully."""
        self._stop_health_monitor()
        self._close_http_client()
        if self._proc is not None:
            try:
                self._proc.terminate()
                await asyncio.sleep(0.5)
                if self._proc.poll() is None:
                    self._proc.kill()
                    await asyncio.sleep(0.2)
            except Exception:
                pass
            self._proc = None
        self._started = False

    async def restart(self, wait_timeout: float = 30.0) -> bool:
        """Stop and restart the llama-server process."""
        self.metrics.server_restarts += 1
        log.info(
            "[%s] Restarting BitNet server (restart #%d)",
            self._correlation_id, self.metrics.server_restarts,
        )
        await self.stop()
        await asyncio.sleep(1.0)
        return await self.start(wait_timeout)

    def is_running(self) -> bool:
        return self._started and self._proc is not None and self._proc.poll() is None

    # ── Server command builder ────────────────────────────────────────────────
    def _build_server_cmd(self) -> List[str]:
        cmd = [
            str(self.server_binary),
            "--model",         str(self.gguf_model),
            "--host",          self.host,
            "--port",          str(self.port),
            "--threads",       str(self.n_threads),
            "--ctx-size",      str(self._n_ctx),
            "--batch-size",    str(self.n_batch),
            "--n-gpu-layers",  str(self.n_gpu_layers),
            "--parallel",      "1",       # BitNet: single request at a time
            "--log-disable",
        ]
        cmd.extend(self.extra_args)
        return cmd

    # ── Health / readiness ────────────────────────────────────────────────────
    async def _wait_for_ready(self, timeout: float) -> bool:
        """Poll /health until 200 OK or timeout."""
        try:
            import httpx
        except ImportError:
            log.warning("httpx not installed — assuming server ready after 3s delay")
            await asyncio.sleep(3.0)
            return True

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Check if process died during startup
            if self._proc and self._proc.poll() is not None:
                rc = self._proc.returncode
                log.error(
                    "[%s] llama-server exited during startup (rc=%d)",
                    self._correlation_id, rc,
                )
                return False
            try:
                async with httpx.AsyncClient(timeout=1.0) as client:
                    r = await client.get(f"{self._base_url}/health")
                    if r.status_code == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
        return False

    def _start_health_monitor(self) -> None:
        """Start background asyncio task that polls /health and auto-restarts."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._monitor_task = loop.create_task(self._health_monitor_loop())
        except Exception:
            pass

    def _stop_health_monitor(self) -> None:
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
        self._monitor_task = None

    async def _health_monitor_loop(self) -> None:
        """
        Poll /health at intervals; auto-restart if the server dies OR
        becomes unresponsive while still alive.

        PRODUCTION FIX: previously only acted when the process had
        actually exited (self._proc.poll() is not None). A process that
        is alive but internally stuck — the exact scenario behind the
        multi-minute generation hangs this was built to catch, and a
        real, currently-open class of upstream llama.cpp bug (server
        freezes/becomes unresponsive while the OS process keeps running)
        — fell through every branch here and did nothing: the loop just
        kept silently re-polling forever with no escalation. Now tracks
        CONSECUTIVE health-check failures; after a threshold is reached
        while the process is still alive, it's treated the same as a
        death — restart() is called (which itself stops any existing
        process before respawning, so this is safe even though the old
        process never actually exited on its own). This is the
        proactive half of the fix; agenerate()'s own enforced
        total_timeout (see its docstring) is the reactive half that
        bounds any single in-flight request regardless of what this
        background loop is doing.
        """
        try:
            import httpx
        except ImportError:
            return

        # PRODUCTION FIX: threshold derived from health_interval so the
        # effective detection window stays consistent (~30s) regardless
        # of what health_interval is configured to. 3 consecutive misses
        # avoids reacting to a single transient blip (e.g. the server
        # busy with one slow request under --parallel 1) while still
        # catching a genuine stuck state well before a user-facing
        # request would have to wait out its own full timeout.
        consecutive_failures = 0
        failure_threshold = max(2, round(30.0 / max(self.health_interval, 1.0)))

        while self._started:
            await asyncio.sleep(self.health_interval)
            if not self._started:
                break
            healthy = False
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    r = await client.get(f"{self._base_url}/health")
                    if r.status_code == 200:
                        healthy = True
                    else:
                        log.warning(
                            "[%s] Health check: unexpected status %d",
                            self._correlation_id, r.status_code,
                        )
            except Exception:
                pass

            if healthy:
                consecutive_failures = 0
                continue

            consecutive_failures += 1
            process_died = bool(self._proc and self._proc.poll() is not None)
            process_stuck = (
                not process_died
                and consecutive_failures >= failure_threshold
            )

            if not process_died and not process_stuck:
                # Still within tolerance — log at debug to avoid noise on
                # a single transient miss, keep polling.
                log.debug(
                    "[%s] Health check miss %d/%d (process still alive, "
                    "not yet escalating)",
                    self._correlation_id, consecutive_failures, failure_threshold,
                )
                continue

            if process_died:
                rc = self._proc.returncode
                log.warning(
                    "[%s] llama-server died (rc=%d)",
                    self._correlation_id, rc,
                )
            else:
                log.warning(
                    "[%s] llama-server process is alive but unresponsive "
                    "for %d consecutive health checks (~%.0fs) — treating "
                    "as stuck and restarting",
                    self._correlation_id, consecutive_failures,
                    consecutive_failures * self.health_interval,
                )

            if self._restart_count < self.max_restarts and not self.breaker.is_open():
                self._restart_count += 1
                self._started = False
                consecutive_failures = 0
                ok = await self.restart()
                if not ok:
                    log.error(
                        "[%s] Auto-restart #%d failed — giving up",
                        self._correlation_id, self._restart_count,
                    )
                    self.breaker.record_failure()
                    break
                log.info(
                    "[%s] Auto-restart #%d succeeded",
                    self._correlation_id, self._restart_count,
                )
            else:
                log.error(
                    "[%s] Max restarts (%d) reached or circuit open — "
                    "marking backend unavailable",
                    self._correlation_id, self.max_restarts,
                )
                self.breaker.record_failure()
                break

    # ── HTTP client lifecycle ─────────────────────────────────────────────────
    async def _get_http_client(self) -> Any:
        """
        Return a reusable httpx.AsyncClient (connection pool).

        PRODUCTION FIX (defensive — Tier 2 safety net; the primary fix is
        in inference_engine.py's _is_native_async_backend /
        _bitnet_cpp_dispatch, which ensures agenerate() is only ever
        called from the engine's one persistent event loop): an
        httpx.AsyncClient's internal connection pool, locks, and
        transport are bound to whichever asyncio event loop is running
        when it's created. Reusing it from a DIFFERENT loop doesn't
        raise — it typically hangs forever (awaiting a lock/condition the
        original, now-dead loop will never signal again), which is
        exactly what produced "No tokens for 30.0s — producer stalled":
        the cached client was created on a throwaway per-request loop
        that had already been closed by the time the next request reused
        it. This method now detects that mismatch and transparently
        recreates the client instead.

        Deliberately does NOT attempt an async aclose() on a stale,
        loop-mismatched client — doing so would itself require awaiting
        on primitives bound to the dead loop, i.e. the exact same bug in
        miniature. The reference is simply dropped; httpx's own
        finalizer logs an "unclosed client" warning, which is the
        correct and harmless outcome here (the underlying OS socket is
        cleaned up by the process either way).
        """
        import httpx
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        _stale = (
            self._http_client is not None
            and self._http_client_loop is not None
            and current_loop is not None
            and self._http_client_loop is not current_loop
        )
        if _stale:
            log.debug(
                "[%s] Dropping httpx client bound to a different event "
                "loop than the one currently running (expected if "
                "something still calls agenerate() from a fresh loop)",
                self._correlation_id,
            )
            self._http_client = None

        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=5.0,
                    read=60.0,
                    write=5.0,
                    pool=5.0,
                ),
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                ),
            )
            self._http_client_loop = current_loop
        return self._http_client

    def _close_http_client(self) -> None:
        if self._http_client is not None:
            try:
                # Fire-and-forget close attempt
                import asyncio as _aio
                loop = _aio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._http_client.aclose())
                else:
                    loop.run_until_complete(self._http_client.aclose())
            except Exception:
                pass
            self._http_client = None
            self._http_client_loop = None

    # ── Token streaming ───────────────────────────────────────────────────────
    async def agenerate(
        self,
        prompt: str,
        cfg: Optional[BitNetCppConfig] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream tokens via the OpenAI-compatible /v1/completions endpoint.

        PRODUCTION FIX (real, enforced timeout — closes a regression):
        previously relied SOLELY on httpx's per-chunk `read` timeout
        (Timeout(read=cfg.first_token_timeout)) to bound this call. Two
        problems with that, both now fixed:

        1. httpx's read timeout resets on EVERY byte received, including
           non-token SSE lines (heartbeats, partial frames). A server that
           is internally stuck but still trickling ANY bytes — a real,
           currently-open class of upstream llama.cpp bug (server
           freezes/hangs intermittently, sometimes only during prompt
           processing on CPU-only hardware) — could keep resetting that
           clock indefinitely while never producing a real token,
           defeating the timeout entirely. This was observed directly: a
           7+ minute hang with no exception ever raised.

        2. cfg.total_timeout (declared in BitNetCppConfig, default 300s)
           was never actually enforced anywhere — dead configuration.

        Both are fixed by wrapping each line fetch in `asyncio.wait_for`,
        using two independent clocks evaluated every iteration: a
        per-step idle bound (no data at all for N seconds → stalled) and
        a cumulative wall-clock bound from request start (cfg.total_timeout
        → runaway-but-trickling generation). Both raise a clear,
        diagnosable TimeoutError instead of hanging — this is also what
        makes generate_stream() in inference_engine.py safe to call with
        no executor-thread/AsyncTokenBuffer bridge: the bound now lives at
        the true source instead of relying on a wrapper that was removed.

        Deliberately implemented with asyncio.wait_for on the SAME
        currently-running event loop — NOT a new thread or a fresh event
        loop — so this cannot reintroduce the cross-event-loop httpx
        client bug that _get_http_client()'s loop-tracking already
        guards against.
        """
        if self.breaker.is_open():
            raise RuntimeError(
                f"[{self._correlation_id}] Circuit breaker OPEN — "
                "BitNet server is unavailable"
            )

        if not self._started:
            ok = await self.start()
            if not ok:
                raise RuntimeError(
                    f"[{self._correlation_id}] BitNet server could not be started"
                )

        if cfg is None:
            cfg = BitNetCppConfig()

        effective_stop = list(stop or cfg.stop or [])
        payload = {
            "prompt":         prompt,
            "max_tokens":     cfg.max_tokens,
            "temperature":    max(cfg.temperature, 1e-7),
            "top_p":          cfg.top_p,
            "top_k":          cfg.top_k,
            "repeat_penalty": cfg.repetition_penalty,
            "stream":         True,
            "stop":           effective_stop,
            "seed":           cfg.seed,
        }
        if cfg.mirostat > 0:
            payload["mirostat"]     = cfg.mirostat
            payload["mirostat_tau"] = cfg.mirostat_tau
            payload["mirostat_eta"] = cfg.mirostat_eta

        t_start = time.monotonic()
        token_count = 0
        # PRODUCTION FIX: idle-stall bound, independent of cfg.total_timeout.
        # Uses first_token_timeout as the per-step idle ceiling throughout
        # (not just for the very first token) — a 60s gap between ANY two
        # tokens is just as much a sign of trouble mid-generation as before
        # the first one, and bounding every step this way means a server
        # that goes silent at token 50 is caught exactly as fast as one
        # that goes silent at token 0.
        idle_limit = max(1.0, cfg.first_token_timeout)

        try:
            import httpx
            client = await self._get_http_client()
            async with client.stream(
                "POST",
                f"{self._base_url}/v1/completions",
                json=payload,
                # Connection-level timeout kept as a defensive outer layer
                # (catches a fully dead TCP connection); the real
                # generation-level bound is enforced explicitly below.
                timeout=httpx.Timeout(
                    connect=5.0,
                    read=idle_limit,
                    write=5.0,
                    pool=5.0,
                ),
            ) as response:
                response.raise_for_status()
                line_iter = response.aiter_lines()
                while True:
                    elapsed = time.monotonic() - t_start
                    remaining_total = cfg.total_timeout - elapsed
                    if remaining_total <= 0:
                        raise TimeoutError(
                            f"[{self._correlation_id}] BitNet generation exceeded "
                            f"total_timeout={cfg.total_timeout:.0f}s "
                            f"({token_count} tokens received, {elapsed:.1f}s elapsed). "
                            "The BitNet.cpp server is producing output too slowly "
                            "or has stalled mid-generation."
                        )
                    step_timeout = min(idle_limit, remaining_total)
                    try:
                        line = await asyncio.wait_for(
                            line_iter.__anext__(), timeout=step_timeout
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        if remaining_total <= step_timeout + 0.05:
                            # The total wall-clock budget ran out during this
                            # wait, rather than a genuine idle_limit-length
                            # stall — report it as such so the message
                            # doesn't misleadingly say "stalled for 0s".
                            raise TimeoutError(
                                f"[{self._correlation_id}] BitNet generation exceeded "
                                f"total_timeout={cfg.total_timeout:.0f}s "
                                f"({token_count} tokens received, {elapsed:.1f}s elapsed)."
                            ) from None
                        raise TimeoutError(
                            f"[{self._correlation_id}] BitNet generation stalled — "
                            f"no data received for {step_timeout:.0f}s "
                            f"({token_count} tokens received, {elapsed:.1f}s elapsed). "
                            "This usually means the BitNet.cpp server process is "
                            "still alive but its inference loop is stuck (a known "
                            "upstream llama.cpp issue class, more common under CPU-"
                            "only or memory-constrained conditions); consider "
                            "restarting the BitNet server."
                        ) from None

                    token = _parse_sse_token(line)
                    if token is not None:
                        token_count += 1
                        yield token

            self.breaker.record_success()
            elapsed = time.monotonic() - t_start
            self.metrics.record_generation(token_count, elapsed)

        except ImportError:
            # httpx not installed — fall back to direct subprocess
            async for token in self._subprocess_generate(prompt, cfg, effective_stop):
                token_count += 1
                yield token
            elapsed = time.monotonic() - t_start
            self.metrics.record_generation(token_count, elapsed)

        except Exception as exc:
            self.metrics.record_error()
            self.breaker.record_failure()
            log.error(
                "[%s] agenerate error: %s",
                self._correlation_id, exc, exc_info=True,
            )
            raise

    async def _subprocess_generate(
        self,
        prompt: str,
        cfg: BitNetCppConfig,
        stop: List[str],
    ) -> AsyncIterator[str]:
        """
        Direct llama-cli subprocess fallback (no HTTP).
        Uses clean PYTHONPATH-aware env for the C++ binary.
        """
        cli = _find_binary(_LLAMA_CLI_NAMES, state=_load_bitnet_state())
        if cli is None:
            raise RuntimeError(
                "Neither httpx (for llama-server) nor llama-cli found. "
                "Install httpx: pip install httpx"
            )

        cmd = [
            str(cli),
            "--model",         str(self.gguf_model),
            "--prompt",        prompt,
            "--n-predict",     str(cfg.max_tokens),
            "--threads",       str(self.n_threads),
            "--ctx-size",      str(self._n_ctx),
            "--temp",          str(max(cfg.temperature, 1e-7)),
            "--top-p",         str(cfg.top_p),
            "--top-k",         str(cfg.top_k),
            "--repeat-penalty", str(cfg.repetition_penalty),
            "--no-display-prompt",
            "--log-disable",
            "-ngl", "0",       # BitNet: always CPU
            "-b",   "1",       # batch=1 for ternary kernels
        ]
        for s in stop:
            cmd += ["--reverse-prompt", s]

        # Build clean env for C++ binary (PYTHONPATH fix applies here too)
        proc_env = SubprocessEnvironmentBuilder.for_cpp_binary()

        log.debug("[%s] subprocess generate: %s", self._correlation_id, " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=proc_env,   # ← clean env
            cwd=str(self.bitnet_dir) if self.bitnet_dir else None,
        )
        buffer = b""
        assert proc.stdout is not None
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        proc.stdout.read(64),
                        timeout=cfg.first_token_timeout,
                    )
                except asyncio.TimeoutError:
                    log.warning(
                        "[%s] subprocess: first-token timeout (%.0fs)",
                        self._correlation_id, cfg.first_token_timeout,
                    )
                    break
                if not chunk:
                    break
                buffer += chunk
                while buffer:
                    try:
                        text = buffer.decode("utf-8")
                        yield text
                        buffer = b""
                    except UnicodeDecodeError:
                        if len(buffer) > 8:
                            yield buffer.decode("utf-8", errors="replace")
                            buffer = b""
                        break
        finally:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except Exception:
                pass

    # ── Synchronous generator compatibility ──────────────────────────────────
    def create_completion(
        self,
        prompt:      str,
        max_tokens:  int   = 512,
        temperature: float = 0.7,
        top_p:       float = 0.95,
        top_k:       int   = 40,
        stop:        Optional[List[str]] = None,
        stream:      bool  = True,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        """Synchronous generator (llama.cpp-compatible interface)."""
        if not stream:
            raise NotImplementedError(
                "Non-streaming not implemented. Use stream=True or agenerate()."
            )
        cfg = BitNetCppConfig(
            max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, stop=stop,
        )

        def _run() -> Iterator[Dict[str, Any]]:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def _collect() -> AsyncIterator[str]:
                    async for tok in self.agenerate(prompt, cfg):
                        yield tok
                gen = _collect()
                while True:
                    try:
                        tok = loop.run_until_complete(gen.__anext__())
                        yield {"choices": [{"text": tok}]}
                    except StopAsyncIteration:
                        break
            finally:
                loop.close()

        return _run()

    def __call__(self, prompt: str, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        """
        Alias for create_completion(), matching llama_cpp.Llama's convention
        where the object itself is callable. Without this, inference_engine.py's
        dispatch — which checks `callable(llm)` before routing to the
        llama.cpp-compatible streaming path — would skip this backend entirely
        and silently fall through to the simulation/fake-output branch.
        """
        return self.create_completion(prompt, **kwargs)

    # ── Properties and stats ──────────────────────────────────────────────────
    @property
    def n_ctx(self) -> int:
        return self._n_ctx

    @n_ctx.setter
    def n_ctx(self, v: int) -> None:
        self._n_ctx = v

    def stats(self) -> Dict[str, Any]:
        return {
            "backend":        "bitnet_cpp",
            "server_binary":  str(self.server_binary),
            "gguf_model":     str(self.gguf_model),
            "server_url":     self._base_url,
            "running":        self.is_running(),
            "circuit_state":  self.breaker.state,
            "restart_count":  self._restart_count,
            "correlation_id": self._correlation_id,
            "metrics":        self.metrics.to_dict(),
            "spectre_gguf_py": self.spectre_gguf_py,
        }

    def __repr__(self) -> str:
        return (
            f"BitNetCppBackend(port={self.port}, "
            f"model={self.gguf_model.name}, "
            f"running={self.is_running()})"
        )


# =============================================================================
# Chat wrapper
# =============================================================================
class BitNetCppChatBackend(BitNetCppBackend):
    """
    Chat-formatted wrapper over BitNetCppBackend.
    Formats OpenAI-style message lists into the prompt template expected
    by BitNet b1.58 models.
    """

    SYSTEM_TEMPLATE   = "<|system|>\n{system}\n<|end|>\n"
    USER_TEMPLATE     = "<|user|>\n{user}\n<|end|>\n<|assistant|>\n"
    ASSISTANT_TEMPLATE = "{assistant}\n<|end|>\n"

    DEFAULT_STOP_TOKENS = ["<|end|>", "<|user|>", "<|system|>", "<|endoftext|>"]

    def format_chat_prompt(
        self,
        messages: List[Dict[str, str]],
        system:   Optional[str] = None,
    ) -> str:
        """Format a list of chat messages into a single prompt string."""
        parts: List[str] = []
        if system:
            parts.append(self.SYSTEM_TEMPLATE.format(system=system))
        for msg in messages:
            role    = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(self.SYSTEM_TEMPLATE.format(system=content))
            elif role == "user":
                parts.append(self.USER_TEMPLATE.format(user=content))
            elif role == "assistant":
                parts.append(self.ASSISTANT_TEMPLATE.format(assistant=content))
        return "".join(parts)

    async def achat(
        self,
        messages: List[Dict[str, str]],
        cfg:      Optional[BitNetCppConfig] = None,
        system:   Optional[str] = None,
        stop:     Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async streaming chat interface."""
        prompt = self.format_chat_prompt(messages, system=system)
        combined_stop = list(stop or []) + self.DEFAULT_STOP_TOKENS
        # Deduplicate
        seen: set = set()
        deduped_stop = [s for s in combined_stop if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]
        async for tok in self.agenerate(prompt, cfg, stop=deduped_stop, **kwargs):
            yield tok

    async def chat_complete(
        self,
        messages: List[Dict[str, str]],
        cfg:      Optional[BitNetCppConfig] = None,
        system:   Optional[str] = None,
    ) -> str:
        """Non-streaming: collect full response and return as string."""
        parts: List[str] = []
        async for tok in self.achat(messages, cfg=cfg, system=system):
            parts.append(tok)
        return "".join(parts)


# =============================================================================
# BitNetRunInferenceBackend — Tier 1: run_inference.py subprocess
# =============================================================================
class BitNetRunInferenceBackend:
    """
    Uses run_inference.py as a subprocess. Demoted from primary to Tier 1
    (behind BitNetCliBackend, the new Tier 0): confirmed directly against
    Microsoft's actual run_inference.py source
    (github.com/microsoft/BitNet/blob/main/run_inference.py) that it is
    itself just an argparse wrapper which shells out to llama-cli — so
    going through it adds a Python interpreter startup and a second
    subprocess hop on top of the exact same binary BitNetCliBackend calls
    directly, for no behavioral benefit. Kept as a fallback for
    environments where BitNetCliBackend's binary discovery somehow fails
    but run_inference.py is independently locatable.

    PRODUCTION FIX: the previous version of this class called
    run_inference.py with a "--gguf-path" flag that does not exist in the
    upstream script (confirmed against its actual argparse definition —
    real flags are -m/--model, -n/--n-predict, -p/--prompt, -t/--threads,
    -c/--ctx-size, -temp/--temperature). The previous docstring's claim
    that this was "the new enhanced flag from bitnet_setup.sh v8" could
    not be verified against any source available here, and given it
    directly contradicts the actual upstream script, this backend would
    very likely have failed immediately with an argparse error on any
    real invocation. Now uses "-m", matching the documented, stable
    upstream interface that exists in every standard BitNet.cpp install
    regardless of fork or version.
    """


    def __init__(
        self,
        bitnet_repo_dir: str,
        gguf_model:      str,
        n_threads:       int   = 0,
        n_ctx:           int   = 4096,
        python_bin:      Optional[str] = None,
        spectre_gguf_py: Optional[str] = None,
    ) -> None:
        self.repo_dir        = Path(bitnet_repo_dir)
        self.gguf_model      = Path(gguf_model)
        self.n_threads       = n_threads or _auto_thread_count()
        self._n_ctx          = n_ctx
        self.python_bin      = python_bin or sys.executable
        self.spectre_gguf_py = spectre_gguf_py or \
            SubprocessEnvironmentBuilder._read_spectre_gguf_py_path()
        self._run_script = self.repo_dir / "run_inference.py"
        self._correlation_id = str(uuid.uuid4())[:8]
        self.metrics         = BackendMetrics()

        if not self._run_script.exists():
            raise FileNotFoundError(
                f"run_inference.py not found in {bitnet_repo_dir}. "
                "Ensure BitNet is cloned correctly."
            )

    @property
    def n_ctx(self) -> int:
        return self._n_ctx

    @n_ctx.setter
    def n_ctx(self, v: int) -> None:
        self._n_ctx = v

    @classmethod
    def auto_detect(
        cls,
        model_dir: str,
        **kwargs: Any,
    ) -> Optional["BitNetRunInferenceBackend"]:
        """
        Locate BitNet repo and GGUF model using state file first, then search.
        """
        state = _load_bitnet_state()
        repo_candidates: List[Path] = []
        python_bin: Optional[str] = None
        spectre_gguf_py: Optional[str] = None

        if state:
            bdir = state.get("bitnet_dir", "")
            if bdir:
                repo_candidates.append(Path(bdir))
            pb = state.get("python_bin", "")
            if pb and Path(pb).is_file():
                python_bin = pb
            sp = state.get("spectre_gguf_py_path", "")
            if sp and Path(sp).is_dir():
                spectre_gguf_py = sp

        repo_candidates.extend([
            Path("BitNet"),
            Path.home() / "BitNet",
            Path.home() / "AMB-OPT" / "BitNet",
            Path.home() / "microsoft" / "BitNet",
            Path("/opt/BitNet"),
        ])

        for repo in repo_candidates:
            if not (repo / "run_inference.py").exists():
                continue
            # Prefer GGUF from state file
            if state and state.get("gguf_path"):
                gguf_path = state["gguf_path"]
                if Path(gguf_path).is_file():
                    log.info(
                        "[run_inference] Using repo=%s gguf=%s", repo, gguf_path
                    )
                    return cls(
                        str(repo), gguf_path,
                        python_bin=python_bin,
                        spectre_gguf_py=spectre_gguf_py,
                        **kwargs,
                    )
            # Fallback: search model_dir
            gguf = _find_gguf_model(model_dir)
            if gguf:
                log.info("[run_inference] Using repo=%s gguf=%s", repo, gguf)
                return cls(
                    str(repo), str(gguf),
                    python_bin=python_bin,
                    spectre_gguf_py=spectre_gguf_py,
                    **kwargs,
                )

        return None

    async def agenerate(
        self,
        prompt: str,
        cfg:    Optional[BitNetCppConfig] = None,
        stop:   Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream output from run_inference.py subprocess with PYTHONPATH fix.

        The env passed to create_subprocess_exec is built by
        SubprocessEnvironmentBuilder.for_python_script() which:
          1. Validates existing PYTHONPATH (removes stale entries)
          2. Injects SPECTRE gguf-py at the front
          3. Sets PYTHONUNBUFFERED=1 for streaming output

        PRODUCTION FIX: enforces the same idle-stall + cfg.total_timeout
        wall-clock bound as the other two backend tiers (see
        BitNetCppBackend.agenerate()'s docstring for the full rationale).
        Previously only applied cfg.first_token_timeout, and only as a
        SILENT early break (no exception raised) rather than a real,
        diagnosable error — a stall here would have looked identical to a
        normal short completion to any caller, the same class of bug
        fixed in api_server.py's validate_stream_response() separately.
        """
        if cfg is None:
            cfg = BitNetCppConfig()

        # Build clean Python subprocess env with SPECTRE gguf-py injected
        proc_env = SubprocessEnvironmentBuilder.for_python_script(
            spectre_gguf_py=self.spectre_gguf_py,
        )

        cmd = [
            self.python_bin,
            str(self._run_script),
            # PRODUCTION FIX: "--gguf-path" does not exist in upstream
            # run_inference.py (confirmed against its actual source) — the
            # real, documented, stable flag is "-m"/"--model". Using the
            # flag every standard BitNet.cpp install actually supports,
            # rather than an unverified assumed local patch.
            "-m", str(self.gguf_model),
            "-p", prompt,
            "-n", str(cfg.max_tokens),
            "-t", str(self.n_threads),
            "-temp", str(max(cfg.temperature, 1e-7)),
            "-c", str(cfg.n_ctx),
        ]

        log.debug(
            "[%s] run_inference subprocess: %s",
            self._correlation_id, " ".join(cmd),
        )

        t_start = time.monotonic()
        token_count = 0
        idle_limit = max(1.0, cfg.first_token_timeout)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,   # ← clean PYTHONPATH-aware env (SPECTRE fix)
            cwd=str(self.repo_dir),
        )
        buffer = b""
        assert proc.stdout is not None
        try:
            while True:
                elapsed = time.monotonic() - t_start
                remaining_total = cfg.total_timeout - elapsed
                if remaining_total <= 0:
                    raise TimeoutError(
                        f"[{self._correlation_id}] run_inference.py exceeded "
                        f"total_timeout={cfg.total_timeout:.0f}s "
                        f"({token_count} chunks received, {elapsed:.1f}s elapsed)."
                    )
                step_timeout = min(idle_limit, remaining_total)
                try:
                    chunk = await asyncio.wait_for(
                        proc.stdout.read(128),
                        timeout=step_timeout,
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(
                        f"[{self._correlation_id}] run_inference.py stalled — "
                        f"no output for {step_timeout:.0f}s "
                        f"({token_count} chunks received, {elapsed:.1f}s elapsed)."
                    ) from None
                if not chunk:
                    break
                buffer += chunk
                while buffer:
                    try:
                        text = buffer.decode("utf-8")
                        token_count += 1
                        yield text
                        buffer = b""
                    except UnicodeDecodeError:
                        if len(buffer) > 8:
                            yield buffer.decode("utf-8", errors="replace")
                            buffer = b""
                        break
        finally:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except Exception:
                pass
            elapsed = time.monotonic() - t_start
            self.metrics.record_generation(token_count, elapsed)

    def create_completion(
        self,
        prompt:      str,
        max_tokens:  int   = 512,
        temperature: float = 0.7,
        top_p:       float = 0.95,
        top_k:       int   = 40,
        stop:        Optional[List[str]] = None,
        stream:      bool  = True,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        """Synchronous generator (llama.cpp-compatible interface)."""
        if not stream:
            raise NotImplementedError(
                "Non-streaming not implemented. Use stream=True or agenerate()."
            )
        cfg = BitNetCppConfig(
            max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, stop=stop, n_ctx=self._n_ctx,
        )

        def _run() -> Iterator[Dict[str, Any]]:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def _collect() -> AsyncIterator[str]:
                    async for tok in self.agenerate(prompt, cfg, stop=stop):
                        yield tok
                gen = _collect()
                while True:
                    try:
                        tok = loop.run_until_complete(gen.__anext__())
                        yield {"choices": [{"text": tok}]}
                    except StopAsyncIteration:
                        break
            finally:
                loop.close()

        return _run()

    def __call__(self, prompt: str, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        """Alias for create_completion() — see BitNetCppBackend.__call__ for why
        this is required for inference_engine.py's dispatch to recognize this
        backend as a valid llama.cpp-compatible generator."""
        return self.create_completion(prompt, **kwargs)

    def stats(self) -> Dict[str, Any]:
        return {
            "backend":        "run_inference",
            "repo_dir":       str(self.repo_dir),
            "gguf_model":     str(self.gguf_model),
            "python_bin":     self.python_bin,
            "n_threads":      self.n_threads,
            "correlation_id": self._correlation_id,
            "metrics":        self.metrics.to_dict(),
            "spectre_gguf_py": self.spectre_gguf_py,
        }


# =============================================================================
# Utilities
# =============================================================================
def _parse_sse_token(line: str) -> Optional[str]:
    """
    Parse a single SSE line from llama-server /v1/completions.
    Hardened against:
      - Missing 'data:' prefix (heartbeat lines, comment lines)
      - Malformed JSON (partial frames)
      - Both text and delta content fields (completions vs chat)
    """
    line = line.strip()
    if not line:
        return None
    if line.startswith(":"):
        return None  # SSE comment / heartbeat
    if line == "data: [DONE]":
        return None
    if not line.startswith("data:"):
        return None
    raw = line[5:].strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        log.debug("SSE: malformed JSON frame: %s", raw[:80])
        return None
    choices = obj.get("choices", [])
    if not choices:
        return None
    first = choices[0]
    # /v1/completions returns {text: ...}
    text = first.get("text")
    if text is not None:
        return text if text else None
    # /v1/chat/completions returns {delta: {content: ...}}
    delta = first.get("delta", {})
    content = delta.get("content")
    if content is not None:
        return content if content else None
    return None


# =============================================================================
# BackendResult — typed result from create_best_backend
# =============================================================================
@dataclass
class BackendResult:
    """Result from create_best_backend() with reason string."""
    backend: Optional[object] = None
    tier:    str = "none"       # "server" | "run_inference" | "none"
    reason:  str = ""
    state:   Optional[Dict[str, Any]] = None

    @property
    def available(self) -> bool:
        return self.backend is not None


# =============================================================================
# create_best_backend — public factory
# =============================================================================
def create_best_backend(
    model_dir:           str,
    host:                str  = "127.0.0.1",
    port:                int  = 18080,
    n_threads:           int  = 0,
    n_ctx:               int  = 4096,
    n_gpu_layers:        int  = 0,
    auto_find_port:      bool = True,
    extra_binary_paths:  Optional[List[Path]] = None,
    use_chat_backend:    bool = False,
) -> BackendResult:
    """
    Try to create the best available BitNet backend in tier order:
      Tier 1  BitNetCppBackend (llama-server HTTP, preferred)
      Tier 2  BitNetRunInferenceBackend (run_inference.py, fallback)
      Tier 3  None → BackendResult(backend=None) → caller falls back to transformers

    Returns a BackendResult with .backend, .tier, and .reason.
    """
    state = _load_bitnet_state()

    # Augment extra_binary_paths from state file
    if state and state.get("bitnet_dir"):
        for sub in ["build/bin", "build/Release/bin"]:
            ep = Path(state["bitnet_dir"]) / sub
            if ep.is_dir():
                extra_binary_paths = (extra_binary_paths or []) + [ep]

    # ── Tier 1: llama-server ─────────────────────────────────────────────────
    BcClass = BitNetCppChatBackend if use_chat_backend else BitNetCppBackend
    backend = BcClass.auto_detect(  # type: ignore[attr-defined]
        model_dir,
        host=host,
        port=port,
        n_threads=n_threads,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        auto_find_port=auto_find_port,
        extra_binary_paths=extra_binary_paths,
    )
    if backend is not None:
        return BackendResult(
            backend=backend,
            tier="server",
            reason="llama-server binary and GGUF model found",
            state=state,
        )

    # ── Tier 2: run_inference.py ─────────────────────────────────────────────
    ri_backend = BitNetRunInferenceBackend.auto_detect(
        model_dir, n_threads=n_threads
    )
    if ri_backend is not None:
        return BackendResult(
            backend=ri_backend,
            tier="run_inference",
            reason="run_inference.py found (llama-server not available)",
            state=state,
        )

    # ── Tier 3: nothing found ─────────────────────────────────────────────────
    log.info(
        "No BitNet.cpp binary or run_inference.py found — falling back to "
        "transformers (slower, no 1-bit kernel speedups). To enable them, "
        "build BitNet.cpp once: clone https://github.com/microsoft/BitNet, "
        "then run `python setup_env.py --hf-repo microsoft/bitnet-b1.58-2B-4T-gguf -q i2_s` "
        "inside that directory."
    )
    return BackendResult(
        backend=None,
        tier="none",
        reason=(
            "No llama-server binary or run_inference.py found. "
            "Build BitNet.cpp once via its setup_env.py (see microsoft/BitNet on GitHub)."
        ),
        state=state,
    )


# =============================================================================
# Convenience helpers (public API)
# =============================================================================

def get_state() -> Optional[Dict[str, Any]]:
    """Return the loaded bitnet_state.json dict, or None if unavailable."""
    return _load_bitnet_state()


def is_bitnet_cpp_available() -> bool:
    """
    Quick check: is the llama-server binary present and GGUF model found?
    Does not start a server.
    """
    state = _load_bitnet_state()
    if state:
        srv = state.get("llama_server_binary", "")
        gguf = state.get("gguf_path", "")
        if srv and Path(srv).is_file() and gguf and Path(gguf).is_file():
            return True
    return _find_binary(_LLAMA_SERVER_NAMES, state=state) is not None


def get_recommended_thread_count() -> int:
    """Return the ATLAS-recommended thread count from state file."""
    return _auto_thread_count()
