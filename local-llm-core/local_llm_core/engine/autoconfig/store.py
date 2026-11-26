"""Persistent state for the auto-configuration subsystem.

Two JSON files live in the state directory:

``bench_decode.json``
    One measurement record per model (:class:`BenchStore`).  Records are keyed
    by file name and carry the model fingerprint, the hardware fingerprint and
    the ``llama-cpp-python`` version they were measured with, so a record is
    only trusted on the machine and library build that produced it.

``model_profiles.json``
    The plan computed at start-up: every profile field for every model, the
    resident set and the reasoning behind each value.  The server merges it
    into its in-memory configuration and ``local_llm_core`` can read the same
    file (:func:`plan_entry_for_path`).

Both files are written atomically (temporary file + rename).

Version: 1.0.0
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

__version__ = "1.0.0"
STORE_VERSION = "1.0.0"

log = logging.getLogger(__name__)
_lock = threading.Lock()


# ── locations ───────────────────────────────────────────────────────────
def state_dir() -> Path:
    """State directory: ``$LLM_STATE_DIR``, ``/var/lib/local-llm`` or the XDG state home."""
    explicit = os.environ.get("LLM_STATE_DIR", "").strip()
    if explicit:
        return Path(explicit)
    var_lib = Path("/var/lib/local-llm")
    try:
        if var_lib.is_dir() and os.access(var_lib, os.W_OK | os.R_OK):
            return var_lib
    except OSError:
        pass
    xdg = Path(os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state"))
    return xdg / "local-llm"


def bench_path(directory: Optional[Path] = None) -> Path:
    """Benchmark store location (``$LLM_BENCH_FILE`` overrides)."""
    env = os.environ.get("LLM_BENCH_FILE", "").strip()
    if env:
        return Path(env)
    return Path(directory or state_dir()) / "bench_decode.json"


def plan_path(directory: Optional[Path] = None) -> Path:
    """Computed-profile file location (``$LLM_PLAN_FILE`` overrides)."""
    env = os.environ.get("LLM_PLAN_FILE", "").strip()
    if env:
        return Path(env)
    return Path(directory or state_dir()) / "model_profiles.json"


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
        tmp.replace(path)
        return True
    except OSError as exc:
        log.warning("could not write %s: %s", path, exc)
        return False


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


# ── benchmark store ─────────────────────────────────────────────────────
class BenchStore:
    """Per-model benchmark records."""

    #: A failed calibration is not retried for this long.
    RETRY_FAILED_AFTER_S = 24 * 3600

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else bench_path()

    # reading ---------------------------------------------------------
    def records(self) -> Dict[str, Dict[str, Any]]:
        """All records keyed by file name.

        Understands this store's format and the single-record format written
        by older tools.
        """
        data = _read_json(self.path)
        if not data:
            return {}
        models = data.get("models")
        if isinstance(models, dict):
            return {str(k): v for k, v in models.items() if isinstance(v, dict)}
        if data.get("decode_tps") is not None or data.get("model_path"):
            key = Path(str(data.get("model_path") or "")).name or str(data.get("model_id") or "_legacy")
            return {key: data}
        return {}

    def get(self, file_name: str) -> Optional[Dict[str, Any]]:
        return self.records().get(file_name)

    # decisions -------------------------------------------------------
    @staticmethod
    def matches(rec: Mapping[str, Any], model_fp: str, hardware_fp: str, llama_cpp: str = "") -> bool:
        """True when ``rec`` was measured for this model on this machine and build."""
        if rec.get("fingerprint") and rec["fingerprint"] != model_fp:
            return False
        if rec.get("hardware") and rec["hardware"] != hardware_fp:
            return False
        if llama_cpp and rec.get("llama_cpp") and rec["llama_cpp"] != llama_cpp:
            return False
        return True

    def is_current(self, file_name: str, model_fp: str, hardware_fp: str, llama_cpp: str = "") -> bool:
        """A usable measurement exists and nothing it depends on has changed."""
        rec = self.get(file_name)
        if not rec or not rec.get("fingerprint") or not rec.get("hardware"):
            return False
        if rec.get("failed"):
            return False
        return self.matches(rec, model_fp, hardware_fp, llama_cpp) and float(rec.get("decode_tps", 0) or 0) > 0

    def recently_failed(self, file_name: str, model_fp: str, hardware_fp: str, llama_cpp: str = "") -> bool:
        rec = self.get(file_name)
        if not rec or not rec.get("failed"):
            return False
        if not self.matches(rec, model_fp, hardware_fp, llama_cpp):
            return False
        return (time.time() - float(rec.get("measured_at", 0) or 0)) < self.RETRY_FAILED_AFTER_S

    # writing ---------------------------------------------------------
    def put(self, file_name: str, record: Mapping[str, Any]) -> bool:
        """Insert or replace one model's record, keeping every other model."""
        rec = dict(record)
        rec.setdefault("measured_at", time.time())
        rec.setdefault("hostname", socket.gethostname())
        with _lock:
            store = self.records()
            store[file_name] = rec
            return _atomic_write(self.path, {"version": STORE_VERSION, "models": store})


# ── plan file ───────────────────────────────────────────────────────────
def write_plan(payload: Mapping[str, Any], path: Optional[Path] = None) -> Optional[Path]:
    """Write the computed profiles.  Returns the path or ``None`` on failure."""
    target = Path(path) if path else plan_path()
    body = {"version": STORE_VERSION, "generated_at": time.time(), **payload}
    return target if _atomic_write(target, body) else None


def read_plan(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    return _read_json(Path(path) if path else plan_path())


def plan_entry_for_path(model_path: str, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The computed profile for the model stored at ``model_path`` (exact path, then file name)."""
    data = read_plan(path)
    if not data:
        return None
    models = data.get("models") or {}
    base = Path(model_path).name
    by_name = None
    for entry in models.values():
        ep = str(entry.get("path") or "")
        if ep == model_path:
            return entry
        if Path(ep).name == base:
            by_name = by_name or entry
    return by_name
