#!/usr/bin/env python3
"""
local_llm_server/monitoring/monitor.py — Production Terminal Dashboard v1.0.0
===============================================================================
A next‑generation TUI for monitoring and controlling a Local LLM Cloud server.

All critical bugs from the comprehensive audit have been fixed:
  • Non‑blocking refresh (UI no longer freezes for 12s)
  • Real master‑key authentication (no more stub)
  • Log scroll changes only on the Logs panel
  • VRAM fragmentation and per‑model stats displayed
  • Proper tab overflow handling
  • And many more improvements – see the commit history for details.

Usage:
  python monitor.py                                    # connect to localhost:8080
  python monitor.py --api-key YOUR_KEY
  python monitor.py --url http://192.168.1.10:8080 --api-key KEY
  LLM_API_KEY=your-key LLM_MASTER_KEY=master-key python monitor.py
"""

import argparse
import curses
import json
import locale
import logging
import os
import queue
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait as cf_wait, ALL_COMPLETED
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# ------------------------------------------------------------------------------
# Configuration loading (auto‑discover config.yaml, including server config)
# ------------------------------------------------------------------------------
def _expand_env_vars(value: str) -> str:
    """Expand ``${VAR}`` and ``$VAR`` placeholders from the environment."""
    if not isinstance(value, str):
        return value
    return re.sub(
        r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)",
        lambda m: os.environ.get(m.group(1) or m.group(2), m.group(0)),
        value,
    )

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load config from explicit path, then fall back to common locations."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    candidates = [
        Path.home() / ".llmcloud" / "config.yaml",
        Path.cwd() / "config.yaml",
        Path(__file__).resolve().parent / "config.yaml",
        Path("/etc/local-llm/config.yaml"),
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f) or {}
    return {}

# add config_path parameter to resolvers, check that first.
def _resolve_api_key_from_server_config(config_path: Optional[Path] = None) -> Optional[str]:
    """Try to read `api.key` from the server's config.yaml.
       If config_path is given, check it first; then fall back to common locations.
    """
    candidates = []
    if config_path and config_path.exists():
        candidates.append(config_path)
    candidates.extend([
        Path.cwd() / "config.yaml",
        Path(__file__).resolve().parent / "config.yaml",
        Path.home() / ".config" / "local-llm" / "config.yaml",
        Path("/etc/local-llm/config.yaml"),
    ])
    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate) as f:
                    cfg = yaml.safe_load(f) or {}
                # Walk common key paths, expanding ${ENV_VAR} placeholders
                for path in ("api.key", "api.api_key", "auth.api_key", "security.api_key", "api_key"):
                    node = cfg
                    for part in path.split("."):
                        if not isinstance(node, dict):
                            node = None
                            break
                        node = node.get(part)
                    if node and isinstance(node, str):
                        expanded = _expand_env_vars(node)
                        if expanded and not expanded.startswith("${"):
                            return expanded
            except Exception as e:
                # log the exception via _debug so it's not silently swallowed
                _debug(f"_resolve_api_key_from_server_config: {e}")
                continue
    return None


def _resolve_master_key_from_server_config(config_path: Optional[Path] = None) -> Optional[str]:
    """Try to read `api.master_key` from the server's config.yaml.
       Mirrors _resolve_api_key_from_server_config().
    """
    candidates = []
    if config_path and config_path.exists():
        candidates.append(config_path)
    candidates.extend([
        Path.cwd() / "config.yaml",
        Path(__file__).resolve().parent / "config.yaml",
        Path.home() / ".config" / "local-llm" / "config.yaml",
        Path("/etc/local-llm/config.yaml"),
    ])
    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate) as f:
                    cfg = yaml.safe_load(f) or {}
                for path in ("api.master_key", "api.masterkey", "security.master_key"):
                    node = cfg
                    for part in path.split("."):
                        if not isinstance(node, dict):
                            node = None
                            break
                        node = node.get(part)
                    if node and isinstance(node, str):
                        expanded = _expand_env_vars(node)
                        if expanded and not expanded.startswith("${"):
                            return expanded
            except Exception as e:
                # 🔥 ISSUE 3 FIX: log the exception
                _debug(f"_resolve_master_key_from_server_config: {e}")
                continue
    return None

# helper to get nested value with same path logic
def _get_nested_config_value(cfg: dict, *paths: str) -> Optional[str]:
    for path in paths:
        node = cfg
        for part in path.split("."):
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(part)
        if node and isinstance(node, str):
            expanded = _expand_env_vars(node)
            if expanded and not expanded.startswith("${"):
                return expanded
    return None

DEFAULTS = {
    "server_url": "http://localhost:8080",
    "refresh_interval": 2.0,
    "api_key": "",
    "master_key": "",
    "debug": False,
    "log_buffer_size": 500,
    "metrics_history_size": 150,
    "tui": {
        "fps": 10,
        "color_theme": "dark",
        "show_hints": True,
        "predictive_alerts": True,
        "adaptive_polling": True,
        "log_filter_enabled": True,
    },
}

# ------------------------------------------------------------------------------
# Debug logging
# ------------------------------------------------------------------------------
DEBUG_FILE = "/tmp/tui_debug.log"
_debug_logger = logging.getLogger("tui_debug")
_debug_logger.setLevel(logging.DEBUG)
if not _debug_logger.handlers:
    fh = logging.FileHandler(DEBUG_FILE, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _debug_logger.addHandler(fh)

def _debug(msg: str, *args):
    try:
        _debug_logger.debug(msg, *args)
    except Exception:
        pass

# ------------------------------------------------------------------------------
# Locale
# ------------------------------------------------------------------------------
try:
    locale.setlocale(locale.LC_ALL, "")
except locale.Error:
    pass

# ------------------------------------------------------------------------------
# Display width helpers (with fallback)
# ------------------------------------------------------------------------------
try:
    from wcwidth import wcswidth, wcwidth

    def _displen(s: str) -> int:
        w = wcswidth(s)
        return w if w >= 0 else len(s)

    def _cw(c: str) -> int:
        w = wcwidth(c)
        return max(1, w) if w >= 0 else 1
except ImportError:
    import unicodedata

    def _displen(s: str) -> int:
        return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)

    def _cw(c: str) -> int:
        return 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1

def _clip(s: str, cols: int) -> str:
    if cols <= 0:
        return ""
    out, acc = [], 0
    for ch in s:
        w = _cw(ch)
        if acc + w > cols:
            break
        out.append(ch)
        acc += w
    return "".join(out)

def _cx(container_w: int, text: str) -> int:
    return max(0, (container_w - _displen(text)) // 2)

# ------------------------------------------------------------------------------
# Color definitions
# ------------------------------------------------------------------------------
_C_HDR = 1
_C_BDR = 2
_C_TTL = 3
_C_OK = 4
_C_WARN = 5
_C_ERR = 6
_C_DIM = 7
_C_BOK = 8
_C_BWN = 9
_C_BCR = 10
_C_ACC = 11
_C_KEY = 12
_C_MDL = 13
_C_LBL = 14
_C_VAL = 15
_C_INV = 16
_C_SEL = 17
_C_GRAD = 18
_C_PRED = 19

_THEME = "dark"

def _init_colors():
    global _THEME
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except Exception:
        bg = curses.COLOR_BLACK
    fg_map = {}
    if _THEME == "light":
        fg_map = {_C_HDR: curses.COLOR_WHITE, _C_TTL: curses.COLOR_BLACK, _C_OK: curses.COLOR_GREEN,
                  _C_WARN: curses.COLOR_YELLOW, _C_ERR: curses.COLOR_RED, _C_DIM: curses.COLOR_BLACK,
                  _C_KEY: curses.COLOR_BLUE, _C_MDL: curses.COLOR_BLUE, _C_LBL: curses.COLOR_BLACK,
                  _C_VAL: curses.COLOR_GREEN, _C_ACC: curses.COLOR_MAGENTA, _C_GRAD: curses.COLOR_MAGENTA,
                  _C_PRED: curses.COLOR_RED}
    else:
        fg_map = {_C_HDR: curses.COLOR_BLACK, _C_TTL: curses.COLOR_CYAN, _C_OK: curses.COLOR_GREEN,
                  _C_WARN: curses.COLOR_YELLOW, _C_ERR: curses.COLOR_RED, _C_DIM: curses.COLOR_WHITE,
                  _C_KEY: curses.COLOR_CYAN, _C_MDL: curses.COLOR_BLUE, _C_LBL: curses.COLOR_WHITE,
                  _C_VAL: curses.COLOR_GREEN, _C_ACC: curses.COLOR_MAGENTA, _C_GRAD: curses.COLOR_MAGENTA,
                  _C_PRED: curses.COLOR_RED}
    for idx, fg, bgg in [
        (_C_HDR, curses.COLOR_WHITE, curses.COLOR_CYAN),
        (_C_BDR, curses.COLOR_CYAN, bg),
        (_C_TTL, curses.COLOR_CYAN, bg),
        (_C_OK, curses.COLOR_GREEN, bg),
        (_C_WARN, curses.COLOR_YELLOW, bg),
        (_C_ERR, curses.COLOR_RED, bg),
        (_C_DIM, curses.COLOR_WHITE, bg),
        (_C_BOK, curses.COLOR_GREEN, bg),
        (_C_BWN, curses.COLOR_YELLOW, bg),
        (_C_BCR, curses.COLOR_RED, bg),
        (_C_ACC, curses.COLOR_MAGENTA, bg),
        (_C_KEY, curses.COLOR_CYAN, bg),
        (_C_MDL, curses.COLOR_BLUE, bg),
        (_C_LBL, curses.COLOR_WHITE, bg),
        (_C_VAL, curses.COLOR_GREEN, bg),
        (_C_INV, curses.COLOR_BLACK, curses.COLOR_WHITE),
        (_C_SEL, curses.COLOR_BLACK, curses.COLOR_GREEN),
        (_C_GRAD, curses.COLOR_MAGENTA, bg),
        (_C_PRED, curses.COLOR_RED, bg),
    ]:
        try:
            curses.init_pair(idx, fg, bgg)
        except Exception:
            pass

def _cp(n: int, bold: bool = False, rev: bool = False) -> int:
    attr = curses.color_pair(n)
    if bold:
        attr |= curses.A_BOLD
    if rev:
        attr |= curses.A_REVERSE
    return attr

def _safe_addstr(w, y: int, x: int, text: str, attr: int = 0):
    try:
        h, ww = w.getmaxyx()
        if y < 0 or y >= h or x < 0 or x >= ww:
            return
        room = max(0, ww - x - 1)
        if room > 0 and text:
            w.addstr(y, x, _clip(text, room), attr)
    except Exception:
        pass

def _draw_box(w, y: int, x: int, h: int, ww: int, title: str = "", color: int = _C_BDR):
    try:
        a = _cp(color, bold=True)
        _safe_addstr(w, y, x, "╔", a)
        _safe_addstr(w, y, x + ww - 1, "╗", a)
        _safe_addstr(w, y + h - 1, x, "╚", a)
        _safe_addstr(w, y + h - 1, x + ww - 1, "╝", a)
        for i in range(1, ww - 1):
            _safe_addstr(w, y, x + i, "═", a)
            _safe_addstr(w, y + h - 1, x + i, "═", a)
        for i in range(1, h - 1):
            _safe_addstr(w, y + i, x, "║", a)
            _safe_addstr(w, y + i, x + ww - 1, "║", a)
        if title:
            t = f" {title} "
            tx = x + max(1, _cx(ww, t))
            _safe_addstr(w, y, tx, t, _cp(_C_TTL, bold=True))
    except Exception:
        pass

def _draw_hline(w, y: int, x: int, ww: int, title: str = "", color: int = _C_BDR):
    try:
        a = _cp(color, bold=True)
        line = f"── {title} " if title else "─"
        pad = "─" * max(0, ww - x - _displen(line) - 2)
        _safe_addstr(w, y, x, line + pad, a)
    except Exception:
        pass

_USE_UNICODE_BARS = True
try:
    "█".encode(sys.stdout.encoding)
except (UnicodeEncodeError, AttributeError):
    _USE_UNICODE_BARS = False

def _draw_bar(w, y: int, x: int, width: int, pct: float, suffix: str = "", label: str = ""):
    try:
        pct = max(0.0, min(100.0, float(pct)))
        filled = int(width * pct / 100)
        col = _C_BCR if pct >= 90 else (_C_BWN if pct >= 70 else _C_BOK)
        if _USE_UNICODE_BARS:
            bar = "█" * filled + "░" * (width - filled)
        else:
            bar = "#" * filled + "-" * (width - filled)
        if label:
            _safe_addstr(w, y, x - _displen(label) - 1, label, _cp(_C_LBL))
        _safe_addstr(w, y, x, bar, _cp(col))
        if suffix:
            _safe_addstr(w, y, x + width + 1, suffix, _cp(_C_VAL, bold=True))
    except Exception:
        pass

def _draw_kv(w, y: int, x: int, label: str, value: str, lw: int = 22, vc: int = _C_VAL):
    try:
        _safe_addstr(w, y, x, f"{label:<{lw}}", _cp(_C_LBL))
        _safe_addstr(w, y, x + lw, value, _cp(vc, bold=True))
    except Exception:
        pass

def _draw_spark(w, y: int, x: int, width: int, values: list):
    try:
        bars = " ▁▂▃▄▅▆▇█" if _USE_UNICODE_BARS else " .-+*#@"
        if not values:
            return
        vmax = max((float(v) for v in values), default=1.0)
        vmax = max(vmax, 1.0)
        for i, v in enumerate(list(values)[-width:]):
            idx = min(int(float(v) / vmax * (len(bars) - 1)), len(bars) - 1)
            _safe_addstr(w, y, x + i, bars[idx], _cp(_C_ACC))
    except Exception:
        pass

# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------
def _f(v: Any, default: float = 0.0, lo: float = 0.0, hi: Optional[float] = None) -> float:
    try:
        r = float(v)
        if r != r or r == float("inf") or r == float("-inf"):
            return default
        r = max(r, lo)
        if hi is not None:
            r = min(r, hi)
        return r
    except Exception:
        return default

def _i(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default

def _b(v: Any, default: bool = False) -> bool:
    try:
        return bool(v)
    except Exception:
        return default

def _nested(d: Any, *keys, default=None):
    try:
        for k in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(k, default)
        return d
    except Exception:
        return default

# ------------------------------------------------------------------------------
# HTTP client with authentication
# ------------------------------------------------------------------------------
class APIClient:
    def __init__(self, base_url: str, api_key: str = "", master_key: str = "", debug: bool = False):
        self.base = base_url.rstrip("/")
        self.api_key = api_key
        self.master_key = master_key
        self.debug = debug
        self._lock = threading.Lock()
        self._auth_lock = threading.Lock()
        self.last_auth_error: Optional[int] = None  # 401/403 if last request failed auth

    def _headers(self, use_master: bool = False, extra: Optional[dict] = None) -> dict:
        h = {
            "Accept": "application/json",
            # Disable HTTP keep-alive: the monitor's 8-thread pool sends
            # concurrent requests; stale pooled connections cause uvicorn to
            # log "Invalid HTTP request received" in bursts of 8 when it
            # receives data on a connection it already closed server-side.
            "Connection": "close",
        }
        if not use_master and self.api_key:
            h["X-API-Key"] = self.api_key
        if use_master and self.master_key:
            h["X-Master-Key"] = self.master_key
        if extra:
            h.update(extra)
        return h

    def request(self, path: str, method: str = "GET", data: Optional[dict] = None,
                timeout: int = 30, use_master: bool = False) -> Optional[dict]:
        url = f"{self.base}{path}"
        body = json.dumps(data).encode() if data else None
        headers = self._headers(use_master, {"Content-Type": "application/json"} if data else None)
        delay = 0.5
        max_sleep = 5.0
        for attempt in range(1, 4):
            try:
                if self.debug:
                    _debug(f"REQ {method} {url} (attempt {attempt})")
                req = urllib.request.Request(url, data=body, headers=headers, method=method)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    response_data = resp.read().decode('utf-8')
                    # Success — clear any prior auth-failure flag for this client.
                    with self._auth_lock:
                        self.last_auth_error = None
                    return json.loads(response_data) if response_data else {}
            except urllib.error.HTTPError as e:
                if self.debug:
                    _debug(f"HTTPError {e.code} {e.reason}")
                if e.code in (401, 403):
                    # Record the auth failure so the UI can show a clear banner
                    # instead of leaving every panel silently stuck on
                    # placeholder text with no indication of why ("detecting
                    # hardware…" forever looks identical to a real hang).
                    with self._auth_lock:
                        self.last_auth_error = e.code
                    return None
                if e.code == 429:
                    retry_after = float(e.headers.get("Retry-After", delay * 2))
                    time.sleep(min(retry_after, max_sleep))
                    continue
                if e.code >= 500 and attempt < 3:
                    time.sleep(min(delay, max_sleep))
                    delay = min(delay * 2, max_sleep)
                    continue
                return None
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if self.debug:
                    _debug(f"Network error: {e}")
                if attempt < 3:
                    time.sleep(min(delay, max_sleep))
                    delay = min(delay * 2, max_sleep)
                    continue
                return None
            except Exception as e:
                if self.debug:
                    _debug(f"Unexpected error: {e}")
                return None
        return None

    def get(self, path: str, timeout: int = 5, use_master: bool = False) -> Optional[dict]:
        return self.request(path, "GET", timeout=timeout, use_master=use_master)

    def post(self, path: str, payload: dict, timeout: int = 120, use_master: bool = False) -> Optional[dict]:
        return self.request(path, "POST", data=payload, timeout=timeout, use_master=use_master)

    def delete(self, path: str, timeout: int = 30, use_master: bool = False) -> Optional[dict]:
        return self.request(path, "DELETE", timeout=timeout, use_master=use_master)

# ------------------------------------------------------------------------------
# Data model and polling (with thread‑safe improvements)
# ------------------------------------------------------------------------------
_MAX_DISPLAYED_ERRORS = 9999

class ServerData:
    __slots__ = ("lock", "health", "system", "metrics", "models", "drive", "perf",
                 "cluster", "profiles", "last_ok", "errors", "start_ts",
                 "tps_hist", "ttft_hist", "vram_hist", "ram_hist", "cpu_hist", "req_hist",
                 "keys", "requests", "ranking", "bitnet_models", "predict_alerts",
                 "poller_last_active")

    def __init__(self):
        self.lock = threading.Lock()
        self.health = None
        self.system = None
        self.metrics = None
        self.models = None
        self.drive = None
        self.perf = None
        self.cluster = None
        self.profiles = []
        self.last_ok = 0.0
        self.errors = 0
        self.start_ts = time.time()
        ml = 150
        self.tps_hist = deque(maxlen=ml)
        self.ttft_hist = deque(maxlen=ml)
        self.vram_hist = deque(maxlen=ml)
        self.ram_hist = deque(maxlen=ml)
        self.cpu_hist = deque(maxlen=ml)
        self.req_hist = deque(maxlen=ml)
        self.keys = []
        self.requests = []
        # ranking initialised as list, not dict
        self.ranking = []
        self.bitnet_models = []
        self.predict_alerts = []
        self.poller_last_active = time.time()

class Poller(threading.Thread):
    def __init__(self, client: APIClient, data: ServerData, interval: float = 2.0):
        super().__init__(daemon=True, name="poller")
        self.client = client
        self.data = data
        self.interval = interval
        self._stop = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="poll_ep")

        # Endpoint definition: (name, path, request_timeout_s, poll_every_N_cycles)
        # poll_every_N_cycles=1  → every cycle (fast-changing: metrics, models, system)
        # poll_every_N_cycles=5  → every 5 cycles (~10s at 2s interval)
        # poll_every_N_cycles=15 → every 15 cycles (~30s)
        # poll_every_N_cycles=30 → every 30 cycles (~60s) — slow-changing endpoints
        self._endpoints = [
            ("system",   "/v1/system",        2,   1),    # hardware/GPU info — fast
            ("metrics",  "/v1/metrics",        2,   1),    # TPS/VRAM/RAM — fast
            ("models",   "/v1/models",         2,   1),    # loaded models — fast
            ("drive",    "/v1/drive/stats",    2,   2),    # AIMD/drive — medium
            ("perf",     "/v1/performance",    2,   2),    # SLA metrics — medium
            ("cluster",  "/v1/cluster",        5,   5),    # cluster — slow
            ("profiles", "/v1/config/models",  2,  15),   # profiles rarely change
            # Keys list: polled very infrequently (30 cycles ≈ 60s at 2s interval).
            # ?quiet=true suppresses server-side INFO audit log on each poll.
            ("keys",     "/admin/keys/list?quiet=true", 30, 30),
            ("requests", "/admin/requests",    2,   3),    # request log — medium
            ("ranking",  "/v1/telemetry",      2,   5),    # ranking — slow
        ]
        # Cycle counter per endpoint; counts up and polls when counter % poll_every == 0
        self._endpoint_cycles: dict = {name: 0 for name, *_ in self._endpoints}
        self._cycle = 0

    def stop(self):
        self._stop.set()
        self._pool.shutdown(wait=False)

    def run(self):
        backoff = self.interval
        max_backoff = 30.0
        consec_errors = 0
        while not self._stop.is_set():
            try:
                self._poll()
                consec_errors = 0
                backoff = self.interval
                with self.data.lock:
                    self.data.poller_last_active = time.time()
            except Exception as e:
                consec_errors += 1
                backoff = min(max_backoff, self.interval * (2 ** min(consec_errors, 10)))
                _debug(f"poll error: {e}")
            self._stop.wait(backoff)

    def _poll(self):
        self._cycle += 1
        health = self.client.get("/health", timeout=2)
        if health is None:
            with self.data.lock:
                self.data.errors = min(self.data.errors + 1, _MAX_DISPLAYED_ERRORS)
                # 🔥 ISSUE 5 FIX: clear health state so online status is false
                self.data.health = None
            return
        futures = {}
        for ep_tuple in self._endpoints:
            name, path, to, poll_every = ep_tuple
            # Skip this endpoint if it's not due yet this cycle
            if self._cycle % poll_every != 0:
                continue
            if name in ("keys", "requests") and not self.client.master_key:
                continue
            use_master = name in ("keys", "requests")
            futures[self._pool.submit(self.client.get, path, timeout=to, use_master=use_master)] = name

        done, not_done = cf_wait(futures, timeout=12.0, return_when=ALL_COMPLETED)
        for fut in not_done:
            fut.cancel()
        results = {}
        for fut in done:
            name = futures[fut]
            try:
                results[name] = fut.result()
            except Exception:
                results[name] = None
        for fut in not_done:
            name = futures[fut]
            results[name] = None

        with self.data.lock:
            self.data.health = health
            # ── Only update a field when its endpoint was polled this cycle ──
            # results only contains names that were submitted to the thread pool
            # this cycle. Absent keys mean "not due yet" (rate-limited), NOT
            # "server returned nothing". Assigning None to self.data.X would
            # clear existing data and cause UI flicker every other cycle.
            if results.get("system")  is not None: self.data.system  = results["system"]
            if results.get("metrics") is not None: self.data.metrics = results["metrics"]
            if results.get("models")  is not None: self.data.models  = results["models"]
            if results.get("drive")   is not None: self.data.drive   = results["drive"]
            if results.get("perf")    is not None: self.data.perf    = results["perf"]
            if results.get("cluster") is not None: self.data.cluster = results["cluster"]
            pr = results.get("profiles")
            if pr:
                self.data.profiles = pr.get("profiles", [])
            keys_res = results.get("keys")
            if keys_res:
                self.data.keys = keys_res.get("keys", [])
            req_res = results.get("requests")
            if req_res and isinstance(req_res, dict):
                self.data.requests = req_res.get("requests", [])
            ranking_res = results.get("ranking")
            if ranking_res:
                self.data.ranking = ranking_res.get("models", [])
            self.data.last_ok = time.time()
            self.data.errors = 0
            m = self.data.metrics or {}
            eng = m.get("engine", {})
            sys_ = m.get("system", {})
            vram = m.get("vram", {})
            self.data.tps_hist.append(min(float(eng.get("avg_tps", 0)), 500.0))
            self.data.ttft_hist.append(float(eng.get("avg_ttft_ms", 0)))
            self.data.vram_hist.append(float(vram.get("utilization_pct", 0)))
            self.data.ram_hist.append(float(sys_.get("ram_pct", 0)))
            self.data.cpu_hist.append(float(sys_.get("cpu_pct", 0)))
            self.data.req_hist.append(float(sys_.get("req_rate", 0)))
            self._update_predictive_alerts()

    def _update_predictive_alerts(self):
        tps = list(self.data.tps_hist)
        if len(tps) >= 10:
            slope = (tps[-1] - tps[-10]) / 10
            if slope < -0.5:
                self.data.predict_alerts.append((
                    time.time(), "⚠ TPS declining - may breach SLA soon"
                ))
        self.data.predict_alerts = self.data.predict_alerts[-10:]

class LogBuffer:
    def __init__(self, maxlen: int = 500):
        self._buf = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._seen = set()
        self._seen_max = 2000

    def add(self, ts: str, level: str, logger: str, msg: str, ts_us: int = 0, req_id: str = ""):
        key = (ts_us, msg, req_id)
        with self._lock:
            if ts_us > 0 and key in self._seen:
                return
            self._buf.append((ts, level, logger, msg, req_id))
            if ts_us > 0:
                self._seen.add(key)
                if len(self._seen) > self._seen_max:
                    sorted_keys = sorted(self._seen, key=lambda k: k[0])
                    self._seen = set(sorted_keys[self._seen_max // 2:])

    def lines(self) -> list:
        with self._lock:
            return list(self._buf)

class LogPoller(threading.Thread):
    def __init__(self, client: APIClient, buf: LogBuffer, interval: float = 2.0):
        super().__init__(daemon=True, name="log_poller")
        self.client = client
        self.buf = buf
        self.interval = interval
        self._stop = threading.Event()
        self._last_ts_us = 0
        self._first_conn = True

    def stop(self):
        self._stop.set()

    def run(self):
        backoff = self.interval
        max_backoff = 30.0
        consec_errors = 0
        while not self._stop.is_set():
            try:
                if self._first_conn:
                    url = f"/v1/logs?n=500&since=0.0"
                else:
                    ts_s = self._last_ts_us / 1_000_000.0
                    url = f"/v1/logs?n=200&since={ts_s}"
                resp = self.client.get(url, timeout=3)
                if resp:
                    entries = resp.get("entries", [])
                    max_ts = self._last_ts_us
                    for e in entries:
                        ts_f = e.get("ts", 0.0)
                        ts_us = int(ts_f * 1_000_000)
                        max_ts = max(max_ts, ts_us)
                        ts_str = datetime.fromtimestamp(ts_f).strftime("%H:%M:%S")
                        lgr = e.get("logger", "").split(".")[-1][:14]
                        self.buf.add(ts_str, e.get("level", "INFO"), lgr, str(e.get("msg", "")),
                                     ts_us=ts_us, req_id=e.get("req_id", ""))
                    self._last_ts_us = max_ts
                    self._first_conn = False
                    consec_errors = 0
                    backoff = self.interval
                else:
                    consec_errors += 1
                    backoff = min(max_backoff, self.interval * (2 ** min(consec_errors, 6)))
            except Exception as e:
                consec_errors += 1
                backoff = min(max_backoff, self.interval * (2 ** min(consec_errors, 10)))
                _debug(f"logpoll error: {e}")
            self._stop.wait(backoff)

# ------------------------------------------------------------------------------
# Main TUI class (all panels & modals preserved, with fixes)
# ------------------------------------------------------------------------------
class TUIMonitor:
    def __init__(self, stdscr, config: Dict[str, Any], client: APIClient, start_panel: int = 0):
        self.scr = stdscr
        self.config = config
        self.client = client
        self.data = ServerData()
        self.logs = LogBuffer(maxlen=config.get("log_buffer_size", 500))
        base_interval = config.get("refresh_interval", 2.0)
        self.poller = Poller(client, self.data, interval=base_interval)
        self.log_poll = LogPoller(client, self.logs, interval=base_interval)
        self.running = True
        self.panel = start_panel
        self.modal: Optional[str] = None
        self.sel_model = 0
        self.sel_profile = 0
        self.sel_key = 0
        self.log_scroll = 0
        self._lf = {"model_id": "", "model_path": "", "context": "2048", "model_type": "auto"}
        self._lf_focus = 0
        self._last_draw = 0.0
        self._fps = config.get("tui", {}).get("fps", 10)
        self._start_ts = time.time()
        self._notif: Optional[tuple] = None
        # real admin authentication – only if master key is already set
        self._admin_authenticated = bool(client.master_key)
        self._admin_input_buf = ""  # for master key input modal
        self._new_key = {"key_id": "", "rps": "10.0", "capacity": "20.0", "description": "", "expires": ""}
        self._new_key_focus = 0
        self._selected_key: Optional[dict] = None
        self._panels = [
            "🖥  Hardware", "📊 Performance", "🧠 Memory", "💿 Drive",
            "🤖 Models", "📋 Logs", "🔑 Keys", "📊 Activity", "⚙  Config", "🏆 Ranking"
        ]
        # Log filter state (persistent)
        self._log_filter_level: Optional[str] = None
        self._log_filter_text: str = ""
        self._log_filter_regex: Optional[re.Pattern] = None
        # declare instance vars used in modal handlers
        self._log_search_buf: str = ""
        self._log_regex_buf: str = ""
        self._filter_level_sel: int = 0

        self._ui_queue = queue.Queue()
        self._ui_lock = threading.Lock()

        # Adaptive polling
        self._adaptive_polling = config.get("tui", {}).get("adaptive_polling", True)
        self._base_poll_interval = base_interval

        # Watchdog
        self._watchdog_interval = 30.0
        self._last_watchdog_check = time.monotonic()

        # draw error counter for visible indicator
        self._draw_errors = 0
        self._last_draw_error_ts = 0.0

    def _enqueue_ui(self, action: str, *args, **kwargs):
        self._ui_queue.put((action, args, kwargs))

    def _process_ui_queue(self):
        try:
            while True:
                action, args, kwargs = self._ui_queue.get_nowait()
                if action == "notify":
                    self._notif = (args[0], time.time(), args[1] if len(args) > 1 else "INFO")
                # FIX #1: trigger_poll removed from UI queue – we spawn thread directly
                elif action == "adaptive_interval":
                    new_int = args[0]
                    self.poller.interval = new_int
                    self.log_poll.interval = new_int
        except queue.Empty:
            pass

    def _watchdog_check(self):
        """Restart poller if it appears stuck (no updates for > 60s)."""
        now = time.time()
        with self.data.lock:
            last = self.data.poller_last_active
        if now - last > 60 and self.poller.is_alive():
            _debug("Watchdog: poller appears stuck, restarting...")
            self._restart_poller()

    def _restart_poller(self):
        old_cycle = getattr(self.poller, "_cycle", 0)
        self.poller.stop()
        self.poller.join(timeout=2.0)
        self.poller = Poller(self.client, self.data, interval=self._base_poll_interval)
        self.poller._cycle = old_cycle  # preserve cycle counter
        self.poller.start()
        self._notify("Poller restarted", "WARN")

    def _adaptive_adjust_polling(self):
        if not self._adaptive_polling:
            return
        with self.data.lock:
            cpu_pct = _f(_nested(self.data.metrics, "system", "cpu_pct"), 0)
            ram_pct = _f(_nested(self.data.metrics, "system", "ram_pct"), 0)
        if cpu_pct > 80 or ram_pct > 90:
            new_interval = min(self._base_poll_interval * 2, 5.0)
        else:
            new_interval = self._base_poll_interval
        if abs(new_interval - self.poller.interval) > 0.1:
            self._enqueue_ui("adaptive_interval", new_interval)

    def _log(self, level: str, msg: str, logger: str = "tui"):
        self.logs.add("--:--:--", level, logger, msg, req_id="")

    def run(self):
        global _THEME
        _THEME = self.config.get("tui", {}).get("color_theme", "dark")
        _init_colors()
        curses.curs_set(0)
        self.scr.timeout(100)
        self.scr.keypad(True)
        self.poller.start()
        self.log_poll.start()
        self._log("INFO", f"⚡ Monitor v1.0.0 → {self.client.base}")
        self._draw_splash()
        try:
            while self.running:
                self._process_ui_queue()
                try:
                    key = self.scr.getch()
                    if key == curses.KEY_RESIZE:
                        curses.update_lines_cols()
                        self.scr.erase()
                        self._draw()
                        continue
                    if key != curses.ERR:
                        self._handle_key(key)
                except Exception as e:
                    _debug(f"key error: {e}")
                now = time.monotonic()
                if now - self._last_draw >= 1.0 / self._fps:
                    try:
                        self._adaptive_adjust_polling()
                        if now - self._last_watchdog_check >= self._watchdog_interval:
                            self._watchdog_check()
                            self._last_watchdog_check = now
                        self._draw()
                        # reset draw error counter on successful draw
                        self._draw_errors = 0
                    except Exception as e:
                        # count draw errors and show warning
                        _debug(f"draw error: {e}")
                        self._draw_errors += 1
                        self._last_draw_error_ts = time.time()
                        # If too many consecutive errors, force a redraw of just the status bar
                        if self._draw_errors >= 5:
                            try:
                                self.scr.erase()
                                self._draw_statusbar(h=self.scr.getmaxyx()[0], w=self.scr.getmaxyx()[1])
                                self.scr.refresh()
                            except Exception:
                                pass
                    self._last_draw = now
        finally:
            self.poller.stop()
            self.log_poll.stop()
            self.poller.join(timeout=2.0)
            self.log_poll.join(timeout=2.0)

    def _draw_splash(self):
        """Show a short splash screen until the first data arrives."""
        h, w = self.scr.getmaxyx()
        messages = [
            "⚡ Local LLM Cloud TUI Monitor v1.0.0",
            "",
            f"Connecting to {self.client.base} ...",
            "Use [Tab] to switch panels, [q] to quit.",
            "",
            "Waiting for server response...",
        ]
        y = max(0, (h - len(messages)) // 2)
        for i, line in enumerate(messages):
            _safe_addstr(self.scr, y + i, max(0, (w - len(line)) // 2), line, _cp(_C_TTL, bold=True))
        self.scr.refresh()
        time.sleep(0.8)
        self.scr.erase()

    # --------------------------------------------------------------------------
    # Key handling
    # --------------------------------------------------------------------------
    def _handle_key(self, key: int):
        if self.modal == "load_form":
            self._key_load_form(key)
        elif self.modal == "load_profile":
            self._key_load_profile(key)
        elif self.modal == "confirm_unload":
            self._key_confirm_unload(key)
        elif self.modal == "help":
            self._key_help(key)
        elif self.modal == "admin_auth":
            self._key_admin_auth(key)
        elif self.modal == "create_key":
            self._key_create_key(key)
        elif self.modal == "key_details":
            self._key_key_details(key)
        elif self.modal == "log_search":
            self._key_log_search(key)
        elif self.modal == "log_filter_level":
            self._key_log_filter_level(key)
        elif self.modal == "log_regex":
            self._key_log_regex(key)
        else:
            self._key_main(key)

    def _key_main(self, key: int):
        ch = chr(key) if 32 <= key < 127 else None
        if ch == "q" or key == 3:
            self.running = False
        elif ch == "?":
            self.modal = "help"
        elif key in (9, curses.KEY_RIGHT, ord("n")):
            self.panel = (self.panel + 1) % len(self._panels)
        elif key in (353, curses.KEY_LEFT, ord("b")):
            self.panel = (self.panel - 1) % len(self._panels)
        elif ch == "l":
            self.modal = "load_form"
            self._lf_focus = 0
        elif ch == "p":
            self.modal = "load_profile"
            self.sel_profile = 0
        elif ch == "u":
            with self.data.lock:
                models = (self.data.models or {}).get("data", [])
            if models:
                self.modal = "confirm_unload"
        # trigger poll runs in background thread, no UI freeze
        elif ch in ("r", "") or key == curses.KEY_F5:
            self._trigger_poll()
            self._notify("🔄 Refreshing…")
        elif ch == "c" and self.panel != 6 and self.panel != 7:
            r = self.client.delete("/v1/context-cache/clear")
            self._notify("✅ KV cache cleared" if r else "❌ Clear failed", "OK" if r else "ERROR")
        # restrict log_scroll changes to Logs panel (index 5)
        elif key in (curses.KEY_UP, ord("k")):
            if self.panel == 4:
                self.sel_model = max(0, self.sel_model - 1)
            elif self.panel == 9:
                self.sel_model = max(0, self.sel_model - 1)
            elif self.panel == 6:
                self.sel_key = max(0, self.sel_key - 1)
            elif self.panel == 5:
                self.log_scroll += 1
        elif key in (curses.KEY_DOWN, ord("j")):
            if self.panel == 4:
                with self.data.lock:
                    n = len((self.data.models or {}).get("data", []))
                self.sel_model = min(max(n - 1, 0), self.sel_model + 1)
            elif self.panel == 9:
                with self.data.lock:
                    n = len(self.data.ranking)
                self.sel_model = min(max(n - 1, 0), self.sel_model + 1)
            elif self.panel == 6:
                with self.data.lock:
                    n = len(self.data.keys)
                self.sel_key = min(max(n - 1, 0), self.sel_key + 1)
            elif self.panel == 5:
                self.log_scroll = max(0, self.log_scroll - 1)
        if self.panel == 5:
            if ch == "/":
                self.modal = "log_search"
                self._log_search_buf = ""
            elif ch == "f":
                self.modal = "log_filter_level"
                self._filter_level_sel = 0
            elif ch == "F":
                self.modal = "log_regex"
                self._log_regex_buf = ""
        if self.panel == 6:
            if ch == "c" and not self._admin_authenticated:
                self.modal = "admin_auth"
                self._admin_input_buf = ""
            elif ch == "c" and self._admin_authenticated:
                self.modal = "create_key"
                self._new_key = {"key_id": "", "rps": "10.0", "capacity": "20.0", "description": "", "expires": ""}
                self._new_key_focus = 0
            elif ch == "d" and self._admin_authenticated and self.sel_key < len(self.data.keys):
                self._admin_delete_key()
            elif ch == "r" and self._admin_authenticated and self.sel_key < len(self.data.keys):
                self._admin_revoke_key()
            elif ch == "a" and self._admin_authenticated and self.sel_key < len(self.data.keys):
                self._admin_activate_key()
            elif ch == "i" and self._admin_authenticated and self.sel_key < len(self.data.keys):
                self.modal = "key_details"
                self._selected_key = self.data.keys[self.sel_key]

    # --------------------------------------------------------------------------
    # Trigger poll (non‑blocking)
    # --------------------------------------------------------------------------
    def _trigger_poll_sync(self):
        """Performs the actual synchronous poll (called from background thread)."""
        try:
            self.poller._poll()
        except Exception as e:
            _debug(f"trigger_poll_sync: {e}")

    def _trigger_poll(self):
        """Spawn a daemon thread to poll – does NOT block the UI."""
        threading.Thread(target=self._trigger_poll_sync, daemon=True).start()

    def _notify(self, msg: str, level: str = "INFO"):
        self._enqueue_ui("notify", msg, level)

    # --------------------------------------------------------------------------
    # Modal handlers (with improvements)
    # --------------------------------------------------------------------------
    def _key_load_form(self, key: int):
        fields = list(self._lf.keys())
        if key == 27:
            self.modal = None
            return
        if key in (curses.KEY_DOWN, 9):
            self._lf_focus = (self._lf_focus + 1) % (len(fields) + 1)
        elif key == curses.KEY_UP:
            self._lf_focus = (self._lf_focus - 1) % (len(fields) + 1)
        elif key in (10, 13):
            if self._lf_focus < len(fields):
                self._lf_focus += 1
            else:
                self._submit_load()
        elif self._lf_focus < len(fields):
            f = fields[self._lf_focus]
            if key in (curses.KEY_BACKSPACE, 127, 8):
                self._lf[f] = self._lf[f][:-1]
            elif 32 <= key < 127:
                self._lf[f] += chr(key)

    def _submit_load(self):
        self.modal = None
        payload = {
            "model_id": self._lf["model_id"].strip(),
            "model_path": self._lf["model_path"].strip(),
            "context_length": _i(self._lf.get("context", "4096"), 4096),
            "num_gpu_layers": -1,
            "quantization": "auto",
            "model_type": self._lf.get("model_type", "auto"),
        }
        if not payload["model_id"] or not payload["model_path"]:
            self._notify("❌ model_id and model_path are required", "ERROR")
            return
        self._notify("⏳ Loading model…")
        def _bg():
            r = self.client.post("/v1/models/load", payload, timeout=120)
            if r and "model_id" in r:
                self._notify(f"✅ Loaded: {r['model_id']}", "OK")
            else:
                err = (r or {}).get("detail", "server error")
                self._notify(f"❌ Load failed: {err}", "ERROR")
            self._trigger_poll()
        threading.Thread(target=_bg, daemon=True).start()

    def _key_load_profile(self, key: int):
        with self.data.lock:
            profiles = list(self.data.profiles)
        if key == 27 or key == ord("q"):
            self.modal = None
            return
        if key == curses.KEY_UP:
            self.sel_profile = max(0, self.sel_profile - 1)
        elif key == curses.KEY_DOWN:
            self.sel_profile = min(max(len(profiles) - 1, 0), self.sel_profile + 1)
        elif key in (10, 13) and profiles:
            pid = profiles[self.sel_profile].get("id", "")
            self.modal = None
            self._notify(f"⏳ Loading profile: {pid}")
            def _bg():
                r = self.client.post(f"/v1/models/load-profile/{pid}", {}, timeout=120)
                if r and "status" in r:
                    self._notify(f"✅ Profile loaded: {pid}", "OK")
                else:
                    self._notify(f"❌ Profile failed: {pid}", "ERROR")
                self._trigger_poll()
            threading.Thread(target=_bg, daemon=True).start()

    def _key_confirm_unload(self, key: int):
        if key in (ord("y"), ord("Y")):
            with self.data.lock:
                models = (self.data.models or {}).get("data", [])
            mid = models[self.sel_model]["id"] if (models and self.sel_model < len(models)) else ""
            self.modal = None
            if mid:
                self._notify(f"⏳ Unloading {mid}…", level="INFO")
                def _bg():
                    try:
                        r = self.client.delete(f"/v1/models/{mid}", timeout=30)
                        if r is not None:
                            self._notify(f"✅ Unloaded: {mid}", level="OK")
                        else:
                            time.sleep(2.0)
                            health = self.client.get("/health", timeout=4)
                            if health and health.get("status") == "ok":
                                self._notify(f"✅ Unloaded: {mid}", level="OK")
                            else:
                                self._notify(f"⚠  {mid} unloaded (llama.cpp segfault caught — server survives)", level="WARN")
                    except Exception as exc:
                        self._notify(f"❌ Unload error: {exc}", level="ERR")
                    self._trigger_poll()
                threading.Thread(target=_bg, daemon=True).start()
        else:
            self.modal = None

    def _key_log_filter_level(self, key: int):
        levels = ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL", "CLEAR", "CLEAR ALL"]
        # added "CLEAR ALL" option
        if key == 27:
            self.modal = None
        elif key == curses.KEY_UP:
            self._filter_level_sel = max(0, self._filter_level_sel - 1)
        elif key == curses.KEY_DOWN:
            self._filter_level_sel = min(len(levels) - 1, self._filter_level_sel + 1)
        elif key in (10, 13):
            chosen = levels[self._filter_level_sel]
            if chosen == "CLEAR ALL":
                self._log_filter_level = None
                self._log_filter_text = ""
                self._log_filter_regex = None
                self._log_search_buf = ""
                self._log_regex_buf = ""
                self._notify("All log filters cleared", "OK")
            elif chosen == "CLEAR":
                self._log_filter_level = None
            else:
                self._log_filter_level = chosen
            self.modal = None

    def _key_log_search(self, key: int):
        if key == 27:
            self._log_search_buf = ""
            self.modal = None
        elif key in (10, 13):
            new_text = self._log_search_buf.strip()
            self._log_filter_text = new_text
            self._log_search_buf = ""
            self.modal = None
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            self._log_search_buf = self._log_search_buf[:-1]
        elif 32 <= key < 127:
            self._log_search_buf += chr(key)

    def _key_log_regex(self, key: int):
        if key == 27:
            self._log_regex_buf = ""
            self.modal = None
        elif key in (10, 13):
            txt = self._log_regex_buf.strip()
            self._log_regex_buf = ""
            if txt:
                try:
                    self._log_filter_regex = re.compile(txt)
                except re.error:
                    self._notify("❌ Invalid regex", "ERROR")
                    self._log_filter_regex = None
            else:
                self._log_filter_regex = None
            self.modal = None
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            self._log_regex_buf = self._log_regex_buf[:-1]
        elif 32 <= key < 127:
            self._log_regex_buf += chr(key)

    def _key_help(self, _key: int):
        self.modal = None

    # proper admin authentication with text input
    def _key_admin_auth(self, key: int):
        if key == 27:
            self._admin_input_buf = ""
            self.modal = None
        elif key in (10, 13):
            if self._admin_input_buf == self.client.master_key:
                self._admin_authenticated = True
                self._notify("✅ Admin authenticated", "OK")
            else:
                self._notify("❌ Invalid master key", "ERROR")
            self._admin_input_buf = ""
            self.modal = None
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            self._admin_input_buf = self._admin_input_buf[:-1]
        elif 32 <= key < 127:
            self._admin_input_buf += chr(key)

    def _key_create_key(self, key: int):
        fields = list(self._new_key.keys())
        if key == 27:
            self.modal = None
            return
        if key in (curses.KEY_DOWN, 9):
            self._new_key_focus = (self._new_key_focus + 1) % (len(fields) + 1)
            return
        if key == curses.KEY_UP:
            self._new_key_focus = (self._new_key_focus - 1) % (len(fields) + 1)
            return
        if key in (10, 13):
            if self._new_key_focus < len(fields):
                self._new_key_focus += 1
            else:
                self._submit_create_key()
            return
        if self._new_key_focus >= len(fields):
            return
        f = fields[self._new_key_focus]
        if key in (curses.KEY_BACKSPACE, 127, 8):
            self._new_key[f] = self._new_key[f][:-1]
        elif 32 <= key < 127:
            self._new_key[f] += chr(key)

    def _submit_create_key(self):
        self.modal = None
        key_id = self._new_key["key_id"].strip()
        if not key_id:
            self._notify("❌ Key ID required", "ERROR")
            return
        try:
            rps_val = float(self._new_key["rps"])
            cap_val = float(self._new_key["capacity"])
        except ValueError:
            self._notify("❌ RPS and Capacity must be numbers", "ERROR")
            return
        payload = {
            "key_id": key_id,
            "rate_limit_rps": rps_val,
            "rate_limit_capacity": cap_val,
            "description": self._new_key["description"],
        }
        expires = self._new_key["expires"].strip()
        if expires:
            try:
                payload["expires_in_seconds"] = float(expires)
            except ValueError:
                self._notify("❌ Expires must be a number (seconds)", "ERROR")
                return
        self._notify(f"⏳ Creating key {key_id}…")
        def _bg():
            r = self.client.post("/admin/keys/create", payload, use_master=True, timeout=30)
            if r and "plain_key" in r:
                self._notify(f"✅ Key created: {r['plain_key'][:20]}…", "OK")
            else:
                err = (r or {}).get("detail", "server error")
                self._notify(f"❌ Create failed: {err}", "ERROR")
            self._trigger_poll()
        threading.Thread(target=_bg, daemon=True).start()

    def _key_key_details(self, key: int):
        if key != -1:
            self.modal = None

    def _admin_delete_key(self):
        with self.data.lock:
            keys_snap = list(self.data.keys)
        if self.sel_key >= len(keys_snap):
            return
        key = keys_snap[self.sel_key]
        key_id = key["id"]
        self._notify(f"Deleting key {key_id}...")
        def _bg():
            r = self.client.delete(f"/admin/keys/delete/{key_id}", use_master=True, timeout=30)
            if r is not None:
                self._notify(f"✅ Deleted {key_id}", "OK")
            else:
                self._notify(f"❌ Delete failed for {key_id}", "ERROR")
            self._trigger_poll()
        threading.Thread(target=_bg, daemon=True).start()

    def _admin_revoke_key(self):
        with self.data.lock:
            keys_snap = list(self.data.keys)
        if self.sel_key >= len(keys_snap):
            return
        key = keys_snap[self.sel_key]
        key_id = key["id"]
        self._notify(f"Revoking key {key_id}...")
        def _bg():
            r = self.client.post(f"/admin/keys/revoke/{key_id}", {}, use_master=True, timeout=30)
            if r:
                self._notify(f"✅ Revoked {key_id}", "OK")
            else:
                self._notify(f"❌ Revoke failed for {key_id}", "ERROR")
            self._trigger_poll()
        threading.Thread(target=_bg, daemon=True).start()

    def _admin_activate_key(self):
        with self.data.lock:
            keys_snap = list(self.data.keys)
        if self.sel_key >= len(keys_snap):
            return
        key = keys_snap[self.sel_key]
        key_id = key["id"]
        self._notify(f"Activating key {key_id}...")
        def _bg():
            r = self.client.post(f"/admin/keys/activate/{key_id}", {}, use_master=True, timeout=30)
            if r:
                self._notify(f"✅ Activated {key_id}", "OK")
            else:
                self._notify(f"❌ Activation failed for {key_id}", "ERROR")
            self._trigger_poll()
        threading.Thread(target=_bg, daemon=True).start()

    # --------------------------------------------------------------------------
    # Drawing (dimensions read once at top)
    # --------------------------------------------------------------------------
    def _draw(self):
        try:
            h, w = self.scr.getmaxyx()
        except Exception:
            return
        if h < 15 or w < 60:
            try:
                self.scr.erase()
                self.scr.addstr(0, 0, "⚠ Terminal too small — resize to ≥60×15")
                self.scr.refresh()
            except Exception:
                pass
            return
        try:
            self.scr.erase()
            self._draw_header(h, w)
            self._draw_statusbar(h, w)
            ph = h - 5
            if ph > 3:
                self._draw_panel(3, 0, ph, w)
            if self._notif:
                self._draw_notif(h, w)
            # Reconnection overlay
            with self.data.lock:
                online = self.data.health is not None
            if not online:
                self._draw_reconnect_overlay(h, w)
            if self.modal == "load_form":
                self._modal_load_form(h, w)
            elif self.modal == "load_profile":
                self._modal_load_profile(h, w)
            elif self.modal == "confirm_unload":
                self._modal_confirm_unload(h, w)
            elif self.modal == "help":
                self._modal_help(h, w)
            elif self.modal == "admin_auth":
                self._modal_admin_auth(h, w)
            elif self.modal == "create_key":
                self._modal_create_key(h, w)
            elif self.modal == "key_details":
                self._modal_key_details(h, w)
            elif self.modal == "log_search":
                self._modal_log_search(h, w)
            elif self.modal == "log_filter_level":
                self._modal_log_filter_level(h, w)
            elif self.modal == "log_regex":
                self._modal_log_regex(h, w)
            self.scr.refresh()
        except Exception as e:
            _debug(f"_draw: {e}")

    # reconnect overlay – clear properly, avoid visual artifacts
    def _draw_reconnect_overlay(self, h: int, w: int):
        """Dim overlay when server is offline."""
        overlay_attr = _cp(_C_DIM) | curses.A_REVERSE
        for y in range(3, h - 2):
            _safe_addstr(self.scr, y, 0, " " * (w - 1), overlay_attr)
        msg = " ⚠ SERVER OFFLINE — reconnecting… "
        _safe_addstr(self.scr, h // 2, max(0, (w - _displen(msg)) // 2), msg,
                     _cp(_C_ERR, bold=True) | curses.A_REVERSE)

    # tab bar overflow indicator
    # combined ONLINE + auth error message
    def _draw_header(self, h: int, w: int):
        scr = self.scr
        with self.data.lock:
            online = self.data.health is not None
            models_data = self.data.models
            model_loaded = bool(models_data and isinstance(models_data, dict) and len(models_data.get("data", [])) > 0)
            server_ready = online and model_loaded
            server_booting = online and not model_loaded
            n_err = self.data.errors
            last_ok = self.data.last_ok
        uptime = str(timedelta(seconds=int(time.time() - self._start_ts)))
        title = f"  ⚡ LOCAL LLM CLOUD v1.0.0  │  {self.client.base}  │  up {uptime}"
        hcol = _cp(_C_HDR, bold=True)
        last_ok_age = time.time() - last_ok if last_ok > 0 else 999
        truly_offline = (not online) and (n_err >= 2 or last_ok_age > 8.0 or last_ok == 0)
        transient = (not online) and n_err == 1

        # 🔥 check auth error and combine with online status
        with self.client._auth_lock:
            auth_err = self.client.last_auth_error

        if server_ready:
            scol = _cp(_C_OK, bold=True)
            status = "●  ONLINE "
        elif server_booting:
            scol = _cp(_C_WARN, bold=True)
            status = "●  LOADING…"
        elif transient:
            scol = _cp(_C_WARN, bold=True)
            status = "●  CONNECTING…"
        elif not truly_offline:
            scol = _cp(_C_WARN, bold=True)
            status = "●  CONNECTING…"
        else:
            scol = _cp(_C_ERR, bold=True)
            retry_s = int(last_ok_age) if last_ok > 0 else 0
            status = f"●  OFFLINE  (last seen {retry_s}s ago)" if retry_s else "●  OFFLINE  (server unreachable)"

        # if online but auth error, append combined message
        if online and auth_err in (401, 403):
            status = f"{status}  ⚠ AUTH {auth_err} — master key required for data endpoints"

        _safe_addstr(scr, 0, 0, " " * (w - 1), hcol)
        _safe_addstr(scr, 0, 0, _clip(title, w - _displen(status) - 2), hcol)
        _safe_addstr(scr, 0, max(0, w - _displen(status) - 1), status, scol)
        _safe_addstr(scr, 1, 0, " " * (w - 1), _cp(_C_BDR))
        cx = 1
        total_width = 0
        for lbl in self._panels:
            total_width += _displen(lbl) + 4  # 2 spaces each side
        # Show as many tabs as fit, then +N more
        for i, lbl in enumerate(self._panels):
            tab_label = f" ◀ {lbl} ▶ " if i == self.panel else f"  {lbl}  "
            tab_width = _displen(tab_label)
            if cx + tab_width >= w - 4:
                remaining = len(self._panels) - i
                overflow = f" ▶ +{remaining} more"
                _safe_addstr(scr, 1, w - _displen(overflow) - 2, overflow, _cp(_C_WARN, bold=True))
                break
            attr = _cp(_C_HDR, bold=True) | curses.A_REVERSE if i == self.panel else _cp(_C_DIM)
            _safe_addstr(scr, 1, cx, tab_label, attr)
            cx += tab_width
        _safe_addstr(scr, 2, 0, " " * (w - 1), _cp(_C_BDR))
        if auth_err in (401, 403) and online:
            # This line is kept as a fallback, but the status bar already shows the combined message.
            # We keep it to be safe.
            hints = (
                f"  ⚠ AUTH FAILED ({auth_err}) on data endpoints — set --master-key / "
                "LLM_MASTER_KEY (or use the same key in the master slot). Panels may stay empty."
            )
            _safe_addstr(scr, 2, 0, _clip(hints, w - 1), _cp(_C_ERR, bold=True))
            return
        hints = "  [Tab]Next  [Shift+Tab]Prev  [l]Load  [p]Profile  [u]Unload  [c]Cache  [r]Refresh  [?]Help  [q]Quit"
        if self.panel == 5:
            hints += "  [/]Search  [f]Level  [F]Regex"
            # Show active filter indicators
            if self._log_filter_level or self._log_filter_text or self._log_filter_regex:
                filters = []
                if self._log_filter_level:
                    filters.append(f"level={self._log_filter_level}")
                if self._log_filter_text:
                    filters.append(f"text='{self._log_filter_text}'")
                if self._log_filter_regex:
                    filters.append("regex")
                hints += f"  [active: {', '.join(filters)}]"
        elif self.panel == 6:
            hints = "  [c]Create  [d]Delete  [r]Revoke  [a]Activate  [i]Details  [?]Help  [q]Quit"
        _safe_addstr(scr, 2, 0, _clip(hints, w - 1), _cp(_C_KEY))

    # status bar shows "TPS" not "TP"
    # add draw error indicator in status bar
    def _draw_statusbar(self, h: int, w: int):
        scr = self.scr
        if h < 3 or w < 10:
            return
        sy = h - 2
        with self.data.lock:
            m = self.data.metrics or {}
            eng = m.get("engine", {})
            sys_ = m.get("system", {})
            vram = m.get("vram", {})
            errs = self.data.errors
            predict_alerts = list(self.data.predict_alerts)
        tps = min(_f(eng.get("avg_tps"), hi=500.0), 500.0)
        ttft = _f(eng.get("avg_ttft_ms"), hi=999999.0)
        reqs = _i(eng.get("total_requests"))
        vp = _f(vram.get("utilization_pct"), hi=100.0)
        rp = _f(sys_.get("ram_pct"), hi=100.0)
        cp = _f(sys_.get("cpu_pct"), hi=100.0)
        dw = "W" if _b(eng.get("drive_wired")) else " "
        ts_now = datetime.now().strftime("%H:%M:%S")
        tps_str = f"{tps:.1f}" if tps > 0.01 else "—"
        ttft_str = f"{ttft:.0f}ms" if ttft > 0 else "—"
        # FIX #16: "TPS" not "TP"
        _safe_addstr(scr, sy - 1, 0, "─" * (w - 1), _cp(_C_BDR))
        _safe_addstr(scr, sy, 0, " " * (w - 1), _cp(_C_DIM))
        segs = [
            (f"🚀{tps_str}TPS", _cp(_C_OK) if tps >= 5 else (_cp(_C_WARN) if tps >= 1 else _cp(_C_ERR if tps > 0 else _C_DIM))),
            (f"⏱{ttft_str}", _cp(_C_DIM)),
            (f"📨{reqs:,}", _cp(_C_DIM)),
            (f"🎮V:{vp:.0f}%", _cp(_C_WARN) if vp >= 97 else _cp(_C_DIM)),
            (f"🧠R:{rp:.0f}%", _cp(_C_ERR) if rp >= 90 else (_cp(_C_WARN) if rp >= 75 else _cp(_C_DIM))),
            (f"⚙C:{cp:.0f}%", _cp(_C_DIM)),
            (f"💿{dw}", _cp(_C_OK if dw == "W" else _C_DIM)),
        ]
        if errs > 0:
            segs.append((f"❌{errs}", _cp(_C_ERR, bold=True)))
        segs.append((f"🕐{ts_now}", _cp(_C_DIM)))
        # Predictive alert indicator
        if predict_alerts:
            now = time.time()
            recent = any((now - t) < 10 for t, _ in predict_alerts)
            if recent:
                segs.append(("🔮", _cp(_C_PRED, bold=True)))
        # draw error indicator if repeated draw failures
        if self._draw_errors >= 5:
            segs.append(("⚠ DRAW ERRORS", _cp(_C_ERR, bold=True)))
        cx = 2
        for seg, attr in segs:
            _safe_addstr(scr, sy, cx, seg + "  ", attr)
            cx += _displen(seg) + 2
            if cx >= w - 2:
                break

    def _draw_notif(self, h: int, w: int):
        if not self._notif:
            return
        msg, ts, lvl = self._notif
        if time.time() - ts > 4.0:
            self._notif = None
            return
        col = _cp(_C_OK, bold=True) if lvl in ("OK", "INFO") else _cp(_C_ERR, bold=True)
        nb = f"  {msg}  "
        nx = max(0, (w - _displen(nb)) // 2)
        safe_row = max(4, h - 10)
        _safe_addstr(self.scr, safe_row, nx, nb, col | curses.A_REVERSE)

    def _draw_panel(self, y: int, x: int, h: int, w: int):
        panels = [
            self._panel_hardware, self._panel_performance, self._panel_memory,
            self._panel_drive, self._panel_models, self._panel_logs,
            self._panel_keys, self._panel_activity, self._panel_config,
            self._panel_ranking,
        ]
        try:
            panels[self.panel % len(panels)](y, x, h, w)
        except Exception as e:
            _debug(f"panel {self.panel}: {e}")

    # --------------------------------------------------------------------------
    # Panel implementations (with all required fixes)
    # --------------------------------------------------------------------------
    def _panel_hardware(self, y: int, x: int, h: int, w: int):
        scr = self.scr
        _draw_box(scr, y, x, h, w, "🖥  Hardware")
        row = y + 2
        with self.data.lock:
            si = self.data.system or {}
            cpu_pct = _f(_nested(self.data.metrics, "system", "cpu_pct"))
            cpu_hist = list(self.data.cpu_hist)
            ram_pct = _f(_nested(self.data.metrics, "system", "ram_pct"))
            # ── Get app‑tracked VRAM from metrics ──
            _vram_m = (self.data.metrics or {}).get("vram", {})
            _vram_total_bytes = _i(_vram_m.get("total_bytes", 0))
            _vram_used_bytes = _i(_vram_m.get("used_bytes", 0))
            _vram_free_bytes = max(0, _vram_total_bytes - _vram_used_bytes)
            _vram_util_pct = _f(_vram_m.get("utilization_pct"), hi=100.0)

        bw = max(20, min(w - 52, 45))
        _draw_hline(scr, row, x + 2, w - 4, "⚙  CPU")
        row += 1
        _draw_kv(scr, row, x + 3, "Model", si.get("cpu_model", "detecting…"), lw=20)
        row += 1
        _draw_kv(scr, row, x + 3, "Cores / Threads", f"{si.get('cpu_cores', '?')} physical  /  {si.get('cpu_threads', '?')} logical", lw=20)
        row += 1
        _safe_addstr(scr, row, x + 3, f"{'Usage':<20}", _cp(_C_LBL))
        _draw_bar(scr, row, x + 23, bw, cpu_pct, suffix=f"{cpu_pct:5.1f}%")
        row += 1
        _safe_addstr(scr, row, x + 3, f"{'History 5min':<20}", _cp(_C_LBL))
        _draw_spark(scr, row, x + 23, min(w - 28, 60), cpu_hist)
        row += 2

        _draw_hline(scr, row, x + 2, w - 4, "🧠 System RAM")
        row += 1
        rt = _i(si.get("ram_total_mb"))
        rf = _i(si.get("ram_free_mb"))
        _draw_kv(scr, row, x + 3, "Total", f"{rt:,} MB  ({rt / 1024:.1f} GB)", lw=20)
        row += 1
        _safe_addstr(scr, row, x + 3, f"{'Used':<20}", _cp(_C_LBL))
        _draw_bar(scr, row, x + 23, bw, ram_pct, suffix=f"{ram_pct:5.1f}%  (target ≤75%  free {rf:,} MB)")
        row += 2

        _draw_hline(scr, row, x + 2, w - 4, "🎮 GPU")
        row += 1
        gpus = si.get("gpus", [])
        rec = si.get("recommended_backend", "").lower()
        _BICO = {"vulkan": "🔺", "rocm": "🔴", "cuda": "🟢", "opencl": "🟡", "mps": "🍎", "cpu": "⚙️"}
        if not gpus:
            _safe_addstr(scr, row, x + 3, "⚠  No GPU — CPU-only mode", _cp(_C_WARN, bold=True))
            row += 1
        else:
            for gi, gpu in enumerate(gpus[:2]):
                if row >= y + h - 5:
                    break
                vt = _i(gpu.get("vram_total_mb"))
                vf = _i(gpu.get("vram_free_mb"))
                temp = gpu.get("temperature_c")
                raw_bknd = (gpu.get("backend", "?") or "?").lower()
                bknd = (rec if rec and rec != "opencl" and raw_bknd == "opencl" else raw_bknd)
                vend = gpu.get("vendor", "?")
                temp_s = f"  🌡️ {temp:.0f}°C" if temp else ""
                flag = "✅" if gpu.get("supported") else "⚠"
                _safe_addstr(scr, row, x + 3, f"{flag} GPU{gi}  {_clip(gpu.get('name', '?'), 50)}", _cp(_C_VAL, bold=True))
                row += 1
                if row >= y + h - 3:
                    break
                icon = _BICO.get(bknd, "•")
                _draw_kv(scr, row, x + 5, "Vendor / Backend", f"{vend}  →  {icon} {bknd.upper()}{temp_s}", lw=20)
                row += 1
                if row >= y + h - 3:
                    break
                # ── Unify VRAM display to app‑tracked values ──
                if _vram_total_bytes > 0:
                    _vt_mb = _vram_total_bytes >> 20
                    _vu_mb = _vram_used_bytes >> 20
                    _vf_mb = _vram_free_bytes >> 20
                    _draw_kv(scr, row, x + 5, "VRAM (tracked)", f"{_vu_mb:,} MB used / {_vt_mb:,} MB total (free {_vf_mb:,} MB)", lw=20)
                else:
                    # Fallback to driver‑reported if app metrics not yet available
                    _draw_kv(scr, row, x + 5, "VRAM (driver)", f"{vt - vf:,} MB used / {vt:,} MB total (free {vf:,} MB)", lw=20)
                row += 1
                if row >= y + h - 3:
                    break
                # Show unified usage bar with app‑tracked percentage
                if _vram_total_bytes > 0 and _vram_util_pct > 0:
                    vgp = _vram_util_pct
                    _vram_suffix = f"{_vram_used_bytes>>20}/{_vram_total_bytes>>20} MB  ({vgp:5.1f}%)  ✅ app tracked"
                else:
                    # fallback to driver‑based
                    vgp = (vt - vf) / max(vt, 1) * 100
                    _vram_suffix = f"{vgp:5.1f}%  (sysfs/DRM only)"
                _safe_addstr(scr, row, x + 5, f"{'VRAM usage':<20}", _cp(_C_LBL))
                _draw_bar(scr, row, x + 25, min(w - 32, 35), vgp, suffix=_vram_suffix)
                row += 2
        if row >= y + h - 5:
            return
        _draw_hline(scr, row, x + 2, w - 4, "🔌 Compute Stack")
        row += 1
        _draw_kv(scr, row, x + 3, "Recommended", (rec or "?").upper(), lw=20)
        row += 1
        avail = "  ".join(si.get("available_backends", [])) or "cpu"
        _draw_kv(scr, row, x + 3, "Available", avail, lw=20)
        row += 1
        _avail = si.get("available_backends", [])
        hv = si.get("has_vulkan", False) or "vulkan" in _avail or rec == "vulkan"
        ho = si.get("has_opencl", False) or "opencl" in _avail
        hc = si.get("has_cuda", False) or "cuda" in _avail
        hr = si.get("has_rocm", False) or "rocm" in _avail
        items = []
        if hc:
            items.append("🟢 CUDA")
        if hr:
            items.append("🔴 ROCm")
        if hv:
            items.append("🔺 Vulkan")
        if ho and not hv:
            items.append("🟡 OpenCL")
        elif ho and hv:
            items.append("(OpenCL avail)")
        flags = "  ".join(items) or "⚠ CPU only"
        _draw_kv(scr, row, x + 3, "Accelerators", flags, lw=20, vc=(_C_OK if items else _C_WARN))

    # added per‑model stats and confidence index; APU detection guard
    def _panel_performance(self, y: int, x: int, h: int, w: int):
        scr = self.scr
        _draw_box(scr, y, x, h, w, "📊 Performance")
        row = y + 2
        with self.data.lock:
            eng = (self.data.metrics or {}).get("engine", {})
            cl = self.data.cluster or {}
            perf = self.data.perf or {}
            tps_hist = list(self.data.tps_hist)
            ttft_hist = list(self.data.ttft_hist)
            # fetch confidence index and per‑model stats
            confidence = perf.get("confidence_index", 0.0)
            per_model = perf.get("per_model", {})
        tps = min(_f(eng.get("avg_tps"), hi=500.0), 500.0)
        p95_tps = min(_f(eng.get("p95_tps"), hi=500.0), 500.0)
        ttft = _f(eng.get("avg_ttft_ms"), hi=999999.0)
        p95_ttft = _f(eng.get("p95_ttft_ms"), hi=999999.0)
        ngram = _f(eng.get("ngram_hit_rate"), hi=1.0)
        qdepth = _i(_nested(eng, "queue", "pending"))
        cb = str(_nested(eng, "circuit_breaker", "state") or "?")
        with self.data.lock:
            _si = self.data.system or {}
        _gpus = _si.get("gpus", [])
        _rec = _si.get("recommended_backend", "").lower()
        # guard APU detection – only if system data is present
        if self.data.system is not None:
            _is_apu = (any("shared" in str(g.get("memory_type", "")).lower() for g in _gpus)
                       or (len(_gpus) > 0 and _gpus[0].get("vram_total_mb", 9999) <= 2048 and _rec in ("vulkan", "opencl", "cpu")))
        else:
            _is_apu = False
        total_r = _i(cl.get("total_requests"))
        total_t = _i(cl.get("total_tokens"))
        total_e = _i(cl.get("total_errors"))
        batch = _i(cl.get("batch_size"))
        tier = perf.get("tier", "?")
        ttft_ok = perf.get("ttft_pass", False)
        tps_ok = perf.get("tps_pass", False)
        bw = max(20, min(w - 50, 40))
        _draw_hline(scr, row, x + 2, w - 4, "🚀 Throughput")
        row += 1
        tps_c = _C_OK if tps >= 5 else (_C_WARN if tps >= 1 else _C_ERR)
        _safe_addstr(scr, row, x + 3, f"  {tps:8.2f} TPS", _cp(tps_c, bold=True))
        badge = f"  ✅ SLA {tier} PASS" if (ttft_ok and tps_ok) else f"  ⚠  SLA {tier} below target"
        _safe_addstr(scr, row, x + 3 + 18, badge, _cp(_C_OK if (ttft_ok and tps_ok) else _C_WARN, bold=True))
        row += 1
        _peak_tps = max(tps, p95_tps, max(tps_hist) if tps_hist else 0, 0.01)
        _scale = max(20.0, _peak_tps * 1.5)
        _safe_addstr(scr, row, x + 3, f"{'Avg TPS':<22}", _cp(_C_LBL))
        _draw_bar(scr, row, x + 25, bw, min(tps, _scale) / _scale * 100, suffix=f"{tps:.2f} TPS")
        row += 1
        _safe_addstr(scr, row, x + 3, f"{'p95 TPS':<22}", _cp(_C_LBL))
        _draw_bar(scr, row, x + 25, bw, min(p95_tps, _scale) / _scale * 100, suffix=f"{p95_tps:.2f} TPS")
        row += 1
        # display confidence index
        if confidence > 0:
            confidence_color = _C_OK if confidence >= 80 else (_C_WARN if confidence >= 50 else _C_ERR)
            _draw_kv(scr, row, x + 3, "Confidence index", f"{confidence:.1f}%", lw=22, vc=confidence_color)
            row += 1
        if _is_apu and row < y + h - 2:
            _safe_addstr(scr, row, x + 3, "  ⚠ APU ceiling: LPDDR4 ~34GB/s / 4.7GB = ~7TPS max (Qwen3-8B)", _cp(_C_DIM))
            row += 1
            _safe_addstr(scr, row, x + 3, "     For 100TPS: use Llama-3.2-1B-Q4 (~15TPS) or RTX3080+ GPU", _cp(_C_DIM))
            row += 1
        _draw_hline(scr, row, x + 2, w - 4, "⏱  Latency")
        row += 1
        ttft_c = _C_OK if ttft_ok else (_C_WARN if ttft < 30000 else _C_ERR)
        _draw_kv(scr, row, x + 3, "Avg TTFT", f"{ttft:.0f} ms", lw=22, vc=ttft_c)
        row += 1
        _draw_kv(scr, row, x + 3, "p95 TTFT", f"{p95_ttft:.0f} ms", lw=22)
        row += 2
        _draw_hline(scr, row, x + 2, w - 4, "🔧 Engine State")
        row += 1
        _draw_kv(scr, row, x + 3, "Queue depth", str(qdepth), lw=22)
        row += 1
        _draw_kv(scr, row, x + 3, "Batch size", str(batch), lw=22)
        row += 1
        cb_c = _C_OK if cb == "closed" else _C_ERR
        _draw_kv(scr, row, x + 3, "Circuit breaker", cb.upper(), lw=22, vc=cb_c)
        row += 2
        _draw_hline(scr, row, x + 2, w - 4, "📈 Totals")
        row += 1
        _draw_kv(scr, row, x + 3, "Requests", f"{total_r:,}", lw=22)
        row += 1
        _draw_kv(scr, row, x + 3, "Tokens", f"{total_t:,}", lw=22)
        row += 1
        _draw_kv(scr, row, x + 3, "Errors", f"{total_e:,}", lw=22, vc=(_C_ERR if total_e > 0 else _C_OK))
        row += 2
        _draw_hline(scr, row, x + 2, w - 4, "🎲 NGram speculative cache")
        row += 1
        ng_c = _C_OK if ngram > 0.2 else _C_DIM
        _draw_kv(scr, row, x + 3, "4-gram hit rate", f"{ngram * 100:.1f}%  (est. +{ngram * 40:.0f}% TPS)", lw=22, vc=ng_c)
        row += 2
        # display per‑model stats
        if per_model and row < y + h - 6:
            _draw_hline(scr, row, x + 2, w - 4, "📊 Per‑Model Stats (TTFT / TPS)")
            row += 1
            for model_name, stats in list(per_model.items())[:h - row - 2]:
                ttft_m = stats.get("avg_ttft_ms", 0)
                tps_m = stats.get("avg_tps", 0)
                _draw_kv(scr, row, x + 3, model_name[:15], f"{ttft_m:.0f}ms / {tps_m:.1f} TPS", lw=22)
                row += 1
        sw = min(w - 10, 80)
        if tps_hist and row < y + h - 4:
            _draw_hline(scr, row, x + 2, w - 4, "📊 TPS  5 min")
            row += 1
            _draw_spark(scr, row, x + 3, min(sw, len(tps_hist)), tps_hist)
            mx = max(tps_hist) if tps_hist else 0
            _safe_addstr(scr, row, x + 3 + min(sw, len(tps_hist)) + 1, f" peak={mx:.2f}", _cp(_C_DIM))
            row += 2
        if ttft_hist and row < y + h - 3:
            _draw_hline(scr, row, x + 2, w - 4, "⏱  TTFT  5 min")
            row += 1
            _draw_spark(scr, row, x + 3, min(sw, len(ttft_hist)), ttft_hist)
            mx2 = max(ttft_hist) if ttft_hist else 0
            _safe_addstr(scr, row, x + 3 + min(sw, len(ttft_hist)) + 1, f" peak={mx2:.0f}ms", _cp(_C_DIM))

    # add VRAM fragmentation row
    def _panel_memory(self, y: int, x: int, h: int, w: int):
        scr = self.scr
        _draw_box(scr, y, x, h, w, "🧠 Memory Hierarchy")
        row = y + 2
        with self.data.lock:
            m = self.data.metrics or {}
            vhist = list(self.data.vram_hist)
            rhist = list(self.data.ram_hist)
        vram = m.get("vram", {})
        ram = m.get("ram", {})
        swap = m.get("swap", {})
        eng = m.get("engine", {})
        bw = max(20, min(w - 52, 45))
        _draw_hline(scr, row, x + 2, w - 4, "🎮 GPU VRAM  (target 95%)")
        row += 1
        vp = _f(vram.get("utilization_pct"), hi=100.0)
        vt = _i(vram.get("total_bytes")) >> 20
        vu = _i(vram.get("used_bytes")) >> 20
        vfr = _i(vram.get("free_bytes")) >> 20
        _safe_addstr(scr, row, x + 3, f"{'Usage':<22}", _cp(_C_LBL))
        _draw_bar(scr, row, x + 25, bw, vp, suffix=f"{vu}/{vt} MB  ({vp:.1f}%)")
        row += 1
        _draw_kv(scr, row, x + 3, "Free", f"{vfr:,} MB", lw=22)
        row += 1
        _draw_kv(scr, row, x + 3, "Cache hits/misses", f"{_i(vram.get('cache_hits')):,}  /  {_i(vram.get('cache_misses')):,}", lw=22)
        row += 1
        _draw_kv(scr, row, x + 3, "Evictions/defrag", f"{_i(vram.get('evictions')):,}  /  {_i(vram.get('defrag_runs'))} runs", lw=22)
        row += 1
        # fragmentation row
        frag_bytes = _i(vram.get("fragmented_bytes"))
        if frag_bytes > 0:
            frag_mb = frag_bytes >> 20
            frag_pct = (frag_mb / max(vt, 1)) * 100
            _draw_kv(scr, row, x + 3, "Fragmentation", f"{frag_mb} MB ({frag_pct:.1f}%)", lw=22, vc=_C_WARN if frag_pct > 10 else _C_DIM)
            row += 1
        _safe_addstr(scr, row, x + 3, f"{'History':<22}", _cp(_C_LBL))
        _draw_spark(scr, row, x + 25, min(w - 30, 60), vhist)
        row += 2
        _draw_hline(scr, row, x + 2, w - 4, "🧠 System RAM  (target 75%)")
        row += 1
        rp = _f(ram.get("system_ram_pct"), hi=100.0)
        rm = _i(ram.get("max_mb"))
        ru = _i(ram.get("used_mb"))
        _safe_addstr(scr, row, x + 3, f"{'System usage':<22}", _cp(_C_LBL))
        _draw_bar(scr, row, x + 25, bw, rp, suffix=f"{rp:.1f}%  (target ≤75%)")
        row += 1
        cp2 = _f(ram.get("utilization"), hi=100.0)
        _safe_addstr(scr, row, x + 3, f"{'Cache budget':<22}", _cp(_C_LBL))
        _draw_bar(scr, row, x + 25, bw, cp2, suffix=f"{ru}/{rm} MB")
        row += 1
        _draw_kv(scr, row, x + 3, "Hot / Warm", f"{_i(ram.get('hot_mb')):,} MB  /  {_i(ram.get('warm_mb')):,} MB   mmaps={_i(ram.get('mmapped_files'))}", lw=22)
        row += 1
        _safe_addstr(scr, row, x + 3, f"{'History':<22}", _cp(_C_LBL))
        _draw_spark(scr, row, x + 25, min(w - 30, 60), rhist)
        row += 2
        _draw_hline(scr, row, x + 2, w - 4, "💿 NVMe Swap")
        row += 1
        sw_mb = _f(swap.get("used_mb"))
        sw_max = _f(swap.get("max_gb"))
        sw_e = _i(swap.get("num_entries"))
        sw_pct = (sw_mb / max(sw_max * 1024, 1)) * 100
        _safe_addstr(scr, row, x + 3, f"{'Usage':<22}", _cp(_C_LBL))
        _draw_bar(scr, row, x + 25, bw, sw_pct, suffix=f"{sw_mb:.0f} MB / {sw_max:.0f} GB  ({sw_e} entries)")
        row += 2
        ctx = eng.get("context_cache") or {}
        if ctx and row < y + h - 4:
            _draw_hline(scr, row, x + 2, w - 4, "🔑 KV Context Cache  (TTFT ~5ms on hit)")
            row += 1
            cm = _i(ctx.get("max_mb", 1))
            cu = _i(ctx.get("used_mb"))
            cp3 = cu / max(cm, 1) * 100
            _safe_addstr(scr, row, x + 3, f"{'Usage':<22}", _cp(_C_LBL))
            _draw_bar(scr, row, x + 25, bw, cp3, suffix=f"{cu}/{cm} MB")
            row += 1
            hit = _f(ctx.get("hit_rate"), hi=1.0)
            hc = _C_OK if hit > 0.3 else (_C_WARN if hit > 0.1 else _C_DIM)
            _draw_kv(scr, row, x + 3, "Hit rate", f"{hit * 100:.1f}%  ({_i(ctx.get('hits')):,} hits / {_i(ctx.get('misses')):,} misses)", lw=22, vc=hc)

    def _panel_drive(self, y: int, x: int, h: int, w: int):
        # unchanged – no bug fixes required
        scr = self.scr
        _draw_box(scr, y, x, h, w, "💿 Drive Pipeline  ·  AIMD + NGram + CPU Isolation")
        row = y + 2
        with self.data.lock:
            d = self.data.drive or {}
        bw = max(20, min(w - 52, 40))
        drive_ok = _b(d.get("drive_wired"))
        _draw_hline(scr, row, x + 2, w - 4, "🔌 Pipeline Status")
        row += 1
        _draw_kv(scr, row, x + 3, "Overall", "✅ ALL SYSTEMS ACTIVE" if drive_ok else "❌ PIPELINE INACTIVE", lw=22, vc=(_C_OK if drive_ok else _C_ERR))
        row += 2
        ctx = d.get("context_cache") or {}
        _draw_hline(scr, row, x + 2, w - 4, "🔑 KV Context Cache")
        row += 1
        _draw_kv(scr, row, x + 3, "Purpose", "Skip prefill → TTFT ~5ms vs 500ms", lw=22)
        row += 1
        if ctx:
            hit = _f(ctx.get("hit_rate"), hi=1.0)
            _draw_kv(scr, row, x + 3, "Entries", f"{_i(ctx.get('entries')):,}", lw=22)
            row += 1
            _draw_kv(scr, row, x + 3, "Hit rate", f"{hit * 100:.1f}%", lw=22, vc=(_C_OK if hit > 0.2 else _C_WARN))
            row += 1
            cp = _i(ctx.get("used_mb")) / max(_i(ctx.get("max_mb"), 1), 1) * 100
            _safe_addstr(scr, row, x + 3, f"{'Memory':<22}", _cp(_C_LBL))
            _draw_bar(scr, row, x + 25, bw, cp, suffix=f"{ctx.get('used_mb', 0)}/{ctx.get('max_mb', 0)} MB")
            row += 2
        else:
            row += 2
        tput = d.get("throughput") or {}
        _draw_hline(scr, row, x + 2, w - 4, "📊 AIMD Throughput Controller")
        row += 1
        _draw_kv(scr, row, x + 3, "Purpose", "Additive-Increase/Mult-Decrease → sustain target TPS", lw=22)
        row += 1
        if tput:
            cur = min(_f(tput.get("current_tps"), hi=500.0), 500.0)
            on = _b(tput.get("on_target"))
            tgt = _f(tput.get("target_tps", 5.0))
            ratio = _f(tput.get("tps_ratio"), hi=10.0)
            if cur == 0.0:
                with self.data.lock:
                    _atps = min(_f((self.data.metrics or {}).get("engine", {}).get("avg_tps"), hi=500.0), 500.0)
                _cur_s = f"idle (avg {_atps:.2f} TPS)"
                _cur_c = _C_DIM
            else:
                _cur_s = f"{cur:.2f}"
                _cur_c = _C_OK if on else _C_WARN
            _draw_kv(scr, row, x + 3, "Current TPS", _cur_s, lw=22, vc=_cur_c)
            row += 1
            _draw_kv(scr, row, x + 3, "Target TPS", f"{tgt:.1f}", lw=22)
            row += 1
            _safe_addstr(scr, row, x + 3, f"{'TPS ratio':<22}", _cp(_C_LBL))
            _draw_bar(scr, row, x + 25, bw, min(ratio * 100, 100), suffix=f"{ratio * 100:.0f}% of target")
            row += 2
        else:
            row += 2
        ngram = d.get("ngram_cache") or {}
        _draw_hline(scr, row, x + 2, w - 4, "🎲 NGram Speculative Cache")
        row += 1
        _draw_kv(scr, row, x + 3, "Purpose", "4-gram prediction → +20–40% TPS  (zero extra inference cost)", lw=22)
        row += 1
        hr = _f(ngram.get("hit_rate"), hi=1.0)
        _draw_kv(scr, row, x + 3, "4-gram hit rate", f"{hr * 100:.1f}%", lw=22, vc=(_C_OK if hr > 0.2 else _C_DIM))
        row += 2
        cpu_t = d.get("cpu_throttler") or {}
        _draw_hline(scr, row, x + 2, w - 4, "⚙  CPU Thread Isolation")
        row += 1
        if cpu_t:
            phys = cpu_t.get("physical_cores", "?")
            limit = cpu_t.get("thread_limit", "?")
            _draw_kv(scr, row, x + 3, "Physical cores", str(phys), lw=22)
            row += 1
            _draw_kv(scr, row, x + 3, "Inference limit", f"{limit}  (half of {phys} — leaves headroom for OS + I/O)", lw=22)
            row += 2
        else:
            row += 2
        mp = d.get("memory_pressure") or {}
        if mp and row < y + h - 5:
            _draw_hline(scr, row, x + 2, w - 4, "🔴 Memory Pressure Monitor  (async, 10s)")
            row += 1
            vp = _f(mp.get("vram_pct"), hi=100.0)
            rp = _f(mp.get("ram_pct"), hi=100.0)
            vok = _b(mp.get("vram_ok", True))
            rok = _b(mp.get("ram_ok", True))
            _draw_kv(scr, row, x + 3, "VRAM", f"{vp:.1f}%  {'✅ OK' if vok else '⚠ PRESSURE — defrag triggered'}", lw=22, vc=(_C_OK if vok else _C_ERR))
            row += 1
            _draw_kv(scr, row, x + 3, "RAM", f"{rp:.1f}%  {'✅ OK' if rok else '⚠ PRESSURE — GC triggered'}", lw=22, vc=(_C_OK if rok else _C_ERR))
            row += 1
            _draw_kv(scr, row, x + 3, "Actions taken", f"{_i(mp.get('actions_taken'))}  (auto fires when targets exceeded)", lw=22)

    def _panel_models(self, y: int, x: int, h: int, w: int):
        # unchanged – no bug fixes required
        scr = self.scr
        _draw_box(scr, y, x, h, w, "🤖 Loaded Models")
        row = y + 2
        with self.data.lock:
            models = (self.data.models or {}).get("data", [])
        if not models:
            _safe_addstr(scr, row, x + 3, "⚠  No models loaded.", _cp(_C_WARN, bold=True))
            row += 1
            _safe_addstr(scr, row, x + 3, "  → [l] load dialog    [p] config profile", _cp(_C_DIM))
            return
        cols = [("ID", 14), ("Type", 8), ("Backend", 9), ("Strategy", 10), ("GPU L", 6), ("CPU L", 6),
                ("VRAM MB", 8), ("Threads", 8), ("Reqs", 7)]
        hdr = "".join(f"{c:<{cw}}" for c, cw in cols)
        _safe_addstr(scr, row, x + 2, _clip(hdr, w - 4), _cp(_C_TTL, bold=True) | curses.A_UNDERLINE)
        row += 1
        for i, m in enumerate(models):
            if row >= y + h - 9:
                break
            active = (i == self.sel_model)
            model_type = m.get("model_type", "gguf")
            vals = [str(m.get("id", "?"))[:13], model_type[:7],
                    str(m.get("backend", "?"))[:8],
                    str(m.get("strategy", "?"))[:9], str(m.get("gpu_layers", 0)),
                    str(m.get("cpu_layers", 0)), str(m.get("vram_mb", 0)),
                    str(m.get("cpu_threads", "?")), str(m.get("requests", 0))]
            line = "".join(f"{v:<{cw}}" for v, (_, cw) in zip(vals, cols))
            a = _cp(_C_SEL, bold=True) if active else _cp(_C_MDL)
            _safe_addstr(scr, row, x + 2, _clip(line, w - 4), a)
            row += 1
        row += 1
        if 0 <= self.sel_model < len(models) and row < y + h - 6:
            m = models[self.sel_model]
            _draw_hline(scr, row, x + 2, w - 4, "🔍 Selected Detail")
            row += 1
            profile_name = m.get("profile_name") or m.get("profile") or "apu_2gb"
            vram_mb = _i(m.get("vram_mb", 0))
            total_vram = sum(_i(mx.get("vram_mb", 0)) for mx in models)
            vram_warn = " ⚠ overcommit risk!" if total_vram > 1900 and len(models) > 1 else ""
            for lbl, val in [("📁 Path", m.get("path", "?")),
                             ("🏷  Quant", m.get("quantization", "?")),
                             ("⏱  Load time", f"{_f(m.get('load_time_s')):.2f}s"),
                             ("🕐 Uptime", f"{_i(m.get('age_s')):,}s"),
                             ("⚙  Profile", profile_name),
                             ("🎮 VRAM", f"{vram_mb} MB{vram_warn if vram_mb == max(_i(mx.get('vram_mb', 0)) for mx in models) else ''}")]:
                if row >= y + h - 2:
                    break
                vc = _C_WARN if "⚠" in str(val) else _C_VAL
                _draw_kv(scr, row, x + 3, lbl, str(val), lw=20, vc=vc)
                row += 1
        if len(models) > 1 and row < y + h - 3:
            total_vram = sum(_i(m.get("vram_mb", 0)) for m in models)
            color = _C_WARN if total_vram > 1900 else _C_OK
            _safe_addstr(scr, row, x + 3, f"  Total VRAM: {total_vram} MB across {len(models)} models" + (" ⚠ exceeds budget!" if total_vram > 1950 else ""), _cp(color))
            row += 1
        _safe_addstr(scr, min(row + 1, y + h - 2), x + 3, "  ↑↓ select   [u] unload   [l] load new   [p] profile", _cp(_C_DIM))

    def _panel_logs(self, y: int, x: int, h: int, w: int):
        # unchanged – filter improvements already in modal
        scr = self.scr
        _draw_box(scr, y, x, h, w, "📋 Server Logs  ·  live from /v1/logs")
        raw_lines = self.logs.lines()
        if self._log_filter_level:
            raw_lines = [l for l in raw_lines if l[1] == self._log_filter_level]
        if self._log_filter_text:
            txt = self._log_filter_text.lower()
            raw_lines = [l for l in raw_lines if txt in l[3].lower()]
        if self._log_filter_regex:
            pat = self._log_filter_regex
            raw_lines = [l for l in raw_lines if pat.search(l[3])]
        lines = raw_lines
        vis_h = h - 3
        scroll = max(0, min(self.log_scroll, max(0, len(lines) - vis_h)))
        start = max(0, len(lines) - vis_h - scroll)
        visible = lines[start:start + vis_h]
        _LCOL = {"INFO": _C_DIM, "DEBUG": _C_ACC, "WARNING": _C_WARN, "WARN": _C_WARN, "ERROR": _C_ERR, "ERR": _C_ERR, "CRITICAL": _C_ERR, "OK": _C_OK}
        _LEMO = {"INFO": "ℹ ", "DEBUG": "🔍", "WARNING": "⚠ ", "WARN": "⚠ ", "ERROR": "❌", "ERR": "❌", "CRITICAL": "🔴", "OK": "✅"}
        if not lines:
            _safe_addstr(scr, y + 2, x + 3, f"⏳ Waiting for logs from {self.client.base}/v1/logs …", _cp(_C_DIM))
            _safe_addstr(scr, y + 4, x + 3, "  • Check server is running:  curl http://localhost:8080/health", _cp(_C_DIM))
            _safe_addstr(scr, y + 5, x + 3, "  • If auth enabled, set:      LLM_API_KEY=your-key python -m tui.monitor", _cp(_C_DIM))
            _safe_addstr(scr, y + 6, x + 3, "  • View log file directly:   tail -f /tmp/llm_server.log", _cp(_C_DIM))
            return
        for i, (ts, lvl, lgr, msg, req_id) in enumerate(visible):
            row = y + 1 + i
            if row >= y + h - 1:
                break
            col = _LCOL.get(lvl, _C_DIM)
            emoji = _LEMO.get(lvl, "  ")
            ts_s = (ts or "")[:8]
            lgr_s = f"[{lgr:<14}]" if lgr else " " * 16
            req_display = f" [{req_id[:8]}]" if req_id else ""
            _safe_addstr(scr, row, x + 1, _clip(f" {ts_s} {emoji} {lgr_s}{req_display} {msg}", w - 3), _cp(col))
        if len(lines) > vis_h:
            filter_info = ""
            if self._log_filter_level:
                filter_info += f" level={self._log_filter_level}"
            if self._log_filter_text:
                filter_info += f" text='{self._log_filter_text}'"
            if self._log_filter_regex:
                filter_info += " regex"
            hint = f"  ↑↓/jk  scroll  ·  {len(lines)} entries  ·  scroll={scroll}{filter_info}  ·  [r] refresh"
            _safe_addstr(scr, y + h - 2, x + 2, _clip(hint, w - 4), _cp(_C_DIM))

    def _panel_keys(self, y: int, x: int, h: int, w: int):
        # unchanged
        scr = self.scr
        _draw_box(scr, y, x, h, w, "🔑 API Keys  ·  Admin actions require master key")
        row = y + 2
        if not self._admin_authenticated:
            _safe_addstr(scr, row, x + 3, "⚠ Admin not authenticated. Press [c] and enter master key.", _cp(_C_WARN, bold=True))
            row += 2
        with self.data.lock:
            keys = list(self.data.keys)
        if not keys:
            _safe_addstr(scr, row, x + 3, "⚠ No API keys found.", _cp(_C_WARN, bold=True))
            return
        cols = [("ID", 12), ("RPS", 6), ("Cap", 6), ("Active", 6), ("Requests", 8), ("Tokens", 8), ("Last Used", 12)]
        hdr = "".join(f"{c:<{cw}}" for c, cw in cols)
        _safe_addstr(scr, row, x + 2, _clip(hdr, w - 4), _cp(_C_TTL, bold=True) | curses.A_UNDERLINE)
        row += 1
        for i, k in enumerate(keys):
            if row >= y + h - 3:
                break
            active = (i == self.sel_key)
            a = _cp(_C_SEL, bold=True) if active else _cp(_C_MDL)
            active_str = "✅" if k.get("is_active") else "❌"
            last_used = time.strftime("%H:%M:%S", time.localtime(k.get("last_used", 0))) if k.get("last_used") else "never"
            vals = [k["id"][:12], f"{k.get('rate_limit_rps', 0):.1f}", f"{k.get('rate_limit_capacity', 0):.0f}",
                    active_str, str(k.get("total_requests", 0)), str(k.get("total_tokens", 0)), last_used]
            line = "".join(f"{v:<{cw}}" for v, (_, cw) in zip(vals, cols))
            _safe_addstr(scr, row, x + 2, _clip(line, w - 4), a)
            row += 1
        if row < y + h - 2:
            _safe_addstr(scr, row, x + 3, "  [c]Create  [d]Delete  [r]Revoke  [a]Activate  [i]Details  [?]Help", _cp(_C_DIM))

    # Activity panel shows correct message when master key missing
    def _panel_activity(self, y: int, x: int, h: int, w: int):
        scr = self.scr
        _draw_box(scr, y, x, h, w, "📊 Recent Request Activity")
        row = y + 2
        # Check master key presence first
        if not self.client.master_key:
            _safe_addstr(scr, row, x + 3, "⚠ Master key required to view request activity.", _cp(_C_WARN))
            row += 1
            _safe_addstr(scr, row, x + 3, "   Restart with --master-key YOUR_KEY or set LLM_MASTER_KEY", _cp(_C_DIM))
            return
        with self.data.lock:
            logs = list(self.data.requests)[- (h - 4):]
        if not logs:
            _safe_addstr(scr, row, x + 3, "⚠ No request logs available.", _cp(_C_WARN))
            row += 1
            # The real reason is that /admin/requests endpoint does not exist (api_server bug)
            # But we show a more accurate message:
            _safe_addstr(scr, row, x + 3, "   The /admin/requests endpoint is not implemented in the server.", _cp(_C_DIM))
            row += 1
            _safe_addstr(scr, row, x + 3, "   Check server logs or implement the endpoint to see activity.", _cp(_C_DIM))
            return
        cols = [("Time", 12), ("Key", 10), ("Path", 20), ("Status", 6), ("Latency", 8), ("Tokens", 8)]
        hdr = "".join(f"{c:<{cw}}" for c, cw in cols)
        _safe_addstr(scr, row, x + 2, _clip(hdr, w - 4), _cp(_C_TTL, bold=True) | curses.A_UNDERLINE)
        row += 1
        for log in logs[- (h - 5):]:
            if row >= y + h - 2:
                break
            ts = time.strftime("%H:%M:%S", time.localtime(log.get("timestamp", 0)))
            key_id = log.get("key_id", "")[:10]
            path = log.get("path", "")[:20]
            status = str(log.get("status", 0))
            latency = f"{log.get('latency_ms', 0):.0f}ms"
            tokens = str(log.get("tokens", 0))
            vals = [ts, key_id, path, status, latency, tokens]
            line = "".join(f"{v:<{cw}}" for v, (_, cw) in zip(vals, cols))
            _safe_addstr(scr, row, x + 2, _clip(line, w - 4), _cp(_C_DIM))
            row += 1

    def _panel_config(self, y: int, x: int, h: int, w: int):
        # unchanged
        scr = self.scr
        _draw_box(scr, y, x, h, w, "⚙  Config  ·  Profiles  ·  API Reference")
        row = y + 2
        with self.data.lock:
            profiles = list(self.data.profiles)
        _draw_hline(scr, row, x + 2, w - 4, "📋 config.yaml Model Profiles")
        row += 1
        _safe_addstr(scr, row, x + 3, "  [p] load a profile interactively from any panel.", _cp(_C_DIM))
        row += 2
        if not profiles:
            _safe_addstr(scr, row, x + 3, "⚠  No profiles — server may be offline", _cp(_C_WARN))
            row += 2
        else:
            pcols = [("ID", 16), ("Quant", 8), ("Ctx", 7), ("Auto", 5), ("Description", 32)]
            hdr = "".join(f"{c:<{cw}}" for c, cw in pcols)
            _safe_addstr(scr, row, x + 3, _clip(hdr, w - 6), _cp(_C_TTL, bold=True) | curses.A_UNDERLINE)
            row += 1
            for p in profiles[:h - 10]:
                if row >= y + h - 8:
                    break
                auto = "✅" if p.get("autoload") else "  "
                vals = [str(p.get("id", "?"))[:15], str(p.get("quantization", "?"))[:7],
                        str(p.get("context_length", 0)), auto, str(p.get("description", ""))[:31]]
                _safe_addstr(scr, row, x + 3, _clip("".join(f"{v:<{cw}}" for v, (_, cw) in zip(vals, pcols)), w - 6), _cp(_C_MDL))
                row += 1
            row += 1
        _draw_hline(scr, row, x + 2, w - 4, "📡 API Endpoint Reference")
        row += 1
        for meth, path, desc in [
            ("GET", "/health", "Liveness probe  (public)"),
            ("GET", "/ready", "Readiness  (503 until model loaded)"),
            ("GET", "/v1/system", "Hardware + GPU + backend info"),
            ("GET", "/v1/models", "List loaded models"),
            ("POST", "/v1/models/load", "Load model from file"),
            ("POST", "/v1/models/load-profile/:id", "Load from config.yaml profile"),
            ("DELETE", "/v1/models/:id", "Unload model"),
            ("POST", "/v1/models/:id/swap", "Hot-swap model (zero downtime)"),
            ("POST", "/v1/chat/completions", "Chat inference  (OpenAI-compat)"),
            ("GET", "/v1/metrics", "All live metrics snapshot"),
            ("GET", "/v1/performance", "SLA evaluation (TTFT / TPS)"),
            ("GET", "/v1/drive/stats", "AIMD + NGram + memory pressure"),
            ("GET", "/v1/logs", "Recent server logs"),
            ("DELETE", "/v1/context-cache/clear", "Flush KV prefix cache"),
        ]:
            if row >= y + h - 2:
                break
            mc = _C_OK if meth == "GET" else (_C_WARN if meth == "DELETE" else _C_ACC)
            _safe_addstr(scr, row, x + 3, f"{meth:<8}", _cp(mc, bold=True))
            _safe_addstr(scr, row, x + 12, f"{path:<38}", _cp(_C_VAL))
            _safe_addstr(scr, row, x + 51, _clip(desc, w - 54), _cp(_C_DIM))
            row += 1

    # Ranking panel – change "Votes" to "Tokens"
    def _panel_ranking(self, y: int, x: int, h: int, w: int):
        scr = self.scr
        _draw_box(scr, y, x, h, w, "🏆 Model Ranking  · Bayesian Popularity Score")
        row = y + 2
        with self.data.lock:
            ranking = self.data.ranking or []
        if not ranking:
            _safe_addstr(scr, row, x + 3, "⏳ No ranking data yet (ensure /v1/telemetry is available)", _cp(_C_DIM))
            return
        cols = [("Model ID", 16), ("Score", 8), ("Requests", 10), ("Avg TPS", 8), ("Tokens", 8)]  # FIX #7
        hdr = "".join(f"{c:<{cw}}" for c, cw in cols)
        _safe_addstr(scr, row, x + 2, _clip(hdr, w - 4), _cp(_C_TTL, bold=True) | curses.A_UNDERLINE)
        row += 1
        for i, rm in enumerate(ranking[:min(h - 6, 20)]):
            if row >= y + h - 2:
                break
            active = (i == self.sel_model)
            a = _cp(_C_SEL, bold=True) if active else _cp(_C_MDL)
            score = rm.get("popularity_score", 0)
            vals = [
                rm.get("model_id", "?")[:15],
                f"{score:.2f}",
                str(rm.get("total_requests", 0)),
                f"{rm.get('avg_tps', 0):.1f}",
                str(rm.get("total_tokens", 0))
            ]
            line = "".join(f"{v:<{cw}}" for v, (_, cw) in zip(vals, cols))
            _safe_addstr(scr, row, x + 2, _clip(line, w - 4), a)
            row += 1

    # --------------------------------------------------------------------------
    # Modal rendering (for clear filters)
    # --------------------------------------------------------------------------
    def _modal_load_form(self, h: int, w: int):
        # unchanged
        scr = self.scr
        mh, mw = 16, min(72, w - 4)
        my = (h - mh) // 2
        mx = (w - mw) // 2
        for by in range(my, my + mh):
            _safe_addstr(scr, by, mx, " " * mw, _cp(_C_INV))
        _draw_box(scr, my, mx, mh, mw, "🤖 Load Model", color=_C_ACC)
        _safe_addstr(scr, my + 2, mx + 3, "  Tab/↓ next  ↑ prev  Enter submit  Esc cancel", _cp(_C_DIM))
        fields = list(self._lf.keys())
        labels = ["🏷  Model ID", "📁 Model Path", "📏 Context Length", "🏗  Model Type (auto/gguf/bitnet)"]
        fw, row = mw - 24, my + 4
        for i, (fk, lbl) in enumerate(zip(fields, labels)):
            active = (i == self._lf_focus)
            _safe_addstr(scr, row, mx + 3, f"{lbl:<20}", _cp(_C_ACC, bold=True) if active else _cp(_C_LBL))
            val = self._lf[fk]
            disp = f"{(val + ('▌' if active else ''))[-fw:]:<{fw}}"
            _safe_addstr(scr, row, mx + 23, disp, _cp(_C_HDR) | (curses.A_REVERSE if active else 0))
            row += 2
        btn = "  ✅  LOAD MODEL  "
        btn_a = _cp(_C_OK, bold=True) | curses.A_REVERSE if self._lf_focus == len(fields) else _cp(_C_OK, bold=True)
        _safe_addstr(scr, row, mx + (mw - _displen(btn)) // 2, btn, btn_a)

    def _modal_load_profile(self, h: int, w: int):
        # unchanged
        scr = self.scr
        with self.data.lock:
            profiles = list(self.data.profiles)
        mh = min(len(profiles) + 8, h - 4)
        mw = min(74, w - 4)
        my = (h - mh) // 2
        mx = (w - mw) // 2
        for by in range(my, my + mh):
            _safe_addstr(scr, by, mx, " " * mw, _cp(_C_INV))
        _draw_box(scr, my, mx, mh, mw, "📋 Select Profile to Load", color=_C_ACC)
        row = my + 2
        _safe_addstr(scr, row, mx + 3, "  ↑↓ navigate   Enter load   Esc cancel", _cp(_C_DIM))
        row += 2
        if not profiles:
            _safe_addstr(scr, row, mx + 3, "⚠  No profiles in config.yaml.", _cp(_C_WARN))
            return
        for i, p in enumerate(profiles[:mh - 5]):
            if row >= my + mh - 1:
                break
            active = (i == self.sel_profile)
            a = _cp(_C_SEL, bold=True) if active else _cp(_C_DIM)
            auto = "🟢" if p.get("autoload") else "  "
            q = p.get("quantization", "?")
            line = f" {auto} {p.get('id', '?'):<18} [{q:<7}] {p.get('description', '')[:28]}"
            _safe_addstr(scr, row, mx + 2, _clip(line, mw - 4), a)
            row += 1

    def _modal_confirm_unload(self, h: int, w: int):
        # unchanged
        scr = self.scr
        with self.data.lock:
            models = (self.data.models or {}).get("data", [])
        mid = (models[self.sel_model]["id"] if models and self.sel_model < len(models) else "?")
        mh, mw = 7, 56
        my = (h - mh) // 2
        mx = (w - mw) // 2
        for by in range(my, my + mh):
            _safe_addstr(scr, by, mx, " " * mw, _cp(_C_INV))
        _draw_box(scr, my, mx, mh, mw, "⚠  Confirm Unload", color=_C_WARN)
        _safe_addstr(scr, my + 2, mx + 4, f"Unload model:  {_clip(mid, 32)}", _cp(_C_WARN, bold=True))
        _safe_addstr(scr, my + 4, mx + 4, "  [y] Yes — unload     [any key] Cancel", _cp(_C_DIM))

    def _modal_help(self, h: int, w: int):
        # unchanged
        scr = self.scr
        mh = min(42, h - 4)
        mw = min(70, w - 4)
        my = max(0, (h - mh) // 2)
        mx = max(0, (w - mw) // 2)
        for by in range(my, my + mh):
            _safe_addstr(scr, by, mx, " " * mw, _cp(_C_INV))
        _draw_box(scr, my, mx, mh, mw, "❓ Help  ·  any key to close", color=_C_ACC)
        row = my + 2
        for section, items in [
            ("Navigation", [("Tab / →", "Next panel"), ("Shift+Tab / ←", "Previous panel"), ("n / b", "Next / Previous (tmux-safe)"), ("q / Ctrl+C", "Quit TUI")]),
            ("Actions", [("l", "Open load-model dialog"), ("p", "Load from config.yaml profile"), ("u", "Unload selected model (confirms)"), ("c", "Clear KV context cache"), ("r / F5", "Force-refresh all data")]),
            ("Logs panel", [("↑ / k", "Scroll up"), ("↓ / j", "Scroll down"), ("/", "Search text"), ("f", "Filter by level"), ("F", "Regex filter")]),
            ("Models panel", [("↑ / ↓  or  k / j", "Navigate model list")]),
            ("Ranking panel", [("↑ / ↓  or  k / j", "Navigate model ranking")]),
            ("Data sources", [("/v1/system", "Hardware & GPU"), ("/v1/metrics", "VRAM / RAM / engine stats"), ("/v1/telemetry", "Model ranking")]),
            ("Common fixes", [("No logs in panel 6", "Check LLM_API_KEY if auth enabled"), ("VRAM shows 30–60%", "Normal for APU"), ("TPS = 0.84", "Expected: Qwen3-8B on 2 GB APU via LPDDR4")]),
        ]:
            if row >= my + mh - 1:
                break
            _safe_addstr(scr, row, mx + 3, f"  {section}", _cp(_C_TTL, bold=True))
            row += 1
            for k, desc in items:
                if row >= my + mh - 1:
                    break
                _safe_addstr(scr, row, mx + 5, f"{k:<26}", _cp(_C_KEY, bold=True))
                _safe_addstr(scr, row, mx + 32, _clip(desc, mw - 35), _cp(_C_DIM))
                row += 1
            row += 1

    # admin auth modal with text input
    def _modal_admin_auth(self, h: int, w: int):
        scr = self.scr
        mh, mw = 10, 60
        my = (h - mh) // 2
        mx = (w - mw) // 2
        for by in range(my, my + mh):
            _safe_addstr(scr, by, mx, " " * mw, _cp(_C_INV))
        _draw_box(scr, my, mx, mh, mw, "🔐 Admin Authentication", color=_C_ACC)
        _safe_addstr(scr, my + 2, mx + 3, "Master key is required:", _cp(_C_LBL))
        # Show input buffer
        disp = f"> {self._admin_input_buf}▌"
        _safe_addstr(scr, my + 4, mx + 3, _clip(disp, mw - 6), _cp(_C_VAL, bold=True) | curses.A_REVERSE)
        _safe_addstr(scr, my + 6, mx + 3, "Press Enter to authenticate, Esc to cancel", _cp(_C_DIM))

    def _modal_create_key(self, h: int, w: int):
        # unchanged
        scr = self.scr
        mh, mw = 12, 70
        my = (h - mh) // 2
        mx = (w - mw) // 2
        for by in range(my, my + mh):
            _safe_addstr(scr, by, mx, " " * mw, _cp(_C_INV))
        _draw_box(scr, my, mx, mh, mw, "🔑 Create API Key", color=_C_ACC)
        row = my + 2
        fields = ["key_id", "rps", "capacity", "description", "expires"]
        labels = ["Key ID", "Rate limit (RPS)", "Burst capacity", "Description", "Expires (seconds, optional)"]
        for i, (fk, lbl) in enumerate(zip(fields, labels)):
            active = (i == self._new_key_focus)
            _safe_addstr(scr, row, mx + 4, f"{lbl}:", _cp(_C_ACC, bold=True) if active else _cp(_C_LBL))
            val = self._new_key.get(fk, "")
            disp = val + ("▌" if active else "")
            _safe_addstr(scr, row, mx + 4 + len(lbl) + 2, _clip(disp, 40), _cp(_C_HDR) | (curses.A_REVERSE if active else 0))
            row += 2
        btn = "  ✅ CREATE  "
        btn_a = _cp(_C_OK, bold=True) | curses.A_REVERSE if self._new_key_focus == len(fields) else _cp(_C_OK, bold=True)
        _safe_addstr(scr, row, mx + (mw - _displen(btn)) // 2, btn, btn_a)

    def _modal_key_details(self, h: int, w: int):
        # unchanged
        if not self._selected_key:
            self.modal = None
            return
        k = self._selected_key
        scr = self.scr
        mh, mw = 12, 70
        my = (h - mh) // 2
        mx = (w - mw) // 2
        for by in range(my, my + mh):
            _safe_addstr(scr, by, mx, " " * mw, _cp(_C_INV))
        _draw_box(scr, my, mx, mh, mw, f"🔑 Key Details: {k.get('id', '')}", color=_C_ACC)
        row = my + 2
        _draw_kv(scr, row, mx + 4, "Created", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(k.get("created_at", 0))), lw=15)
        row += 1
        _draw_kv(scr, row, mx + 4, "Expires", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(k.get("expires_at", 0))) if k.get("expires_at") else "never", lw=15)
        row += 1
        _draw_kv(scr, row, mx + 4, "Rate limits", f"{k.get('rate_limit_rps', 0):.1f} RPS / {k.get('rate_limit_capacity', 0):.0f} burst", lw=15)
        row += 1
        _draw_kv(scr, row, mx + 4, "Requests", str(k.get("total_requests", 0)), lw=15)
        row += 1
        _draw_kv(scr, row, mx + 4, "Tokens", str(k.get("total_tokens", 0)), lw=15)
        row += 1
        _draw_kv(scr, row, mx + 4, "Last used", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(k.get("last_used", 0))) if k.get("last_used") else "never", lw=15)
        row += 1
        _draw_kv(scr, row, mx + 4, "Description", k.get("description", ""), lw=15)
        row += 2
        _safe_addstr(scr, row, mx + 4, "Press any key to close", _cp(_C_DIM))

    # add footer hint "(empty = clear filter)"
    def _modal_log_search(self, h: int, w: int):
        scr = self.scr
        mh, mw = 9, 55
        my = (h - mh) // 2
        mx = (w - mw) // 2
        for by in range(my, my + mh):
            _safe_addstr(scr, by, mx, " " * mw, _cp(_C_INV))
        _draw_box(scr, my, mx, mh, mw, "🔍 Log Search", color=_C_ACC)
        _safe_addstr(scr, my + 2, mx + 3, "Enter search text (Enter=confirm  Esc=cancel):", _cp(_C_LBL))
        buf = getattr(self, "_log_search_buf", "")
        disp = f"> {buf}▌"
        _safe_addstr(scr, my + 4, mx + 3, _clip(disp, mw - 6), _cp(_C_VAL, bold=True) | curses.A_REVERSE)
        _safe_addstr(scr, my + 6, mx + 3, "(empty = clear filter)", _cp(_C_DIM))

    # already includes "CLEAR ALL" option
    def _modal_log_filter_level(self, h: int, w: int):
        scr = self.scr
        levels = ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL", "CLEAR", "CLEAR ALL"]
        mh = len(levels) + 4
        mw = 35
        my = (h - mh) // 2
        mx = (w - mw) // 2
        for by in range(my, my + mh):
            _safe_addstr(scr, by, mx, " " * mw, _cp(_C_INV))
        _draw_box(scr, my, mx, mh, mw, "Filter Level", color=_C_ACC)
        row = my + 2
        sel = getattr(self, "_filter_level_sel", 0)
        for i, lvl in enumerate(levels):
            if i == sel:
                _safe_addstr(scr, row + i, mx + 2, f"> {lvl}", _cp(_C_SEL, bold=True))
            else:
                _safe_addstr(scr, row + i, mx + 2, f"  {lvl}", _cp(_C_DIM))

    # add footer hint "(empty = clear filter)"
    def _modal_log_regex(self, h: int, w: int):
        scr = self.scr
        mh, mw = 9, 55
        my = (h - mh) // 2
        mx = (w - mw) // 2
        for by in range(my, my + mh):
            _safe_addstr(scr, by, mx, " " * mw, _cp(_C_INV))
        _draw_box(scr, my, mx, mh, mw, "🔍 Log Regex Filter", color=_C_ACC)
        _safe_addstr(scr, my + 2, mx + 3, "Python regex (Enter=confirm  Esc=cancel):", _cp(_C_LBL))
        buf = getattr(self, "_log_regex_buf", "")
        disp = f"> {buf}▌"
        _safe_addstr(scr, my + 4, mx + 3, _clip(disp, mw - 6), _cp(_C_VAL, bold=True) | curses.A_REVERSE)
        _safe_addstr(scr, my + 6, mx + 3, "(empty = clear filter)", _cp(_C_DIM))

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Local LLM Cloud Terminal Dashboard")
    parser.add_argument("--config", type=Path, help="Path to config.yaml")
    parser.add_argument("--url", help="Server URL (overrides config)")
    parser.add_argument("--refresh", type=float, help="Refresh interval in seconds")
    parser.add_argument("--api-key", help="API key for regular endpoints")
    parser.add_argument("--master-key", help="Master key for admin operations")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--panel", type=int, default=0, help="Start on panel index (0-9)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for k, v in DEFAULTS.items():
        if k not in cfg:
            cfg[k] = v
    if args.url:
        cfg["server_url"] = args.url
    if args.refresh:
        cfg["refresh_interval"] = args.refresh

    # if debug, add a StreamHandler to _debug_logger
    debug_enabled = args.debug or cfg.get("debug", False)
    if debug_enabled:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        _debug_logger.addHandler(sh)

    # use nested lookup for api_key and master_key before falling back to resolvers
    api_key = args.api_key or os.environ.get("LLM_API_KEY") or cfg.get("api_key", "")
    if not api_key:
        # Try nested paths from config
        api_key = _get_nested_config_value(cfg, "api.key", "api.api_key", "auth.api_key", "security.api_key") or ""
    if not api_key:
        api_key = _resolve_api_key_from_server_config(args.config) or ""

    master_key = args.master_key or os.environ.get("LLM_MASTER_KEY") or cfg.get("master_key", "")
    if not master_key:
        master_key = _get_nested_config_value(cfg, "api.master_key", "api.masterkey", "security.master_key") or ""
    if not master_key:
        master_key = _resolve_master_key_from_server_config(args.config) or ""

    cfg["debug"] = debug_enabled

    # startup hint about master key
    if api_key and not master_key:
        print("ℹ️  Only a regular API key provided. Data endpoints that require master key will return 403.",
              file=sys.stderr)
        print("   To access all panels, also set LLM_MASTER_KEY or --master-key (same value works).",
              file=sys.stderr)

    if not api_key and not master_key:
        print("ERROR: No API key or master key provided. Set LLM_API_KEY or LLM_MASTER_KEY, or ensure the server config.yaml is accessible.",
              file=sys.stderr)
        sys.exit(1)

    client = APIClient(cfg["server_url"], api_key=api_key, master_key=master_key, debug=cfg["debug"])

    start_panel = max(0, min(args.panel, 9))

    def _run(stdscr):
        global _THEME
        _THEME = cfg.get("tui", {}).get("color_theme", "dark")
        mon = TUIMonitor(stdscr, cfg, client, start_panel=start_panel)
        mon.run()

    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))
    try:
        curses.wrapper(_run)
    except KeyboardInterrupt:
        pass
    print("\n⚡ Local LLM Cloud monitor closed.\n")

if __name__ == "__main__":
    main()
