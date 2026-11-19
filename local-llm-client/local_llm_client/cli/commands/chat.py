"""
local_llm_client/cli/commands/chat.py — Production Chat Command  v1.0.0

Asynchronous, streaming, interactive chat with conversation persistence,
resume support, performance metrics, and full parameter control.

version = 1.0.0
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncIterator, Set, Tuple, Callable

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.text import Text
from rich.syntax import Syntax


from ...core.async_client import AsyncLLMCloudClient, StreamResult
from ...core.formatter import ServerStats, format_footer
from ...core.models import GenerationResult
from ...core.stream_compactor import StreamCompactor
from ...core.exceptions import (
    APIError, AuthError, ConnectionError, CircuitBreakerOpenError,
    LLMCloudError, StreamInterruptedError,
    ModelNotLoadedError, NotFoundError, RateLimitError,
    ServiceUnavailableError, TimeoutError as CoreTimeoutError,
)
from ..config import get_config, save_config
from ...core.models import ClientConfig
from ...core._accel import estimate_tokens_fast
from ..utils import print_error, print_info, print_success, print_warning, print_notification, confirm

# Optional prompt_toolkit for advanced input
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import WordCompleter, Completer, Completion
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.formatted_text import FormattedText
    _HAS_PROMPT_TOOLKIT = True
except ImportError:
    _HAS_PROMPT_TOOLKIT = False

log = logging.getLogger(__name__)

console = Console()

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
_MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")
DEFAULT_SESSION_DIR = Path.home() / ".llmcloud" / "sessions"
MAX_HISTORY_TOKENS = 4096
LOGO = r"""
[bold violet]
╔══════════════════════════════════════════════════════╗
║                ⚡ LOCAL LLM CLOUD                    ║
║         Private AI – Your Data, Your Hardware        ║
╚══════════════════════════════════════════════════════╝
[/]
"""

_INTERRUPTION_PATTERN = re.compile(
    r"^\[Response interrupted — generation was cancelled after (\d+) tokens\.",
    re.IGNORECASE,
)

# Canonical engine markers (shared with the server and Web UI). The ⚡ glyph
# cannot reasonably appear in model output, so this pattern only matches
# engine-emitted control markers — never user content. Before this regex
# existed the CLI only knew the legacy text form, so a
# `[⚡TRUNCATED:…]` marker was printed to the user verbatim and then saved
# into conversation history, where the model would imitate it.
_STREAM_MARKER_RE = re.compile(
    r"\[⚡(TRUNCATED|INTERRUPTED|STALLED):[^\]]*?tokens=(\d+)[^\]]*\]"
)

# ------------------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------------------
def _get_chat_config() -> Dict[str, Any]:
    cfg = get_config()
    return cfg.get("chat", {})

def _set_chat_config(key: str, value: Any) -> None:
    """previous version did `"chat" not in cfg` on a Config
    object → TypeError. Use Config.set() which handles nested paths."""
    cfg = get_config()
    cfg.set(f"chat.{key}", value)
    save_config(cfg)

def _get_repetition_penalty() -> float:
    """default 1.0 (spec for non-thinking mode). Configurable
    via `chat.repetition_penalty`. Values < 0.01 clamped to 0.01 to avoid
    division-by-zero downstream."""
    try:
        v = _get_chat_config().get("repetition_penalty", 1.0)
        f = float(v)
        return max(0.01, f)
    except (TypeError, ValueError):
        return 1.0

# ------------------------------------------------------------------------------
# Readline history (fallback)
# ------------------------------------------------------------------------------
_HIST_FILE = Path.home() / ".llmcloud" / ".chat_history"
_HIST_FILE.parent.mkdir(parents=True, exist_ok=True)

if not _HAS_PROMPT_TOOLKIT:
    try:
        import readline
        try:
            readline.read_history_file(str(_HIST_FILE))
        except (FileNotFoundError, OSError):
            pass
        readline.set_history_length(500)
        import atexit
        @atexit.register
        def _save_history():
            try:
                readline.write_history_file(str(_HIST_FILE))
            except Exception:
                pass
    except ImportError:
        pass

# ------------------------------------------------------------------------------
# Model name validation
# ------------------------------------------------------------------------------
def validate_model_name(model: str) -> bool:
    return bool(model and isinstance(model, str) and _MODEL_NAME_PATTERN.match(model))

# ------------------------------------------------------------------------------
# Draft‑model filtering
# ------------------------------------------------------------------------------
def filter_user_models(models: List[Dict[str, Any]], show_internal: bool = False) -> List[Dict[str, Any]]:
    if show_internal:
        return models
    return [m for m in models if "draft" not in m.get("tags", [])]

def is_draft_model(model_info: Dict[str, Any]) -> bool:
    return "draft" in model_info.get("tags", [])

# ------------------------------------------------------------------------------
# Conversation persistence
# ------------------------------------------------------------------------------
def _get_session_path(conv_id: str) -> Path:
    session_dir = os.environ.get("LLM_SESSION_DIR", str(DEFAULT_SESSION_DIR))
    path = Path(session_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{conv_id}.json"

def save_conversation(conv_id: str, history: List[Dict[str, str]], title: str) -> None:
    try:
        path = _get_session_path(conv_id)
        data = {
            "id": conv_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "messages": history,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print_warning(f"Failed to save conversation: {e}")

def load_conversation(conv_id: str) -> Optional[Dict[str, Any]]:
    try:
        path = _get_session_path(conv_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print_warning(f"Failed to load conversation: {e}")
        return None

def list_conversations() -> List[Dict[str, Any]]:
    session_dir = os.environ.get("LLM_SESSION_DIR", str(DEFAULT_SESSION_DIR))
    path = Path(session_dir)
    if not path.exists():
        return []
    conversations = []
    for f in sorted(path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                conversations.append({
                    "id": data.get("id", f.stem),
                    "title": data.get("title", "Untitled"),
                    "created_at": data.get("created_at", "unknown"),
                    "message_count": len(data.get("messages", [])),
                })
        except Exception:
            continue
    return conversations

# ── Async wrappers for event-loop-safe file I/O ─────────────────────
# `save_conversation` and `load_conversation` do blocking `open()` and
# `json.dump/load` on small files. On a local SSD this costs a few
# hundred microseconds — but a "few hundred microseconds" on every
# `/save`, `/load`, and post-message write, inside an asyncio loop, is
# precisely the pattern that causes the 11 s event-loop stalls the
# crash log recorded during long generations. Move the whole call
# onto a worker thread.
#
# The sync versions stay for the Click command setup, which runs
# before `asyncio.run(run())` starts the loop and therefore cannot
# block it.
async def save_conversation_async(
    conv_id: str,
    history: List[Dict[str, str]],
    title: str,
) -> None:
    """Async wrapper around `save_conversation`."""
    await asyncio.to_thread(save_conversation, conv_id, history, title)


async def load_conversation_async(conv_id: str) -> Optional[Dict[str, Any]]:
    """Async wrapper around `load_conversation`."""
    return await asyncio.to_thread(load_conversation, conv_id)

# ------------------------------------------------------------------------------
# Client factory
# ------------------------------------------------------------------------------
def _get_client_from_context(
        ctx: click.Context,
        api_key_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
) -> AsyncLLMCloudClient:
    ctx.obj = ctx.obj or {}
    cached = ctx.obj.get("chat_client")
    if cached is not None and not getattr(cached, "_closed", False):
        return cached

    config_path = ctx.obj.get("config_path")
    cfg = get_config(config_path)

    final_api_key = api_key_override or cfg.api_key
    final_base_url = base_url_override or cfg.base_url

    client_cfg = ClientConfig(
        base_url=final_base_url,
        api_key=final_api_key,
        master_key=cfg.master_key,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=float(cfg.timeout),
        max_retries=cfg.max_retries,
        retry_delay=cfg.retry_delay_s,
        retry_backoff=cfg.retry_backoff_factor,
        stream_chunk_ms=cfg.stream_flush_interval_ms,
        pool_size=cfg.pool_size,
        wait_for_model=cfg.wait_for_model,
        wait_timeout=cfg.wait_timeout_s,
        # propagate stats-fetch selector from CLI config (
        # `fetch_server_stats` is derived from `stats_mode`; "chunks" mode
        # skips the extra HTTP round-trip per request).
        fetch_server_stats=getattr(cfg, "fetch_server_stats", True),
    )

    client = AsyncLLMCloudClient(client_cfg)
    ctx.obj["chat_client"] = client
    return client

async def _cleanup_client_async(client: AsyncLLMCloudClient) -> None:
    try:
        await client.close()
    except Exception:
        pass

# ------------------------------------------------------------------------------
# Token counting — BUDGET SIZING ONLY
#
# `_count_tokens` uses estimate_tokens_fast (word-count heuristic). It is
# used ONLY inside `_truncate_history` for prompt-budget arithmetic. It
# MUST NOT be used for user-visible display
# class. Real counts come from StreamResult.stats (streamed) or
# GenerationResult.from_chat_response (non-streamed).
# ------------------------------------------------------------------------------
def _count_tokens(text: str) -> int:
    """Word-count HEURISTIC for internal budget sizing. NEVER display."""
    return estimate_tokens_fast(text)

_SPECIAL_TOKEN_RE = re.compile(
    r"(\[end of text\]"
    r"|<\|eot_id\|>"
    r"|<\|end_of_text\|>"
    r"|<\|begin_of_text\|>"
    r"|<\|start_header_id\|>[^<]*<\|end_header_id\|>"
    r"|<\|end_header_id\|>"
    r"|<\|end\|>"
    r"|<\|user\|>"
    r"|<\|assistant\|>"
    r"|<\|system\|>"
    r")",
    re.DOTALL,
)

def _strip_special_tokens(text: str) -> str:
    if not text:
        return text
    return _SPECIAL_TOKEN_RE.sub("", text)

def _truncate_history(history: List[Dict[str, str]], max_tokens: int = MAX_HISTORY_TOKENS) -> List[Dict[str, str]]:
    total_tokens = 0
    truncated = []
    removed_count = 0
    for msg in reversed(history):
        tokens = _count_tokens(msg["content"]) + 10
        if total_tokens + tokens > max_tokens and truncated:
            removed_count += 1
            continue
        truncated.insert(0, msg)
        total_tokens += tokens
    return truncated, removed_count

# ------------------------------------------------------------------------------
# Streaming helper with retry and fallback
#
# `_result` is populated with `StreamResult.stats` on success (or
# failure-path chunk stats from the inner stream's own finally). No
# token-counting happens here — the caller reads `_result.stats` after
# iteration completes.
# ------------------------------------------------------------------------------
async def _stream_response(
        client: AsyncLLMCloudClient,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        stop: Optional[List[str]],
        no_think: bool,
        timeout: Optional[float] = None,
        retry_count: int = 0,
        max_retries: int = 3,
        _result: Optional[StreamResult] = None,
) -> AsyncIterator[str]:
    kwargs = {}
    if no_think:
        kwargs["no_think"] = True
    if timeout is not None:
        kwargs["timeout_s"] = timeout

    # 0 = unlimited (server decides; engine slides its own
    # context). The old default was 4096, silently truncating every
    # reply longer than that even though the server's own default is
    # unlimited.
    effective_max = max_tokens if max_tokens is not None else 0

    try:
        # chat_stream now returns StreamResult; we
        # read stream.stats after iteration and copy to _result. Also
        # pass `repetition_penalty` explicitly so Qwen3 (default 1.0)
        # does not inherit async_client's 1.1 default.
        stream = client.chat_stream(
            messages=messages,
            model_id=model,
            temperature=temperature,
            max_tokens=effective_max,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            repetition_penalty=_get_repetition_penalty(),
            **kwargs,
        )
        async for chunk in stream:
            yield chunk
        # Success path — propagate stats to caller
        if _result is not None and stream.stats is not None:
            _result.stats = stream.stats
    except RateLimitError as e:
        if retry_count < max_retries:
            delay = (2 ** retry_count) * 0.5
            print_warning(f"Rate limited. Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)
            async for chunk in _stream_response(
                    client, messages, model, temperature, max_tokens,
                    top_p, top_k, stop, no_think, timeout,
                    retry_count + 1, max_retries,
                    _result=_result,
            ):
                yield chunk
        else:
            raise
    except ServiceUnavailableError as e:
        # this is the 503 path that used to be dead code.
        _code  = getattr(e, "error_code", None)
        _retry = float(getattr(e, "retry_after", 0) or 0)

        if _code == "model_starting":
            if retry_count < max_retries:
                _delay = _retry if _retry > 0 else 5.0
                print_warning(
                    f"Model is starting. Retrying in {_delay:.1f}s "
                    f"(attempt {retry_count + 1}/{max_retries})…"
                )
                await asyncio.sleep(_delay)
                async for chunk in _stream_response(
                        client, messages, model, temperature, max_tokens,
                        top_p, top_k, stop, no_think, timeout,
                        retry_count + 1, max_retries,
                        _result=_result,
                ):
                    yield chunk
                return
            print_error(
                f"Model did not become ready after {max_retries} retries. "
                "Check server logs (journalctl -u local-llm-cloud)."
            )
            raise

        if _code == "pool_degraded":
            print_error(
                "Server workers are failing repeatedly. Check server logs: "
                "journalctl -u local-llm-cloud --since '5 minutes ago'"
            )
            raise

        if _code == "queue_deadline_exceeded":
            print_error(
                "Server request queue is saturated — try again in a moment, "
                "or use a smaller prompt."
            )
            raise

        # Unknown 503 — propagate. Caller decides whether to retry.
        raise

    except NotFoundError as e:
        # 404 is permanent — retrying is pointless.
        print_error(f"Model not found on server. {e}")
        raise
    except StreamInterruptedError as e:
        # The server went away mid-answer (typically a restart). What was
        # received has already been printed; hand the UI the canonical marker
        # so it shows "Response interrupted. Type /resume" instead of a
        # traceback and never replays the prompt (which would double the text).
        # Also propagate partial stats (finish_reason="interrupted").
        if _result is not None and getattr(e, "tokens", 0) and _result.stats is None:
            _result.stats = ServerStats(
                completion_tokens=int(getattr(e, "tokens", 0) or 0),
                finish_reason="interrupted",
            )
        print_warning(
            "Lost the connection to the server mid-answer — it probably "
            "restarted. The partial answer above was kept."
        )
        yield f"\n\n[⚡INTERRUPTED:server-restart:tokens={int(getattr(e, 'tokens', 0) or 0)}]"
    except LLMCloudError:
        # Already a clean, user-readable error; the caller prints it.
        raise
    except Exception as e:
        # Genuinely unexpected: log the traceback, don't splash it on the user.
        log.warning("Unexpected stream error", exc_info=True)
        print_error("Stream error: %s", e)
        raise

# ------------------------------------------------------------------------------
# Model availability check
# ------------------------------------------------------------------------------
async def _get_available_models(client: AsyncLLMCloudClient, show_internal: bool = False) -> List[Dict]:
    try:
        models = await client.models()
        if not isinstance(models, list):
            return []
        return filter_user_models(models, show_internal)
    except Exception:
        return []

async def _assert_model_ready(
    client: AsyncLLMCloudClient,
    model_id: str,
    show_internal: bool = False,
) -> Optional[Dict[str, Any]]:
    models = await _get_available_models(client, show_internal)
    for m in models:
        if m.get("id") == model_id:
            return m
    return None

# ------------------------------------------------------------------------------
# Auto-select model when cfg.model is empty
# ------------------------------------------------------------------------------
async def _auto_select_model(client: AsyncLLMCloudClient) -> Optional[str]:
    """Pick the first status:ready model from /v1/models.

    Returns None if the server is unreachable or no model is ready.
    Never raises — the caller decides how to report the failure.
    """
    try:
        models = await client.models()
    except Exception as exc:
        log.warning("Auto-select: models() failed: %s", exc)
        return None
    if not isinstance(models, list):
        return None
    ready = [m for m in models if m.get("status") == "ready"]
    if not ready:
        return None
    return ready[0].get("id") or None

# ------------------------------------------------------------------------------
# Build a ServerStats from a non-streaming GenerationResult
# ------------------------------------------------------------------------------
def _stats_from_generation_result(
    result: GenerationResult,
    total_time_s: float,
) -> ServerStats:
    """Non-streaming path: build a ServerStats from a GenerationResult.

    The server's non-streaming response body has `usage.completion_tokens`
    (authoritative — `stats_source="server"`), but not TTFT or per-phase
    timings. We fill those honestly:

      • ttft_ms = 0        → formatter renders `—` (unknown, not zero)
      • decode_tps = tokens / total_time_s (end-to-end, client wall clock;
        real number but conflates prefill with decode)
      • total_ms = client-measured round-trip time

    This is not as rich as the streaming path (which reads
    `/v1/requests/{id}/stats`), but it is never heuristic: the
    token count is authoritative, the timing is honest wall clock.
    """
    tokens = int(result.tokens_generated or 0)
    tps = (tokens / total_time_s) if (total_time_s > 0 and tokens > 0) else 0.0
    return ServerStats(
        id=result.request_id or "",
        model=result.model or "",
        completion_tokens=tokens,
        prompt_tokens=int(result.prompt_tokens or 0),
        thinking_tokens=int(result.thinking_tokens or 0),
        cached_prompt_tokens=int(result.cached_prompt_tokens or 0),
        ttft_ms=0.0,
        decode_tps=tps,
        total_ms=total_time_s * 1000.0,
        finish_reason=result.finish_reason or "unknown",
    )

# ------------------------------------------------------------------------------
# ThinkParser – incremental <think>/<thinking> tag extraction
# ------------------------------------------------------------------------------
_THINK_OPENS = ("<think>", "<thinking>")
_THINK_CLOSES = ("</think>", "</thinking>")

def _partial_tag_tail_len(s: str, tags: Tuple[str, ...]) -> int:
    best = 0
    for tag in tags:
        limit = min(len(s), len(tag) - 1)
        for l in range(limit, 0, -1):
            if s.endswith(tag[:l]):
                best = max(best, l)
                break
    return best

class ThinkParser:
    def __init__(self):
        self.buffer = ""
        self.in_thinking_block = False

    def push(self, chunk: str) -> Tuple[str, str]:
        self.buffer += chunk
        thinking_parts: List[str] = []
        normal_parts: List[str] = []
        while self.buffer:
            think_open_idx = min(
                self.buffer.find("<think>") if "<think>" in self.buffer else float("inf"),
                self.buffer.find("<thinking>") if "<thinking>" in self.buffer else float("inf")
            )
            think_close_idx = min(
                self.buffer.find("</think>") if "</think>" in self.buffer else float("inf"),
                self.buffer.find("</thinking>") if "</thinking>" in self.buffer else float("inf")
            )
            if not self.in_thinking_block:
                # Stray closer with no matching opener
                _oc = self.buffer.find("</think>")
                _oc2 = self.buffer.find("</thinking>")
                _oc = min([i for i in (_oc, _oc2) if i >= 0], default=-1)
                if _oc >= 0 and (think_open_idx == float("inf") or _oc < think_open_idx):
                    if _oc > 0:
                        normal_parts.append(self.buffer[:_oc])
                    self.buffer = self.buffer[_oc:].split(">", 1)[1].lstrip()
                    continue
                if think_open_idx < float("inf"):
                    if think_open_idx > 0:
                        normal_parts.append(self.buffer[:think_open_idx])
                    if self.buffer.startswith("<think>", think_open_idx):
                        self.buffer = self.buffer[think_open_idx + 7:]
                    else:
                        self.buffer = self.buffer[think_open_idx + 10:]
                    self.in_thinking_block = True
                else:
                    tail = _partial_tag_tail_len(self.buffer, _THINK_OPENS)
                    if tail:
                        if tail < len(self.buffer):
                            normal_parts.append(self.buffer[:-tail])
                        self.buffer = self.buffer[-tail:]
                    else:
                        normal_parts.append(self.buffer)
                        self.buffer = ""
                    break
            else:
                if think_close_idx < float("inf"):
                    thinking_parts.append(self.buffer[:think_close_idx])
                    if self.buffer.startswith("</think>", think_close_idx):
                        self.buffer = self.buffer[think_close_idx + 8:]
                    else:
                        self.buffer = self.buffer[think_close_idx + 11:]
                    self.in_thinking_block = False
                else:
                    tail = _partial_tag_tail_len(self.buffer, _THINK_CLOSES)
                    if tail:
                        if tail < len(self.buffer):
                            thinking_parts.append(self.buffer[:-tail])
                        self.buffer = self.buffer[-tail:]
                    else:
                        thinking_parts.append(self.buffer)
                        self.buffer = ""
                    break
        return "".join(thinking_parts), "".join(normal_parts)

    def flush(self) -> Tuple[str, str]:
        if self.in_thinking_block:
            return self.buffer, ""
        else:
            return "", self.buffer

# ------------------------------------------------------------------------------
# Interruption detection
# ------------------------------------------------------------------------------
def _is_interruption_marker(text: str) -> Optional[Tuple[str, int]]:
    m = _STREAM_MARKER_RE.search(text)
    if m:
        return (m.group(1), int(m.group(2)))
    m = _INTERRUPTION_PATTERN.match(text.strip())
    if m:
        return ("INTERRUPTED", int(m.group(1)))
    return None

# ------------------------------------------------------------------------------
# Checkpoint listing
# ------------------------------------------------------------------------------
async def _list_checkpoints(client: AsyncLLMCloudClient, conv_id: str) -> List[Dict[str, Any]]:
    try:
        resp = await client._request("GET", f"/v1/chat/checkpoints?conv_id={conv_id}")
        return resp.get("checkpoints", [])
    except Exception:
        return []

# ------------------------------------------------------------------------------
# Single message (non‑interactive)
# ------------------------------------------------------------------------------
async def single_chat(
        client: AsyncLLMCloudClient,
        message: str,
        model: str,
        conv_id: Optional[str],
        stream: bool,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        stop: Optional[List[str]],
        no_think: bool,
        system_prompt: Optional[str],
        timeout: Optional[float],
        save: bool,
        output_json: bool,
        show_internal: bool = False,
) -> None:
    model_info = await _assert_model_ready(client, model, show_internal)
    if model_info is None:
        all_models = await client.models()
        if not isinstance(all_models, list):
            all_models = []
        matching = [m for m in all_models if m.get("id") == model]
        if matching and is_draft_model(matching[0]):
            print_error(
                f"Model '{model}' is a draft/internal model and is not exposed to users. "
                "Use --show-internal to allow selecting it for debugging."
            )
        else:
            _avail = sorted({m.get("id", "") for m in all_models if m.get("id")})
            print_error(
                f"Model '{model}' is not in the server catalog. "
                f"Available: {', '.join(_avail) or 'none'}"
            )
        sys.exit(1)
    _status = model_info.get("status") or "unknown"
    if _status != "ready":
        if _status == "loading":
            print_error(
                f"Model '{model}' is loading. Please wait and retry in a few seconds."
            )
        elif _status == "degraded":
            print_error(
                f"Model '{model}' workers are degraded (repeated failures). "
                f"Check server logs: journalctl -u local-llm-cloud --since '5 minutes ago'"
            )
        elif _status == "error":
            print_error(
                f"Model '{model}' failed to load. Check server logs for the "
                f"specific error."
            )
        else:
            print_error(f"Model '{model}' is not ready (status: {_status}).")
        sys.exit(1)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})

    start_time = time.perf_counter()
    full_response = ""
    is_error = False
    interrupted = False
    interrupt_info = None
    stats: Optional[ServerStats] = None

    if stream:
        console.print()  # blank line before header
        console.print("[bold cyan]🤖 AI:[/]")
        parser = ThinkParser()
        think_acc = ""
        answer_acc = ""
        printed_think_header = False
        spinner_stopped = False

        # compactor — one per stream, `None` in preserve mode.
        _bl = get_config().blank_lines
        compactor = (
            StreamCompactor(max_blank_lines=int(_bl))
            if isinstance(_bl, int) else None
        )

        # StreamResult captured by _stream_response, read after loop.
        _result = StreamResult()

        with console.status("[cyan]Waiting for model...", spinner="dots") as status:
            client.on_stream_event = (
                lambda m, _st=status: _st.update(f"[yellow]{m}")
            )
            try:
                async for chunk in _stream_response(
                    client, messages, model, temperature, max_tokens, top_p,
                    top_k, stop, no_think, timeout,
                    _result=_result,
                ):
                    if chunk.strip().startswith("[Error:") or chunk.strip().startswith("ERROR:"):
                        is_error = True
                        full_response = chunk
                        break
                    inter = _is_interruption_marker(chunk)
                    if inter is not None:
                        interrupted = True
                        interrupt_info = inter
                        full_response = chunk
                        break

                    if not spinner_stopped:
                        status.stop()
                        spinner_stopped = True

                    think_part, normal_part = parser.push(chunk)
                    if think_part:
                        if not printed_think_header:
                            console.print(Text("✣ Thinking", style="dim cyan"))
                            printed_think_header = True
                        console.print(think_part, end="", style="dim italic",
                                       highlight=False, markup=False)
                        think_acc += think_part
                    if normal_part:
                        if printed_think_header and not answer_acc:
                            console.print()
                        # compact before display AND storage
                        if compactor is not None:
                            normal_part = compactor.feed(normal_part)
                            if not normal_part:
                                continue
                        console.print(normal_part, end="", highlight=False, markup=False)
                        answer_acc += normal_part
                        full_response += normal_part
                        # NO token_count accumulation
            finally:
                client.on_stream_event = None
                if not is_error and not interrupted:
                    t2, n2 = parser.flush()
                    if t2:
                        console.print(t2, end="", style="dim italic",
                                       highlight=False, markup=False)
                        think_acc += t2
                    if n2:
                        if compactor is not None:
                            n2 = compactor.feed(n2)
                        if n2:
                            console.print(n2, end="", highlight=False, markup=False)
                            answer_acc += n2
                            full_response += n2
                if compactor is not None:
                    compactor.finish()   # drops trailing newlines; returns ""
                if not spinner_stopped:
                    status.stop()

        if not is_error and not interrupted:
            console.print()  # newline after the raw stream text
            stats = _result.stats
            _src = stats.stats_source if stats is not None else "unknown"
            _stats_obj = stats if stats is not None else ServerStats()
            console.print(Text(
                f"\n{format_footer(_stats_obj, stats_source=_src)}",
                style="dim",
            ))

    else:
        console.print()
        console.print("[bold cyan]🤖 AI:[/]")
        t0 = time.perf_counter()
        with Progress(SpinnerColumn(), TextColumn("[cyan]Generating...[/]"), transient=True) as progress:
            task = progress.add_task("generating", total=None)
            resp = await client.chat_completions(
                messages=messages,
                model_id=model,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens is not None else 0,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                stream=False,
                repetition_penalty=_get_repetition_penalty(),
                **({"no_think": no_think} if no_think else {}),
                **({"timeout_s": timeout} if timeout else {}),
            )
        total_time_s = time.perf_counter() - t0

        # authoritative provenance via from_chat_response
        result_obj = GenerationResult.from_chat_response(
            resp, server_url=client.cfg.base_url,
        )
        result_obj.total_time_s = total_time_s
        full_response = resp["choices"][0]["message"]["content"]

        if full_response.strip().startswith("[Error:"):
            is_error = True
        inter = _is_interruption_marker(full_response)
        if inter is not None:
            interrupted = True
            interrupt_info = inter

        parser = ThinkParser()
        think_part, normal_part = parser.push(full_response)
        t2, n2 = parser.flush()
        think_acc = think_part + t2
        answer_acc = normal_part + n2
        if is_error:
            console.print(Panel(
                "The model didn't produce a response — please try again.",
                title="⚠️ Error", border_style="yellow", padding=(0, 1), box=box.ROUNDED,
            ))
        elif interrupted:
            console.print(Panel(
                f"⚠️ Response interrupted after {interrupt_info[1]} tokens.",
                title="Interrupted", border_style="yellow", padding=(0, 1), box=box.ROUNDED,
            ))
            if save and conv_id:
                print_notification("To resume, use: /resume")
        else:
            if think_acc:
                console.print(Text("\u2723 Thinking", style="dim cyan"))
                console.print(Text(think_acc, style="dim italic"))
                console.print()
            if answer_acc:
                try:
                    console.print(Markdown(answer_acc))
                except Exception:
                    console.print(Text(answer_acc))
            else:
                console.print("[yellow]No content received.[/]")

        # build ServerStats from the non-streaming result
        stats = _stats_from_generation_result(result_obj, total_time_s)
        console.print(Text(
            f"\n{format_footer(stats, stats_source=result_obj.stats_source)}",
            style="dim",
        ))

    if output_json:
        # JSON output uses server-authoritative tokens from stats,
        # not estimate_tokens_fast.
        tokens_out = stats.completion_tokens if stats is not None else 0
        tps_out = stats.decode_tps if stats is not None else 0.0
        ttft_out = stats.ttft_ms if stats is not None else 0.0
        stats_source_out = stats.stats_source if stats is not None else "unknown"
        json_str = json.dumps({
            "model": model,
            "message": message,
            "response": full_response,
            "ttft_ms": ttft_out,
            "total_time_s": stats.total_ms / 1000.0 if stats else 0.0,
            "tokens": tokens_out if not is_error else 0,
            "tps": round(tps_out, 2) if not is_error else 0,
            "stats_source": stats_source_out,
            "error": is_error,
            "interrupted": interrupted,
        }, indent=2)
        try:
            console.print(Syntax(json_str, "json", theme="monokai", word_wrap=True))
        except ImportError:
            console.print(json_str)
        return

    if save and conv_id and not is_error and not interrupted:
        history = [{"role": "user", "content": message}, {"role": "assistant", "content": full_response}]
        await save_conversation_async(conv_id, history, message[:50])

# ------------------------------------------------------------------------------
# Resume command handler (unchanged — never displays stats)
# ------------------------------------------------------------------------------
async def _handle_resume(
    client: AsyncLLMCloudClient,
    conv_id: str,
    checkpoint_id: Optional[str] = None,
    max_tokens: int = 0,
    temperature: Optional[float] = None,
    stream: bool = True,
) -> bool:
    interrupted_again = False
    try:
        payload = {
            "conv_id": conv_id,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if checkpoint_id:
            payload["checkpoint_id"] = checkpoint_id
        if temperature is not None:
            payload["temperature"] = temperature

        if hasattr(client, "resume"):
            resp = await client.resume(
                conv_id=conv_id,
                checkpoint_id=checkpoint_id,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )
        else:
            resp = await client._request("POST", "/v1/chat/resume", json=payload)

        if stream:
            console.print()
            console.print("[bold cyan]🤖 AI (resumed):[/]")
            # resume(stream=True) returns StreamResult; __aiter__
            # iterates the inner async generator. No stats display here —
            # the ORIGINAL turn carries stats; resume just continues text.
            if hasattr(resp, "__aiter__"):
                async for chunk in resp:
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode()
                    if chunk.startswith("data: "):
                        try:
                            data = json.loads(chunk[6:])
                            choice = (data["choices"][0]
                                      if data.get("choices") else {})
                            if choice.get("finish_reason") == "interrupted":
                                interrupted_again = True
                            delta = choice.get("delta", {}).get("content", "")
                            if delta:
                                if _STREAM_MARKER_RE.search(delta):
                                    interrupted_again = True
                                    delta = _STREAM_MARKER_RE.sub("", delta).strip()
                                if delta:
                                    console.print(delta, end="", highlight=False)
                            elif data.get("error"):
                                print_error(f"Resume error: {data['error']}")
                                break
                        except json.JSONDecodeError:
                            pass
                    else:
                        if _STREAM_MARKER_RE.search(chunk):
                            interrupted_again = True
                            clean = _STREAM_MARKER_RE.sub("", chunk).strip()
                            if clean:
                                console.print(clean, end="", highlight=False)
                        else:
                            console.print(chunk, end="", highlight=False)
                console.print()
            else:
                if resp and "choices" in resp and resp["choices"]:
                    text = resp["choices"][0]["message"]["content"]
                    console.print()
                    console.print("[bold cyan]🤖 AI (resumed):[/]")
                    console.print(Markdown(text))
                else:
                    print_error("Resume response missing content.")
                    return False
        else:
            if resp and "choices" in resp and resp["choices"]:
                text = resp["choices"][0]["message"]["content"]
                console.print()
                console.print("[bold cyan]🤖 AI (resumed):[/]")
                console.print(Markdown(text))
            else:
                print_error("Resume response missing content.")
                return False
    except ServiceUnavailableError as e:
        _code = getattr(e, "error_code", None)
        if _code == "pool_degraded":
            print_error(
                "Cannot resume — server workers are degraded. "
                "Check server logs: journalctl -u local-llm-cloud"
            )
        elif _code == "model_starting":
            _r = float(getattr(e, "retry_after", 0) or 5.0)
            print_warning(f"Model is starting. Try resume again in ~{_r:.0f}s.")
        elif _code == "queue_deadline_exceeded":
            print_error("Server queue is saturated. Try resume again shortly.")
        else:
            print_error(f"Failed to resume: {e}")
        return False
    except NotFoundError as e:
        print_error(
            f"Resume checkpoint not found. The conversation may have been "
            f"pruned or the server restarted with a different store. {e}"
        )
        return False
    except Exception as e:
        print_error(f"Failed to resume: {e}")
        return False
    return not interrupted_again

# ------------------------------------------------------------------------------
# Interactive chat
# ------------------------------------------------------------------------------
async def interactive_chat(
        client: AsyncLLMCloudClient,
        model: str,
        conv_id: Optional[str],
        stream: bool,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        stop: Optional[List[str]],
        no_think: bool,
        system_prompt: Optional[str],
        timeout: Optional[float],
        save: bool,
        initial_history: Optional[List[Dict[str, str]]] = None,
        initial_title: Optional[str] = None,
        show_internal: bool = False,
) -> None:
    console.print(LOGO)
    print_info(f"Interactive chat with model '{model}'")
    print_info("Type /help for commands, /exit to quit, Ctrl+C to interrupt")

    chat_cfg = _get_chat_config()
    show_dashboard = chat_cfg.get("show_dashboard", False)
    auto_resume = chat_cfg.get("auto_resume", False)
    show_internal = show_internal or chat_cfg.get("show_internal_models", False)

    history: List[Dict[str, str]] = initial_history if initial_history else []
    current_model = model
    current_temp = temperature if temperature is not None else 0.7
    current_max_tokens = max_tokens if max_tokens is not None else 0
    current_top_p = top_p
    current_top_k = top_k
    current_stop = stop or []
    current_no_think = no_think
    current_system = system_prompt or ""
    current_timeout = timeout
    current_conv_id = conv_id or f"chat_{int(time.time())}"
    session_title = initial_title or "New Conversation"
    current_show_internal = show_internal
    last_interrupt_info: Optional[Dict[str, Any]] = None
    model_defaults: Dict[str, Dict[str, Any]] = {}
    user_temperature_override = temperature is not None

    if history:
        print_info(f"Loaded {len(history)} messages from history.")

    available_models: Set[str] = set()
    model_context_length: int = MAX_HISTORY_TOKENS

    async def refresh_models() -> None:
        nonlocal available_models, model_context_length, model_defaults
        all_models = await client.models()
        if not isinstance(all_models, list):
            all_models = []
        user_models = filter_user_models(all_models, current_show_internal)
        available_models = {m["id"] for m in user_models}
        for m in all_models:
            mid = m.get("id")
            if mid:
                default_temp = m.get("default_temperature")
                if default_temp is None or not isinstance(default_temp, (int, float)):
                    default_temp = 0.7
                model_defaults[mid] = {
                    "temperature": float(default_temp),
                    "context_length": m.get("context_length", MAX_HISTORY_TOKENS),
                }
            if m.get("id") == current_model:
                cl = m.get("context_length")
                if isinstance(cl, int) and cl > 0:
                    model_context_length = cl
                if not user_temperature_override and current_model in model_defaults:
                    default_temp = model_defaults[current_model].get("temperature")
                    if default_temp is not None:
                        current_temp = default_temp

    await refresh_models()

    def _history_budget() -> int:
        if current_max_tokens and current_max_tokens > 0:
            reserve_for_response = current_max_tokens
        else:
            reserve_for_response = max(512, model_context_length // 4)
        safety_margin = 128
        return max(256, model_context_length - reserve_for_response - safety_margin)

    def _prompt():
        if _HAS_PROMPT_TOOLKIT:
            internal_flag = " [I]" if current_show_internal else ""
            return FormattedText([
                ('fg:ansigreen bold', 'You'),
                ('', ' ('),
                ('fg:ansibrightblack', current_model + internal_flag),
                ('', f' T={current_temp:.1f}'),
                ('', ' 🧠' if current_no_think else ''),
                ('', '): ')
            ])
        else:
            GREEN = "\033[32;1m"
            DIM = "\033[2m"
            RESET = "\033[0m"
            you = f"{GREEN}You{RESET}"
            model = f"{DIM}{current_model}{RESET}"
            internal = " [I]" if current_show_internal else ""
            temp = f"T={current_temp:.1f}"
            think = " 🧠" if current_no_think else ""
            return f"{you} ({model}{internal} {temp}{think}): "

    def _get_footer() -> str:
        total_tokens = sum(_count_tokens(m["content"]) for m in history)
        footer = f"Model: {current_model} | Temp: {current_temp:.1f} | ~{total_tokens} tok (est) | Msgs: {len(history)}"
        if current_conv_id:
            footer += f" | ID: {current_conv_id[:8]}"
        if show_dashboard:
            footer += " | 📊 dashboard ON"
        if current_show_internal:
            footer += " | [internal]"
        if last_interrupt_info:
            footer += " | [resume available]"
        return footer

    def _show_stats() -> None:
        total_tokens = sum(_count_tokens(m["content"]) for m in history)
        ctx_len = model_context_length
        used_pct = (total_tokens / ctx_len * 100) if ctx_len else 0
        bar_len = 30
        filled = int(bar_len * min(used_pct / 100, 1))
        bar = "█" * filled + "░" * (bar_len - filled)
        console.print(
            f"[dim]Messages: {len(history)} | ~Tokens: {total_tokens}/{ctx_len} "
            f"[{bar}] {used_pct:.0f}% (est)[/]"
        )

    # Command handlers
    async def cmd_help() -> None:
        help_text = """
[bold]Available commands:[/]
  /exit, /quit           - Exit the chat session
  /clear                 - Clear conversation history
  /undo                  - Remove the last exchange
  /save                  - Save current conversation to disk
  /load <id>             - Load a saved conversation
  /title <text>          - Set conversation title
  /model <name>          - Switch to a different model
  /models                - List models available on the server
  /system <prompt>       - Set a system prompt
  /apikey <key>          - Set the API key for this session
  /temp <value>          - Set temperature (0.0‑2.0)
  /max_tokens <n>        - Set maximum response tokens
  /top_p <value>         - Set nucleus sampling p
  /top_k <value>         - Set top‑k sampling
  /stop <token1,token2>  - Set stop tokens
  /timeout <seconds>     - Set request timeout
  /toggle_think          - Toggle thinking mode (Qwen)
  /compact on|off|N      - Blank-line compaction (on=0, off=preserve, N=max)
  /stats                 - Show conversation statistics
  /export [json|md|txt]  - Export conversation to file
  /history [N]           - Show last N messages
  /reset                 - Reset all state (clears history, saves current)
  /dashboard             - Toggle performance dashboard
  /config set <key> <value> - Set a chat config key
  /config get <key>      - Show a config value
  /config list           - List all chat config keys
  /resume [conv_id]      - Resume an interrupted conversation (latest checkpoint)
  /resume <checkpoint_id> - Resume from a specific checkpoint
  /checkpoints           - List checkpoints for current conversation
  /toggle_internal       - Toggle display of internal/draft models
  /help                  - Show this help message

[bold]Keyboard shortcuts:[/]
  Enter                  - Send message
  Ctrl+C                 - Interrupt generation
  Ctrl+D                 - Exit
  Ctrl+L                 - Clear screen
  Ctrl+E                 - Open editor for multi‑line input (if prompt_toolkit)
"""
        console.print(Panel(help_text, title="Help", border_style="cyan", box=box.ROUNDED))

    async def cmd_export(format: str = "json") -> None:
        if not history:
            print_warning("No conversation to export.")
            return
        ext = {"json": "json", "md": "md", "txt": "txt"}.get(format, "json")
        filename = f"{current_conv_id}.{ext}"
        path = Path.cwd() / filename

        # Wrap every `open()` in a thread. A three-format conditional is
        # still smaller than three near-identical async helpers, and the
        # closure captures `history`/`current_conv_id`/`session_title`
        # cleanly.
        def _write() -> None:
            if format == "json":
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"id": current_conv_id, "title": session_title, "messages": history},
                        f, indent=2,
                    )
            elif format == "md":
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"# {session_title}\n\n")
                    for msg in history:
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        f.write(f"**{role.capitalize()}:**\n{content}\n\n")
            else:
                with open(path, "w", encoding="utf-8") as f:
                    for msg in history:
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        f.write(f"[{role.upper()}]: {content}\n")

        try:
            await asyncio.to_thread(_write)
            print_success(f"Exported to {path}")
        except Exception as e:
            print_error(f"Export failed: {e}")

    async def cmd_history(n: int = 10) -> None:
        if not history:
            print_info("No history.")
            return
        msgs = history[-n:]
        for i, msg in enumerate(msgs, start=len(history)-len(msgs)+1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            preview = content[:80] + "..." if len(content) > 80 else content
            console.print(f"[dim]{i}. {role.capitalize()}:[/] {preview}")

    async def cmd_config_set(key: str, value: str) -> None:
        if value.lower() in ("true", "false"):
            val = value.lower() == "true"
        elif value.isdigit():
            val = int(value)
        elif value.replace(".", "", 1).isdigit():
            val = float(value)
        else:
            val = value
        _set_chat_config(key, val)
        print_success(f"Config {key} set to {val}")

    async def cmd_config_get(key: str) -> None:
        val = _get_chat_config().get(key)
        if val is None:
            print_warning(f"Config key '{key}' not set.")
        else:
            console.print(f"[bold]{key}:[/] {val}")

    async def cmd_config_list() -> None:
        cfg = _get_chat_config()
        if not cfg:
            print_info("No chat config values set.")
            return
        table = Table(title="Chat Config", box=box.ROUNDED)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        for k, v in cfg.items():
            table.add_row(k, str(v))
        console.print(table)

    async def cmd_resume(conv_id_opt: Optional[str] = None, checkpoint_id_opt: Optional[str] = None) -> None:
        resume_conv_id = conv_id_opt or current_conv_id
        if not resume_conv_id:
            print_error("No conversation ID provided and no current conversation.")
            return
        await _handle_resume(client, resume_conv_id, checkpoint_id_opt, current_max_tokens, current_temp, stream)
        print_info("Resume completed.")

    async def cmd_checkpoints() -> None:
        checkpoints = await _list_checkpoints(client, current_conv_id)
        if not checkpoints:
            print_info("No checkpoints found for this conversation.")
            return
        table = Table(title="Checkpoints", box=box.ROUNDED)
        table.add_column("ID", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Tokens", style="green")
        table.add_column("Segment", style="dim")
        for cp in checkpoints:
            table.add_row(
                cp.get("checkpoint_id", "?")[:8],
                cp.get("status", "unknown"),
                str(cp.get("tokens_generated", 0)),
                str(cp.get("segment_id", 0))
            )
        console.print(table)

    async def cmd_toggle_internal() -> None:
        nonlocal current_show_internal
        current_show_internal = not current_show_internal
        _set_chat_config("show_internal_models", current_show_internal)
        await refresh_models()
        state = "ON" if current_show_internal else "OFF"
        print_success(f"Internal model visibility {state}")

    # Prompt setup
    if _HAS_PROMPT_TOOLKIT:
        class DynamicCompleter(Completer):
            def __init__(self, get_models):
                self.get_models = get_models
            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                if text.startswith("/model "):
                    prefix = text[7:]
                    models = self.get_models()
                    for m in models:
                        if m.startswith(prefix):
                            yield Completion(m, start_position=-len(prefix))
                elif text.startswith("/"):
                    commands = ["exit","quit","clear","undo","save","load","title","model",
                                "models","system","apikey","temp","max_tokens","top_p","top_k",
                                "stop","timeout","toggle_think","compact","stats","export","history",
                                "reset","dashboard","config","help","resume","checkpoints",
                                "toggle_internal"]
                    for cmd in commands:
                        if cmd.startswith(text[1:]):
                            yield Completion(cmd + " ", start_position=0)
        def _get_model_names():
            return list(available_models)
        completer = DynamicCompleter(_get_model_names)

        bindings = KeyBindings()
        @bindings.add("c-l")
        def _(event):
            event.app.renderer.clear()
            console.clear()
        @bindings.add("c-e")
        def _(event):
            event.app.buffer.open_in_editor()

        session = PromptSession(
            history=FileHistory(str(_HIST_FILE)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
            key_bindings=bindings,
            enable_open_in_editor=True,
            multiline=False,
        )
        prompt_func = session.prompt
    else:
        prompt_func = lambda prompt_str: input(prompt_str)

    while True:
        try:
            footer = _get_footer()
            if not _HAS_PROMPT_TOOLKIT:
                console.print(f"[dim]{footer}[/]")
            prompt_str = _prompt()
            user_input = await asyncio.get_event_loop().run_in_executor(None, prompt_func, prompt_str)
            user_input = user_input.strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/]")
            if save and history:
                await save_conversation_async(current_conv_id, history, session_title)
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith("/"):
            cmd_parts = user_input[1:].strip().split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

            if cmd in ("exit", "quit"):
                if save and history:
                    await save_conversation_async(current_conv_id, history, session_title)
                break
            elif cmd == "help":
                await cmd_help()
                continue
            elif cmd == "clear":
                history.clear()
                print_success("Conversation history cleared")
                continue
            elif cmd == "undo":
                if history:
                    last_user = None
                    for i in range(len(history) - 1, -1, -1):
                        if history[i]["role"] == "user":
                            last_user = history[i]
                            break
                    if last_user:
                        idx = history.index(last_user)
                        history = history[:idx]
                        print_success("Undid last exchange")
                    else:
                        print_warning("Nothing to undo")
                else:
                    print_warning("Nothing to undo")
                continue
            elif cmd == "save":
                await save_conversation_async(current_conv_id, history, session_title)
                print_success(f"Conversation saved as '{current_conv_id}'")
                continue
            elif cmd == "load" and arg:
                data = await load_conversation_async(arg)
                if data:
                    history = data.get("messages", [])
                    session_title = data.get("title", "Loaded")
                    current_conv_id = arg
                    print_success(f"Loaded conversation '{arg}' with {len(history)} messages")
                else:
                    print_error(f"Conversation '{arg}' not found")
                continue
            elif cmd == "title" and arg:
                session_title = arg
                print_success(f"Title set to '{session_title}'")
                continue
            elif cmd == "model" and arg:
                if validate_model_name(arg):
                    if arg not in available_models:
                        all_models = await client.models()
                        if not isinstance(all_models, list):
                            all_models = []
                        matching = [m for m in all_models if m.get("id") == arg]
                        if matching and is_draft_model(matching[0]):
                            if not current_show_internal:
                                print_warning(
                                    f"Model '{arg}' is a draft/internal model. Use /toggle_internal to show it, "
                                    "or use --show-internal at startup."
                                )
                                continue
                            else:
                                await refresh_models()
                                if arg not in available_models:
                                    print_error(f"Model '{arg}' not found even with internal visibility.")
                                    continue
                        else:
                            print_warning(f"Model '{arg}' not found on server. Available: {', '.join(sorted(available_models)) or 'none'}")
                            continue
                    model_info = await _assert_model_ready(client, arg, current_show_internal)
                    if model_info is None:
                        _avail = sorted({m.get("id", "") for m in (await client.models() or []) if m.get("id")})
                        print_warning(
                            f"Model '{arg}' not in server catalog. "
                            f"Available: {', '.join(_avail) or 'none'}"
                        )
                        continue
                    _status = model_info.get("status") or "unknown"
                    if _status != "ready":
                        if _status == "loading":
                            print_warning(
                                f"Model '{arg}' is still loading. Requests will "
                                f"503 until it becomes ready."
                            )
                        elif _status == "degraded":
                            print_error(
                                f"Model '{arg}' workers are degraded. Requests "
                                f"will fail until workers recover."
                            )
                        elif _status == "error":
                            print_error(
                                f"Model '{arg}' failed to load. Requests will "
                                f"fail until it is reloaded."
                            )
                        else:
                            print_warning(
                                f"Model '{arg}' is not ready (status: {_status})."
                            )
                    current_model = arg
                    await refresh_models()
                    if not user_temperature_override and current_model in model_defaults:
                        current_temp = model_defaults[current_model].get("temperature", 0.7)
                    print_success(f"Switched to model: {current_model}")
                else:
                    print_error(f"Invalid model name: {arg!r}")
                continue
            elif cmd == "models":
                await refresh_models()
                all_models = await client.models()
                if not isinstance(all_models, list):
                    all_models = []
                user_models = filter_user_models(all_models, current_show_internal)
                if user_models:
                    table = Table(title="Available Models", box=box.ROUNDED)
                    table.add_column("ID", style="cyan")
                    table.add_column("Status", style="green")
                    table.add_column("Default T", style="yellow")
                    table.add_column("Tags", style="dim")
                    for m in user_models:
                        mid = m.get("id")
                        default_t = model_defaults.get(mid, {}).get("temperature", "?")
                        tags = ", ".join(m.get("tags", []))
                        table.add_row(mid, m.get("status", "unknown"), str(default_t), tags)
                    console.print(table)
                else:
                    print_info("No user-visible models loaded on server. (Use /toggle_internal to see internal ones.)")
                continue
            elif cmd == "apikey" and arg:
                client.cfg.api_key = arg
                print_success("API key updated. Subsequent requests will use the new key.")
                continue
            elif cmd == "temp" and arg:
                try:
                    new_temp = float(arg)
                    if 0.0 <= new_temp <= 2.0:
                        current_temp = new_temp
                        user_temperature_override = True
                        print_success(f"Temperature set to {current_temp}")
                    else:
                        print_error("Temperature must be 0.0‑2.0")
                except ValueError:
                    print_error("Invalid temperature value")
                continue
            elif cmd == "max_tokens" and arg:
                try:
                    new_max = int(arg)
                    if new_max >= 0:
                        current_max_tokens = new_max
                        _lbl = "unlimited" if new_max == 0 else str(new_max)
                        print_success(f"Max tokens set to {_lbl}")
                    else:
                        print_error("Max tokens must be >= 0")
                except ValueError:
                    print_error("Invalid max_tokens value")
                continue
            elif cmd == "top_p" and arg:
                try:
                    val = float(arg)
                    if 0.0 <= val <= 1.0:
                        current_top_p = val
                        print_success(f"Top‑p set to {current_top_p}")
                    else:
                        print_error("Top‑p must be 0.0‑1.0")
                except ValueError:
                    print_error("Invalid top‑p value")
                continue
            elif cmd == "top_k" and arg:
                try:
                    val = int(arg)
                    if val >= 1:
                        current_top_k = val
                        print_success(f"Top‑k set to {current_top_k}")
                    else:
                        print_error("Top‑k must be >= 1")
                except ValueError:
                    print_error("Invalid top‑k value")
                continue
            elif cmd == "stop" and arg:
                current_stop = [s.strip() for s in arg.split(",") if s.strip()]
                print_success(f"Stop tokens set to: {current_stop}")
                continue
            elif cmd == "system" and arg:
                current_system = arg.strip()
                print_success("System prompt updated")
                continue
            elif cmd == "timeout" and arg:
                try:
                    val = float(arg)
                    if val > 0:
                        current_timeout = val
                        print_success(f"Timeout set to {current_timeout}s")
                    else:
                        print_error("Timeout must be > 0")
                except ValueError:
                    print_error("Invalid timeout value")
                continue
            elif cmd == "toggle_think":
                current_no_think = not current_no_think
                state = "hidden" if current_no_think else "visible"
                print_success(f"Thinking tags are now {state}")
                continue
            elif cmd == "compact":
                # per-session compaction toggle. Writes to
                # cli.blank_lines (the key `Config.blank_lines` reads).
                cfg = get_config()
                arg_lower = arg.strip().lower()
                if arg_lower in ("on", ""):
                    cfg.set("cli.blank_lines", 0)
                    print_success("Blank-line compaction ON (0 blank lines)")
                elif arg_lower == "off":
                    cfg.set("cli.blank_lines", "preserve")
                    print_success("Blank-line compaction OFF (preserve raw output)")
                elif arg_lower.isdigit():
                    cfg.set("cli.blank_lines", int(arg_lower))
                    print_success(f"Blank-line compaction: max {arg_lower} blank line(s)")
                else:
                    print_error("Usage: /compact on|off|N")
                    continue
                save_config(cfg)
                continue
            elif cmd == "toggle_internal":
                await cmd_toggle_internal()
                continue
            elif cmd == "stats":
                _show_stats()
                continue
            elif cmd == "export":
                parts = arg.split(maxsplit=1)
                fmt = parts[0] if parts else "json"
                await cmd_export(fmt)
                continue
            elif cmd == "history":
                n = int(arg) if arg and arg.isdigit() else 10
                await cmd_history(n)
                continue
            elif cmd == "reset":
                if history:
                    if save:
                        await save_conversation_async(current_conv_id, history, session_title)
                    history.clear()
                    print_success("Session reset. Current conversation saved.")
                else:
                    print_warning("No history to reset.")
                continue
            elif cmd == "dashboard":
                show_dashboard = not show_dashboard
                _set_chat_config("show_dashboard", show_dashboard)
                state = "ON" if show_dashboard else "OFF"
                print_success(f"Dashboard {state}")
                if show_dashboard:
                    try:
                        metrics = await client._request("GET", "/v1/metrics")
                        if metrics:
                            vram_usage = metrics.get("vram", {}).get("utilization_pct", "?")
                            ram_usage = metrics.get("ram", {}).get("utilization_pct", "?")
                            queue_depth = metrics.get("engine", {}).get("queue", {}).get("size", 0)
                            console.print(f"[dim]VRAM: {vram_usage}% | RAM: {ram_usage}% | Queue: {queue_depth}[/]")
                    except Exception:
                        pass
                continue
            elif cmd == "config" and arg.startswith("set "):
                parts = arg.split(maxsplit=2)
                if len(parts) >= 3:
                    await cmd_config_set(parts[1], parts[2])
                else:
                    print_error("Usage: /config set <key> <value>")
                continue
            elif cmd == "config" and arg.startswith("get "):
                parts = arg.split(maxsplit=1)
                if len(parts) == 2:
                    await cmd_config_get(parts[1])
                else:
                    print_error("Usage: /config get <key>")
                continue
            elif cmd == "config" and arg == "list":
                await cmd_config_list()
                continue
            elif cmd == "resume":
                parts = arg.split()
                if len(parts) == 0:
                    await cmd_resume()
                elif len(parts) == 1:
                    if parts[0].startswith("cp_"):
                        await cmd_resume(None, parts[0])
                    else:
                        await cmd_resume(parts[0])
                elif len(parts) >= 2:
                    await cmd_resume(parts[0], parts[1])
                continue
            elif cmd == "checkpoints":
                await cmd_checkpoints()
                continue
            else:
                print_error(f"Unknown command: {cmd}. Type /help for available commands.")
                continue

        # Process user message
        model_info = await _assert_model_ready(client, current_model, current_show_internal)
        if model_info is None:
            all_models = await client.models()
            if not isinstance(all_models, list):
                all_models = []
            matching = [m for m in all_models if m.get("id") == current_model]
            if matching and is_draft_model(matching[0]) and not current_show_internal:
                print_error(
                    f"Model '{current_model}' is a draft/internal model and is not exposed. "
                    "Use /toggle_internal to allow selection."
                )
            else:
                _avail = sorted({m.get("id", "") for m in all_models if m.get("id")})
                print_error(
                    f"Model '{current_model}' is not in the server catalog. "
                    f"Available: {', '.join(_avail) or 'none'}"
                )
            continue
        _status = model_info.get("status") or "unknown"
        if _status != "ready":
            if _status == "loading":
                print_warning(
                    f"Model '{current_model}' is loading. Please wait and retry."
                )
            elif _status == "degraded":
                print_error(
                    f"Model '{current_model}' workers are degraded. "
                    f"Check server logs (journalctl -u local-llm-cloud)."
                )
            elif _status == "error":
                print_error(
                    f"Model '{current_model}' failed to load. Check server logs."
                )
            else:
                print_warning(
                    f"Model '{current_model}' is not ready (status: {_status}). Please wait."
                )
            continue

        history.append({"role": "user", "content": user_input})

        system_tokens = _count_tokens(current_system) if current_system else 0
        budget = _history_budget() - system_tokens
        total_history_tokens = sum(_count_tokens(m["content"]) + 10 for m in history)
        if total_history_tokens > budget:
            truncated, removed = _truncate_history(history, budget)
            if removed:
                print_warning(
                    f"History truncated: removed {removed} oldest messages "
                    f"(total ~tokens: {total_history_tokens}/{budget})."
                )
                history = truncated

        msgs = []
        if current_system:
            msgs.append({"role": "system", "content": current_system})
        msgs.extend(history)

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                start_time = time.perf_counter()
                full_response = ""
                is_error = False
                think_content = ""
                interrupted = False
                interrupt_info = None
                turn_stats: Optional[ServerStats] = None

                if stream:
                    console.print()
                    console.print("[bold cyan]🤖 AI:[/]")
                    parser = ThinkParser()
                    think_acc = ""
                    answer_acc = ""
                    printed_think_header = False
                    spinner_stopped = False

                    _bl = get_config().blank_lines
                    compactor = (
                        StreamCompactor(max_blank_lines=int(_bl))
                        if isinstance(_bl, int) else None
                    )

                    _result = StreamResult()

                    with console.status("[cyan]Waiting for model...", spinner="dots") as status:
                        client.on_stream_event = (
                            lambda m, _st=status: _st.update(f"[yellow]{m}")
                        )
                        try:
                            async for chunk in _stream_response(
                                    client, msgs, current_model, current_temp,
                                    current_max_tokens, current_top_p, current_top_k,
                                    current_stop, current_no_think, current_timeout,
                                    retry_count=attempt, max_retries=max_retries,
                                    _result=_result):
                                stripped = chunk.strip()
                                if stripped.startswith("[Error:"):
                                    is_error = True
                                    full_response = chunk
                                    break
                                inter = _is_interruption_marker(chunk)
                                if inter is not None:
                                    interrupted = True
                                    interrupt_info = inter
                                    full_response = chunk
                                    break
                                if not spinner_stopped:
                                    status.stop()
                                    spinner_stopped = True
                                clean_chunk = _strip_special_tokens(chunk)
                                if not clean_chunk:
                                    continue
                                think_part, normal_part = parser.push(clean_chunk)
                                if think_part:
                                    if not printed_think_header:
                                        console.print(Text("✣ Thinking", style="dim cyan"))
                                        printed_think_header = True
                                    console.print(think_part, end="", style="dim italic",
                                                   highlight=False, markup=False)
                                    think_acc += think_part
                                if normal_part:
                                    if printed_think_header and not answer_acc:
                                        console.print()
                                    if compactor is not None:
                                        normal_part = compactor.feed(normal_part)
                                        if not normal_part:
                                            continue
                                    console.print(normal_part, end="", highlight=False, markup=False)
                                    answer_acc += normal_part
                                    full_response += normal_part
                        finally:
                            client.on_stream_event = None
                            if not is_error and not interrupted:
                                t2, n2 = parser.flush()
                                if t2:
                                    console.print(t2, end="", style="dim italic",
                                                   highlight=False, markup=False)
                                    think_acc += t2
                                if n2:
                                    if compactor is not None:
                                        n2 = compactor.feed(n2)
                                    if n2:
                                        console.print(n2, end="", highlight=False, markup=False)
                                        answer_acc += n2
                                        full_response += n2
                            if compactor is not None:
                                compactor.finish()
                            if not spinner_stopped:
                                status.stop()

                    if (not is_error and not interrupted
                            and not full_response.strip() and not think_acc.strip()):
                        is_error = True
                        full_response = (
                            "The model returned an empty response — please try again."
                        )

                    if not is_error and not interrupted:
                        console.print()
                        turn_stats = _result.stats
                        _src = turn_stats.stats_source if turn_stats is not None else "unknown"
                        _stats_obj = turn_stats if turn_stats is not None else ServerStats()
                        console.print(Text(
                            f"\n{format_footer(_stats_obj, stats_source=_src)}",
                            style="dim",
                        ))
                    elif interrupted and interrupt_info is not None:
                        last_interrupt_info = {
                            "conv_id": current_conv_id,
                            "tokens": interrupt_info[1],
                        }
                        _kind = interrupt_info[0] if interrupt_info else None
                        if _kind in ("TRUNCATED", "STALLED"):
                            _clean = False
                            for _ in range(50):
                                print_notification(
                                    f"💡 Auto-continuing ({_kind})…"
                                )
                                _clean = await _handle_resume(
                                    client, current_conv_id, None,
                                    current_max_tokens, current_temp, stream,
                                )
                                if _clean:
                                    break
                            if _clean:
                                print_notification(
                                    "✅ Auto-continue finished cleanly."
                                )
                                last_interrupt_info = None
                            else:
                                print_notification(
                                    "⚠️ Still cut off after 50 auto-continues — "
                                    "type /resume to continue manually."
                                )
                        else:
                            print_notification(
                                "💡 Response interrupted. Type /resume to continue."
                            )
                else:
                    console.print()
                    console.print("[bold cyan]🤖 AI:[/]")
                    t0 = time.perf_counter()
                    with Progress(SpinnerColumn(), TextColumn("[cyan]Thinking...[/]"), transient=True) as progress:
                        task = progress.add_task("reasoning", total=None)
                        resp = await client.chat_completions(
                            messages=msgs,
                            model_id=current_model,
                            temperature=current_temp,
                            max_tokens=current_max_tokens,
                            top_p=current_top_p,
                            top_k=current_top_k,
                            stop=current_stop,
                            stream=False,
                            repetition_penalty=_get_repetition_penalty(),
                            **({"no_think": current_no_think} if current_no_think else {}),
                            **({"timeout_s": current_timeout} if current_timeout else {}),
                        )
                    total_time_s = time.perf_counter() - t0
                    result_obj = GenerationResult.from_chat_response(
                        resp, server_url=client.cfg.base_url,
                    )
                    result_obj.total_time_s = total_time_s
                    raw_text = resp["choices"][0]["message"]["content"]

                    parser = ThinkParser()
                    think_part, normal_part = parser.push(raw_text)
                    t2, n2 = parser.flush()
                    think_content = think_part + t2
                    full_response = normal_part + n2

                    if full_response.strip().startswith("[Error:"):
                        is_error = True
                    inter = _is_interruption_marker(full_response)
                    if inter is not None:
                        interrupted = True
                        interrupt_info = inter

                    if is_error:
                        console.print(Panel(
                            "The model didn't produce a response — please try again.",
                            title="⚠️ Error", border_style="yellow", padding=(0, 1), box=box.ROUNDED,
                        ))
                    elif interrupted:
                        console.print(Panel(
                            f"⚠️ Response interrupted after {interrupt_info[1]} tokens.",
                            title="Interrupted", border_style="yellow", padding=(0, 1), box=box.ROUNDED,
                        ))
                        last_interrupt_info = {"conv_id": current_conv_id, "tokens": interrupt_info[1]}
                        _kind = interrupt_info[0] if interrupt_info else None
                        if _kind in ("TRUNCATED", "STALLED"):
                            _clean = False
                            for _ in range(50):
                                print_notification(
                                    f"💡 Auto-continuing ({_kind})…"
                                )
                                _clean = await _handle_resume(
                                    client, current_conv_id, None,
                                    current_max_tokens, current_temp, stream,
                                )
                                if _clean:
                                    break
                            if _clean:
                                print_notification(
                                    "✅ Auto-continue finished cleanly."
                                )
                                last_interrupt_info = None
                            else:
                                print_notification(
                                    "⚠️ Still cut off after 50 auto-continues — "
                                    "type /resume to continue manually."
                                )
                        else:
                            print_notification("💡 Type /resume to continue from where it left off.")
                    else:
                        if think_content:
                            console.print(Text("\u2723 Thinking", style="dim cyan"))
                            console.print(Text(think_content, style="dim italic"))
                            console.print()
                        if full_response:
                            try:
                                console.print(Markdown(full_response))
                            except Exception:
                                console.print(Text(full_response))
                        else:
                            console.print("[yellow]No content received.[/]")

                    turn_stats = _stats_from_generation_result(result_obj, total_time_s)
                    console.print(Text(
                        f"\n{format_footer(turn_stats, stats_source=result_obj.stats_source)}",
                        style="dim",
                    ))

                if is_error or interrupted:
                    if is_error:
                        console.print(Panel(
                            full_response or "Server refused the request (out of resources).",
                            title="⚠️ Error", border_style="yellow",
                            padding=(0, 1), box=box.ROUNDED,
                        ))
                    history.pop()
                    if interrupted:
                        history.append({"role": "user", "content": user_input})
                        history.append({"role": "assistant", "content": full_response})
                    break

                history.append({"role": "assistant", "content": full_response})
                last_interrupt_info = None
                break

            except ServiceUnavailableError as e:
                _code = getattr(e, "error_code", None)
                if _code == "pool_degraded":
                    print_error(
                        "Server workers are degraded. Check server logs: "
                        "journalctl -u local-llm-cloud --since '5 minutes ago'"
                    )
                elif _code == "queue_deadline_exceeded":
                    print_error(
                        "Server request queue is saturated. Try a smaller "
                        "prompt or wait a moment."
                    )
                elif _code == "model_starting":
                    _r = float(getattr(e, "retry_after", 0) or 5.0)
                    print_error(f"Model is starting. Wait ~{_r:.0f}s and retry.")
                else:
                    print_error(f"Server unavailable: {e}")
                history.pop()
                break
            except NotFoundError as e:
                print_error(f"Model not found on server: {e}")
                history.pop()
                break
            except AuthError as e:
                print_error(f"Authentication failed: {e}")
                # `input()` blocks the event loop until the user types —
                # during that time no tokens can be received from any
                # other stream, no health checks fire, and the systemd
                # watchdog sees a stalled loop. Move to a thread.
                new_key = await asyncio.to_thread(
                    input,
                    "Enter new API key (or press Enter to skip): ",
                )
                new_key = new_key.strip()
                if new_key:
                    client.cfg.api_key = new_key
                    print_success("API key updated. Retrying...")
                    if history and history[-1].get("role") == "user":
                        history.pop()
                    history.append({"role": "user", "content": user_input})
                    continue
                else:
                    history.pop()
                    break
            except (APIError, ConnectionError, CircuitBreakerOpenError) as e:
                print_error(f"Request failed: {e}")
                history.pop()
                break
            except asyncio.TimeoutError:
                console.print(
                    "\n[red]⏱ Request timed out.[/] The model is still loading "
                    "or took too long. Your question was not saved."
                )
                history.pop()
                break
            except asyncio.CancelledError:
                console.print("\n[yellow]Generation cancelled[/]")
                if full_response and not is_error:
                    history.append({"role": "assistant", "content": full_response})
                else:
                    history.pop()
                break
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/]")
                if full_response and not is_error:
                    history.append({"role": "assistant", "content": full_response})
                else:
                    history.pop()
                break

        if is_error or interrupted:
            continue

# ------------------------------------------------------------------------------
# CLI command
# ------------------------------------------------------------------------------
@click.command("chat")
@click.argument("message", required=False)
@click.option("--model", "-m", help="Model to use (e.g., qwen3-8b). Omit to auto-select from server.")
@click.option("--conv-id", help="Conversation ID for persistence")
@click.option("--no-stream", is_flag=True, help="Disable streaming")
@click.option("--temperature", default=None, type=float, help="Temperature (0.0‑2.0). If not set, uses model default.")
@click.option("--max-tokens", default=0, type=int, help="Max tokens (0 = unlimited)")
@click.option("--top-p", default=0.95, type=float, help="Nucleus sampling p")
@click.option("--top-k", default=40, type=int, help="Top‑k sampling")
@click.option("--stop", help="Comma‑separated stop tokens")
@click.option("--no-think", is_flag=True, help="Suppress thinking tags (Qwen models)")
@click.option("--system", "system_prompt", help="System prompt")
@click.option("--timeout", type=float, help="Request timeout in seconds")
@click.option("--api-key", "api_key_override", envvar="LLM_API_KEY", help="API key (env: LLM_API_KEY)")
@click.option("--base-url", "base_url_override", help="Server base URL")
@click.option("--interactive", "-i", is_flag=True, help="Start interactive chat session")
@click.option("--save", is_flag=True, help="Save conversation to disk")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format (single message only)")
@click.option("--load", "load_conv", help="Load conversation by ID")
@click.option("--list-convs", is_flag=True, help="List saved conversations")
@click.option("--resume", is_flag=True, help="Resume the last saved conversation")
@click.option("--resume-conv", help="Resume a specific conversation by ID (interactive mode)")
@click.option("--show-internal", is_flag=True, help="Show internal/draft models in listings and allow selection")
@click.option("--blank-lines", default=None,
              help="Max blank lines in output: 0, 1, 2, ... or 'preserve' (persistent)")
@click.pass_context
def chat(
        ctx: click.Context,
        message: Optional[str],
        model: Optional[str],
        conv_id: Optional[str],
        no_stream: bool,
        temperature: Optional[float],
        max_tokens: int,
        top_p: float,
        top_k: int,
        stop: Optional[str],
        no_think: bool,
        system_prompt: Optional[str],
        timeout: Optional[float],
        api_key_override: Optional[str],
        base_url_override: Optional[str],
        interactive: bool,
        save: bool,
        output_json: bool,
        load_conv: Optional[str],
        list_convs: bool,
        resume: bool,
        resume_conv: Optional[str],
        show_internal: bool,
        blank_lines: Optional[str],
) -> None:
    if list_convs:
        convs = list_conversations()
        if not convs:
            print_info("No saved conversations.")
            return
        table = Table(title="Saved Conversations", box=box.ROUNDED)
        table.add_column("ID", style="cyan")
        table.add_column("Title", style="green")
        table.add_column("Messages", style="yellow")
        table.add_column("Created", style="dim")
        for c in convs:
            table.add_row(c["id"], c["title"][:40], str(c["message_count"]), c["created_at"][:19])
        console.print(table)
        return

    cfg = get_config(ctx.obj.get("config_path") if ctx.obj else None)

    # --blank-lines flag → writes to cli.blank_lines (persistent).
    # "preserve" is stored as the string; ints as ints. Negative ints clamp to 0.
    if blank_lines is not None:
        bl_value: Any
        if blank_lines.strip().lower() == "preserve":
            bl_value = "preserve"
        else:
            try:
                n = int(blank_lines)
                if n < 0:
                    print_warning(f"--blank-lines {n} clamped to 0")
                    n = 0
                bl_value = n
            except ValueError:
                print_error(
                    f"Invalid --blank-lines: {blank_lines!r} "
                    "(use an integer >= 0 or 'preserve')"
                )
                sys.exit(1)
        cfg.set("cli.blank_lines", bl_value)
        save_config(cfg)
        print_info(f"Blank-line compaction set to: {bl_value}")

    # `model` may be None here. Resolution order:
    #   1. --model flag
    #   2. cfg.model (may be "" for fresh install → triggers auto-select
    #      in run() below)
    if not model:
        model = cfg.model
    # NOTE: `validate_model_name(model)` is deferred to run() so the
    # auto-select path (empty model → pick from /v1/models) can fire
    # before validation rejects the empty string.

    stream = not no_stream
    if stream and not cfg.stream_enabled:
        stream = False

    stop_list: Optional[List[str]] = None
    if stop:
        stop_list = [s.strip() for s in stop.split(",") if s.strip()]

    loaded_history = None
    loaded_title = None

    if resume:
        convs = list_conversations()
        if convs:
            latest = convs[0]
            conv_data = load_conversation(latest["id"])
            if conv_data:
                loaded_history = conv_data.get("messages", [])
                loaded_title = conv_data.get("title", "Resumed")
                print_info(f"Resumed conversation '{loaded_title}' with {len(loaded_history)} messages.")
                conv_id = conv_data.get("id")
            else:
                print_warning("Could not load latest conversation.")
        else:
            print_warning("No saved conversations to resume.")

    if load_conv:
        conv_data = load_conversation(load_conv)
        if conv_data:
            loaded_history = conv_data.get("messages", [])
            loaded_title = conv_data.get("title", "Loaded Conversation")
            print_info(f"Loaded conversation '{loaded_title}' with {len(loaded_history)} messages.")
            conv_id = load_conv
        else:
            print_warning(f"Conversation '{load_conv}' not found.")

    if resume_conv:
        conv_id = resume_conv
        conv_data = load_conversation(resume_conv)
        if conv_data:
            loaded_history = conv_data.get("messages", [])
            loaded_title = conv_data.get("title", "Loaded for resume")
            print_info(f"Loaded conversation '{resume_conv}' with {len(loaded_history)} messages.")
        else:
            print_warning(f"Conversation '{resume_conv}' not found. Creating new.")

    client = _get_client_from_context(
        ctx, api_key_override=api_key_override, base_url_override=base_url_override
    )

    async def run() -> None:
        nonlocal_message = message
        # auto-select if no model configured (cfg.model == "").
        _resolved_model = model
        if not _resolved_model:
            print_info("No model configured — auto-selecting from server…")
            _auto = await _auto_select_model(client)
            if not _auto:
                print_error(
                    "Could not auto-select a model: no `status:ready` model on "
                    "the server, or the server is unreachable. Pass `--model <id>` "
                    "or set `inference.model` in your client config."
                )
                sys.exit(1)
            _resolved_model = _auto
            print_info(f"Auto-selected model: {_resolved_model}")
        if not validate_model_name(_resolved_model):
            print_error(f"Invalid model name: {_resolved_model!r}")
            print_info("Model name must contain only letters, numbers, hyphens, underscores, and periods.")
            sys.exit(1)

        try:
            if interactive:
                await interactive_chat(
                    client=client, model=_resolved_model, conv_id=conv_id,
                    stream=stream, temperature=temperature,
                    max_tokens=max_tokens, top_p=top_p, top_k=top_k,
                    stop=stop_list, no_think=no_think,
                    system_prompt=system_prompt, timeout=timeout,
                    save=save, initial_history=loaded_history,
                    initial_title=loaded_title,
                    show_internal=show_internal,
                )
            else:
                if not nonlocal_message and not sys.stdin.isatty():
                    nonlocal_message = sys.stdin.read().strip()
                if not nonlocal_message:
                    print_error("No message provided. Use --interactive or pipe input.")
                    sys.exit(1)

                default_temp = 0.7
                try:
                    models = await _get_available_models(client, show_internal)
                    for m in models:
                        if m.get("id") == _resolved_model:
                            dt = m.get("default_temperature")
                            if dt is not None and isinstance(dt, (int, float)):
                                default_temp = float(dt)
                            break
                except Exception:
                    pass
                final_temp = temperature if temperature is not None else default_temp

                await single_chat(
                    client=client,
                    message=nonlocal_message,
                    model=_resolved_model,
                    conv_id=conv_id,
                    stream=stream,
                    temperature=final_temp,
                    max_tokens=max_tokens if max_tokens is not None else 0,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop_list,
                    no_think=no_think,
                    system_prompt=system_prompt,
                    timeout=timeout,
                    save=save,
                    output_json=output_json,
                    show_internal=show_internal,
                )
        except AuthError as e:
            print_error(str(e))
            print_info(
                "Authentication failed. Set your API key with:\n"
                "  1. export LLM_API_KEY=<your-key>\n"
                "  2. llm config set server.api_key <your-key>\n"
                "  3. Use --api-key option"
            )
            sys.exit(1)
        except (APIError, ConnectionError, CircuitBreakerOpenError) as e:
            print_error(str(e))
            sys.exit(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/]")
            sys.exit(0)
        except LLMCloudError as e:
            print_error(str(e))
            sys.exit(1)
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            sys.exit(1)
        finally:
            await _cleanup_client_async(client)

    asyncio.run(run())
