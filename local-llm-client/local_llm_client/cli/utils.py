"""
local_llm_client/cli/utils.py

Production‑ready CLI utilities for Local LLM Cloud.

Enhancements over the previous version:

- All stream output now uses Rich `console.print` for consistent styling.
- `LiveDisplay` has been corrected to store content in a `Text` instance.
- `progress_bar` now yields a `ProgressBarHandle` object that supports
  `advance(n)` without exposing the internal task ID.
- Added helper functions:
    • `format_duration` / `format_bytes`  – human‑readable sizes and times.
    • `format_token_count` – token counts with contextual hint.
    • `print_hr` – horizontal rule.
    • `print_boxed` – text inside a simple border.
    • `print_dict` / `print_list` – recursive display of nested data.
    • `truncate_text` / `indent_text` – text manipulation for CLI output.
    • `colorize_json` – syntax‑highlighted JSON (requires pygments; falls back to plain).
    • `log_verbose` – conditional verbose output.
    • `ask_choice` / `ask_string` – rich prompts for interactive input.
    • `display_kv_table` – quick key‑value table from a dict.
    • `create_header` – render a branded header.
    • `status_spinner` – async context manager for long async operations.
- All original functions and classes are preserved and improved where needed.
- No hardcoded paths – pure output utilities.
"""

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Any, AsyncIterator, Callable, Dict, Generator, Iterable, Iterator,
    List, Optional, Sequence, Tuple, Union,
)

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn,
    TaskID,
)
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich import box
from rich.prompt import Confirm, Prompt
from rich.live import Live

# Global console instance
console = Console()

# Optional syntax‑highlighted JSON
try:
    from pygments import highlight
    from pygments.lexers import JsonLexer
    from pygments.formatters import TerminalFormatter
    _HAS_PYGMENTS = True
except ImportError:
    _HAS_PYGMENTS = False


# ------------------------------------------------------------------------------
# Basic coloured output
# ------------------------------------------------------------------------------
def print_error(message: str, *args: Any, exc_info: bool = False) -> None:
    """Print an error message in red.

    Accepts printf-style ``%``-formatting via ``*args`` and an optional
    ``exc_info=True`` to also dump the current traceback. The signature
    mirrors ``logging.Logger.error`` so call sites read consistently and
    an unexpected keyword never crashes the CLI.
    """
    if args:
        try:
            message = message % args
        except Exception:
            # Fallback: never crash the error printer itself.
            message = f"{message} {' '.join(str(a) for a in args)}"
    console.print(f"[bold red]Error:[/] {message}")
    if exc_info:
        import traceback
        console.print(traceback.format_exc(), style="dim red")


def print_success(message: str) -> None:
    """Print a success message with a checkmark."""
    console.print(f"[bold green]✓[/] {message}")


def print_warning(message: str) -> None:
    """Print a warning message with a warning symbol."""
    console.print(f"[bold yellow]⚠[/] {message}")


def print_info(message: str) -> None:
    """Print an informational message."""
    console.print(f"[bold cyan]ℹ[/] {message}")


def print_notification(message: str) -> None:
    """Print a notification message with a light bulb icon."""
    console.print(f"[bold yellow]💡[/] {message}")


def log_verbose(message: str, verbose: bool = False) -> None:
    """Print a message only if verbose mode is on."""
    if verbose:
        console.print(f"[dim]{message}[/]")


# ------------------------------------------------------------------------------
# Horizontal rule / boxed text
# ------------------------------------------------------------------------------
def print_hr(style: str = "dim") -> None:
    """Print a horizontal rule."""
    console.rule(style=style)


def print_boxed(content: str, title: str = "", style: str = "cyan") -> None:
    """Print content inside a simple box."""
    panel = Panel(content, title=title, border_style=style)
    console.print(panel)


# ------------------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------------------
def format_duration(seconds: float) -> str:
    """Return human‑readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs}s"
    else:
        hours, remainder = divmod(int(seconds), 3600)
        mins, secs = divmod(remainder, 60)
        return f"{hours}h {mins}m {secs}s"


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human‑readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} EB"


def format_token_count(tokens: int, context_window: Optional[int] = None) -> str:
    """Format token count, optionally showing context usage."""
    if context_window and context_window > 0:
        pct = tokens / context_window * 100
        return f"{tokens} / {context_window} ({pct:.0f}%)"
    return str(tokens)


# ------------------------------------------------------------------------------
# Text manipulation
# ------------------------------------------------------------------------------
def truncate_text(text: str, max_len: int = 120, suffix: str = "…") -> str:
    """Truncate a string to a maximum length."""
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)] + suffix


def indent_text(text: str, indent: int = 4) -> str:
    """Indent every line of a multi‑line string."""
    prefix = " " * indent
    return "\n".join(prefix + line for line in text.split("\n"))


# ------------------------------------------------------------------------------
# Table formatting
# ------------------------------------------------------------------------------
def format_table(
    columns: List[Tuple[str, str]],
    rows: List[List[Any]],
    title: str = "",
) -> None:
    """
    Display a rich table.

    Args:
        columns: List of (column_name, style) tuples.
        rows: List of rows, each a list of cell values.
        title: Optional table title.
    """
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title=title or None,
    )
    for col_name, col_style in columns:
        table.add_column(col_name, style=col_style)
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    console.print(table)


def display_kv_table(
    data: Dict[str, Any],
    title: str = "",
    key_style: str = "cyan",
    value_style: str = "green",
) -> None:
    """Display a two‑column table from a dictionary."""
    rows = [[str(k), str(v)] for k, v in data.items()]
    format_table(
        [("Key", key_style), ("Value", value_style)],
        rows,
        title=title,
    )


# ------------------------------------------------------------------------------
# JSON output
# ------------------------------------------------------------------------------
def print_json(data: Union[Dict, List, Any], indent: int = 2) -> None:
    """
    Pretty print JSON data, with optional syntax highlighting.

    Args:
        data: Python object.
        indent: Number of spaces for indentation.
    """
    raw = json.dumps(data, indent=indent, default=str, ensure_ascii=False)
    if _HAS_PYGMENTS:
        try:
            colored = highlight(raw, JsonLexer(), TerminalFormatter())
            console.print(colored)
            return
        except Exception:
            pass
    console.print(raw)


def colorize_json(data: Union[Dict, List, Any]) -> None:
    """Print JSON with syntax highlighting (fallback to plain)."""
    print_json(data)
    # identical implementation to print_json; kept for backward compatibility if needed


# ------------------------------------------------------------------------------
# Streaming display
# ------------------------------------------------------------------------------
def streaming_display(
    generator: Union[Iterator[str], Generator[str, None, None]],
    title: str = "Response",
) -> str:
    """
    Display streaming tokens in real time (synchronous).

    Args:
        generator: Yields text chunks.
        title: Title displayed before the stream.

    Returns:
        Full concatenated response.
    """
    console.print(f"\n[bold cyan]{title}:[/] ", end="")
    full_text = ""
    try:
        for chunk in generator:
            console.print(chunk, end="", highlight=False)
            full_text += chunk
        console.print()
    except KeyboardInterrupt:
        print_warning("Stream interrupted by user")
    except Exception as e:
        print_error(f"Stream error: {e}")
    return full_text


async def async_streaming_display(
    async_gen: AsyncIterator[str],
    title: str = "Response",
) -> str:
    """
    Display streaming tokens in real time (asynchronous).

    Args:
        async_gen: An async iterator yielding chunks.
        title: Title displayed before the stream.

    Returns:
        Full concatenated response.
    """
    console.print(f"\n[bold cyan]{title}:[/] ", end="")
    full_text = ""
    try:
        async for chunk in async_gen:
            console.print(chunk, end="", highlight=False)
            full_text += chunk
        console.print()
    except asyncio.CancelledError:
        print_warning("Stream cancelled")
        raise
    except Exception as e:
        print_error(f"Stream error: {e}")
    return full_text


# ------------------------------------------------------------------------------
# Markdown & syntax
# ------------------------------------------------------------------------------
def render_markdown(text: str) -> None:
    """Render markdown in the terminal using Rich."""
    if not text:
        return
    console.print(Markdown(text))


def print_syntax(
    code: str,
    language: str = "python",
    theme: str = "monokai",
    line_numbers: bool = True,
) -> None:
    """
    Print code with syntax highlighting.

    Args:
        code: Source code string.
        language: Programming language.
        theme: Pygments theme name.
        line_numbers: Show line numbers.
    """
    syntax = Syntax(code, language, theme=theme, line_numbers=line_numbers)
    console.print(syntax)


# ------------------------------------------------------------------------------
# Panel
# ------------------------------------------------------------------------------
def print_panel(
    content: str,
    title: Optional[str] = None,
    border_style: str = "cyan",
) -> None:
    """Display content inside a Rich panel."""
    console.print(Panel(content, title=title, border_style=border_style))


# ------------------------------------------------------------------------------
# Nested data display
# ------------------------------------------------------------------------------
def print_dict(d: Dict[str, Any], indent: int = 2) -> None:
    """Display a dictionary as a nested tree."""
    text = _dict_to_text(d, indent)
    console.print(text)


def _dict_to_text(d: Dict[str, Any], indent: int, level: int = 0) -> str:
    lines = []
    prefix = " " * (level * indent)
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}[bold]{key}[/]:")
            lines.append(_dict_to_text(value, indent, level + 1))
        elif isinstance(value, list):
            lines.append(f"{prefix}[bold]{key}[/]: [")
            for item in value:
                lines.append(f"{prefix}  {item}")
            lines.append(f"{prefix}]")
        else:
            lines.append(f"{prefix}[bold]{key}[/]: {value}")
    return "\n".join(lines)


def print_list(items: Sequence[Any], bullet: str = "•") -> None:
    """Display a list as bullet points."""
    for item in items:
        console.print(f"  {bullet} {item}")


# ------------------------------------------------------------------------------
# Header / logo
# ------------------------------------------------------------------------------
def create_header(version: str = "") -> None:
    """Print a stylish header for the CLI."""
    content = (
        r"[bold violet]"
        r"╔═══════════════════════════════════════════════╗" "\n"
        r"║         ⚡  LOCAL LLM CLOUD                   ║" "\n"
        r"║    Private AI – Your Data, Your Hardware      ║" "\n"
        r"╚═══════════════════════════════════════════════╝"
        r"[/]"
    )
    if version:
        content += f"\n[dim]Version: {version}[/]"
    console.print(content)


# ------------------------------------------------------------------------------
# Spinner (sync context manager)
# ------------------------------------------------------------------------------
@contextmanager
def with_spinner(message: str) -> Generator[None, None, None]:
    """
    Context manager for showing a spinner.

    Usage:
        with with_spinner("Loading models..."):
            time.sleep(2)
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn(f"[progress.description]{{task.description}}"),
        transient=True,
        console=console,
    )
    with progress:
        task = progress.add_task(message, total=None)
        yield
        progress.update(task, completed=True)


def spinner_decorator(message: str):
    """
    Decorator to wrap a function with a spinner.

    Usage:
        @spinner_decorator("Processing...")
        def long_running_task():
            time.sleep(2)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with with_spinner(message):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ------------------------------------------------------------------------------
# Status spinner (async context manager)
# ------------------------------------------------------------------------------
@asynccontextmanager
async def status_spinner(message: str) -> AsyncIterator[None]:
    """Async spinner that updates a shared Live display."""
    with Live(Text(f"[bold cyan]{message}[/] …"), refresh_per_second=10, transient=True) as live:
        yield
        live.update(Text(f"[bold green]✓[/] {message}"))


# ------------------------------------------------------------------------------
# LiveDisplay
# ------------------------------------------------------------------------------
class LiveDisplay:
    """
    Context manager for updating a live display (e.g., metrics dashboard).

    Usage:
        with LiveDisplay() as display:
            display.update("Status: Running")
            time.sleep(2)
            display.update("Status: Done")
    """

    def __init__(self, refresh_per_second: int = 10, auto_refresh: bool = False) -> None:
        # Use a Text object for the live content for consistent styling
        self._live = Live(
            Text(""),
            refresh_per_second=refresh_per_second,
            auto_refresh=auto_refresh,
        )
        self._content = Text("")

    def __enter__(self) -> "LiveDisplay":
        self._live.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._live.__exit__(*args)

    def update(self, content: str) -> None:
        """Update the live display with new content."""
        self._content = Text.from_markup(content)
        self._live.update(self._content)


# ------------------------------------------------------------------------------
# Progress bar (sync context manager)
# ------------------------------------------------------------------------------
class ProgressBarHandle:
    """Wrapper returned by `progress_bar` for easy increment."""

    def __init__(self, progress: Progress, task_id: TaskID) -> None:
        self._progress = progress
        self._task_id = task_id

    def advance(self, amount: int = 1) -> None:
        """Advance the bar by `amount` steps."""
        self._progress.update(self._task_id, advance=amount)


@contextmanager
def progress_bar(
    description: str,
    total: int = 100,
) -> Generator[ProgressBarHandle, None, None]:
    """
    Context manager for a progress bar.

    Usage:
        with progress_bar("Downloading", total=500) as bar:
            for _ in range(500):
                bar.advance(1)

    Args:
        description: Task description.
        total: Total number of steps.

    Yields:
        ProgressBarHandle with `.advance(n)` method.
    """
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task(description, total=total)
        handle = ProgressBarHandle(progress, task_id)
        yield handle
        progress.update(task_id, completed=total)


# ------------------------------------------------------------------------------
# Exit helpers
# ------------------------------------------------------------------------------
def exit_with_error(message: str, code: int = 1) -> None:
    """Print error and exit with given code."""
    print_error(message)
    sys.exit(code)


def exit_with_success(message: str, code: int = 0) -> None:
    """Print success message and exit."""
    print_success(message)
    sys.exit(code)


# ------------------------------------------------------------------------------
# Interactive prompts
# ------------------------------------------------------------------------------
def confirm(question: str, default: bool = False) -> bool:
    """Ask a yes/no question. Returns True for yes."""
    return Confirm.ask(question, default=default)


def ask_choice(
    question: str,
    choices: List[str],
    default: Optional[str] = None,
) -> str:
    """Prompt the user to select one from a list of choices."""
    from rich.prompt import Prompt
    while True:
        console.print(f"[bold cyan]?[/] {question}")
        for i, c in enumerate(choices, 1):
            console.print(f"  {i}. {c}")
        answer = Prompt.ask("Enter number", default=str(choices.index(default) + 1) if default else None)
        try:
            idx = int(answer) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        except (ValueError, IndexError):
            pass
        print_warning(f"Please enter a number between 1 and {len(choices)}.")


def ask_string(
    question: str,
    default: str = "",
    password: bool = False,
) -> str:
    """Prompt the user for a string."""
    return Prompt.ask(f"[bold cyan]?[/] {question}", default=default, password=password)


# ------------------------------------------------------------------------------
# Clear screen
# ------------------------------------------------------------------------------
def clear_screen() -> None:
    """Clear the terminal screen."""
    console.clear()
