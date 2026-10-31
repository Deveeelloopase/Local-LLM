# local_llm_client/web/

The browser chat UI.

| File | Purpose |
|---|---|
| `web_ui.py` | A single large Python module whose `HTML` constant is a complete self-contained page (HTML + CSS + JS, no CDN dependencies) |
| `__init__.py` | `get_web_ui()` returns the HTML; `HTML` is re-exported at package level |

## How it is served

`local-llm-server` imports `HTML` (`from local_llm_client.web.web_ui import HTML`) at import time and returns it for `GET /` and `GET /demo`. If the client package is missing, the server serves a small placeholder page. Because the HTML is loaded once at import, **restart the server after editing `web_ui.py`**.

The page contains a `__API_KEY_PLACEHOLDER__` token that the server replaces with the configured static API key when one is set; see the security notes in the [server README](../../../local-llm-server/README.md#security-notes).

## Editing tips

- The server only calls `.format(version=...)` on the page when it contains the literal `{version}`; keep that in mind when adding braces.
- There are no JS unit tests; verify in a browser against a running server.
