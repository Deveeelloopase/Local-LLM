# local_llm_server/memory/

Chat-history ("conversation") storage.

| File | Purpose |
|---|---|
| `conversation_store.py` | Conversation stores for several backends (in-memory, SQLite, PostgreSQL, MySQL) behind one interface; created from the `conversation:` section of `config.yaml` |
| `__init__.py` | Re-exports |

## What it is used for

- `serving/api_server.py` keeps a history per **session** (`X-Session-ID` header or `llm_session` cookie). On each request it merges the stored history with the messages the client sent according to `history_mode`:
  - `auto` (default): if the client sends at most one non-system message, it is **appended** to the stored history; if it sends more, the `replace` merge is used,
  - `append`: always append to the stored history,
  - `replace`: the client's messages are authoritative; the longest overlap between the end of the stored history and the start of the incoming messages is not duplicated.
- Resumable generation: checkpoints let `/v1/chat/resume` continue an interrupted stream.

Agents that manage their own history should send `history_mode: "replace"` so the server never adds stored turns to the prompt.

> This package is the **chat-history** store of the server. Do not confuse it with `local_llm_core/memory/`, which manages VRAM/RAM for the engine.
> 