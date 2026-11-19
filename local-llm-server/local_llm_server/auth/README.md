# local_llm_server/auth/

API-key storage and verification.

| File | Purpose |
|---|---|
| `key_store.py` | `KeyStore` and its backends — `SQLiteKeyStore`, `PostgreSQLKeyStore`, `MySQLKeyStore` — created by `create_key_store(config)`; `KeyStatus` (active/revoked/…), per-key `TokenBucket` rate limiting, usage counters, request logging with retention, soft-delete, a key cache, and batched usage updates |
| `__init__.py` | Re-exports the public names |

## How keys are used

- Clients send `X-API-Key: <key>`. The server asks the key store to validate it and apply the key's rate limit (`api.api_key_rate_limit_rps`, `api.api_key_rate_limit_capacity`, `api.api_key_concurrency_limit`).
- Key administration (`/admin/keys/create`, `revoke`, `activate`, `update`, `delete`, `list`, `stats`) requires `X-Master-Key: <api.master_key>`. Model management routes only need a normal API key.
- If `api.key_store_enabled` is false and a static `api.key` is set, that single key is accepted. If neither is set the server runs without authentication.
- Keys are managed with the client CLI: `llm keys create | list | revoke | activate | delete | stats`.

## Configuration (`api:` section)

`storage` (`sqlite` / `postgresql` / `mysql`), `key_store_db` (SQLite path), `host`, `port`, `database`, `username`, `password` (DB connection), `pool_size`, `max_overflow`, `pool_timeout`, `pool_recycle`, `retry_*`, `log_requests`, `request_log_retention_days`, `soft_delete_retention_days`, `key_cache_size`, `prune_interval_hours`, `batch_usage_updates`, `usage_batch_*`, `ip_rate_limit`, `admin_rate_limit`.

Use `${ENV_VAR}` for passwords. Do not commit real values.

## Tests

Covered by `tests/test_all.py` in `local-llm-server`.
