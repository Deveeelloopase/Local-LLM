# local_llm_server/monitoring/

Metrics collection, performance analysis and the terminal dashboard.

| File | Purpose |
|---|---|
| `monitor.py` | The TUI behind `llm-monitor`: a curses application (`TUIMonitor`) that polls the server's HTTP API. Panels: Hardware, Performance, Memory, Drive, Models, Logs, Keys, Activity, Config, Ranking. `[Tab]` switches panels, `[q]` quits |
| `__main__.py` | `python -m local_llm_server.monitoring <subcommand>`: `tui` (dashboard), `check` (health and connectivity diagnostics), `predict` (predictive model ranking and alerts), `version` |
| `metrics_collector.py` | `MetricsCollector`: samples system/GPU/thermal data, keeps running statistics (Holt-Winters), `PredictiveAnomalyDetector`, optional file export |
| `telemetry.py` | `TelemetryCollector`: per-request metrics, Prometheus metrics (when `prometheus_client` is installed), adaptive sampling, file export |
| `performance_analyzer.py` | `PerformanceAnalyzer`: grades throughput against SLA targets, predicts violations, suggests a hardware tier (`evaluate_sla`) |
| `model_telemetry.py` | `ModelTelemetry` and `ModelRankingEngine`: per-model usage/latency, idle-model reaper (`model_idle_reaper`), warm-up helper |
| `__init__.py` | `MonitoringManager`, `get_monitoring_components()`, `shutdown_monitoring_components()` |

## Running the dashboard

```bash
llm-monitor                       # needs a real terminal; install the "tui" extra
llm-monitor --panel 4             # start on a given panel (0-9)
llm-monitor --url http://host:8080 --api-key KEY --refresh 2
python -m local_llm_server.monitoring check
```

`llm-monitor` options: `--config`, `--url`, `--refresh`, `--api-key`, `--master-key`, `--debug`, `--panel`. The URL falls back to `LLM_SERVER_URL`, then to the config file (`server_url`), then `http://localhost:8080`. Panels read `/v1/system`, `/v1/metrics`, `/v1/performance`, `/v1/logs` and related routes, so it can run on another machine as long as the API is reachable.

## Configuration (`monitoring:` section)

Metrics intervals, retention, anomaly thresholds, SLA targets, `prometheus_enabled` and `prometheus_port`. Prometheus export is a separate listener on `prometheus_port`; `/v1/metrics` on the main API returns JSON.
