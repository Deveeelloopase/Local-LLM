# engine/scheduling/

| File | Key classes | Purpose |
|---|---|---|
| `scheduler.py` | `Scheduler`, `ScheduledRequest`, `Priority` | Request scheduler with priority levels, fairness, AIMD (additive-increase / multiplicative-decrease) concurrency control toward a target tokens/s, and preemption |
| `__init__.py` | – | Lazy re-exports |

Config (`scheduler:`): `aimd_target_tps`, `aimd_min_batch`, `aimd_max_batch`, `aimd_interval_s`, `aimd_hysteresis`, `aimd_high_margin`, `aimd_low_margin`, `max_queue_per_priority`, `max_total_queue`, `default_timeout_s`, plus preemption settings (`PreemptionConfig`).

`max_queue_per_priority` and `max_total_queue` bound the queue.
