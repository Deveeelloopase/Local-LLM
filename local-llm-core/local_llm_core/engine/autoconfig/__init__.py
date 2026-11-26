"""Automatic model configuration.

Reads the facts that decide how a model should run (the GGUF header and the
hardware), derives every profile field, decides which models can be resident
together, verifies the result with an isolated benchmark and records it.

Typical use::

    from local_llm_core.engine.autoconfig import resolve, apply_to_config

    resolution = resolve(cfg, system_info=info)
    apply_to_config(cfg, resolution)

Modules
-------
``gguf_inspect``  header parser -> :class:`ModelFacts`
``machine``       hardware facts -> :class:`MachineFacts`
``planner``       field derivation and multi-model capacity plan
``calibrate``     isolated benchmarks, per-model records
``bench_worker``  one measurement in its own process
``store``         benchmark store and ``model_profiles.json``
``resolver``      the end-to-end entry point

Version: 1.0.0
"""
from .calibrate import CalibrationResult, Calibrator, Variant
from .gguf_inspect import GGUFError, ModelFacts, inspect_gguf
from .machine import MachineFacts
from .planner import AUTO_FIELDS, ModelPlan, Plan, Planner, PlannerSettings, is_auto, plan_profiles
from .resolver import (
    CalibrationSettings, Resolution, apply_to_config, collect_profiles,
    merged_profiles, resolve, summary_lines,
)
from .store import BenchStore, bench_path, plan_entry_for_path, plan_path, read_plan, state_dir

__version__ = "1.0.0"

__all__ = [
    "AUTO_FIELDS", "BenchStore", "CalibrationResult", "CalibrationSettings", "Calibrator",
    "GGUFError", "MachineFacts", "ModelFacts", "ModelPlan", "Plan", "Planner",
    "PlannerSettings", "Resolution", "Variant", "apply_to_config", "bench_path",
    "collect_profiles", "inspect_gguf", "is_auto", "merged_profiles", "plan_entry_for_path",
    "plan_path", "plan_profiles", "read_plan", "resolve", "state_dir", "summary_lines",
]
