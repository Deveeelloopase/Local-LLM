"""
model_advisor.py — picks a GGUF that can hit target_decode_tps on this hardware.

Runs in the parent, before the pool spawns. Reads only GGUF headers (cheap).
Reports; never modifies config, never downloads.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)


@dataclass
class ModelAdvice:
    model_path: str
    params_b_est: float
    stream_bytes: int
    kv_bytes_per_token: int
    predicted_decode_tps_low: float
    predicted_decode_tps_high: float
    feasible: bool
    bottleneck: str                    # "memory_bandwidth" | "kv_capacity" | "feasible"


class ModelAdvisor:
    def __init__(
        self,
        *,
        measured_bw_gbps: float,
        efficiency: float = 0.85,
        kv_ram_budget_mb: int = 4096,
        kv_quant: str = "q8_0",
    ) -> None:
        self._bw = float(measured_bw_gbps)
        self._eta = float(efficiency)
        self._kv_budget_mb = int(kv_ram_budget_mb)
        self._kv_quant = kv_quant

    def advise(
        self,
        gguf_paths: List[str],
        n_ctx: int,
        target_decode_tps: float,
    ) -> List[ModelAdvice]:
        """Return per-model advice, best-first by predicted TPS."""
        results: List[ModelAdvice] = []
        for path in gguf_paths:
            try:
                a = self._advise_one(path, n_ctx, target_decode_tps)
                if a is not None:
                    results.append(a)
            except Exception as exc:
                log.warning("ModelAdvisor: cannot analyse %s: %s", path, exc)
        # Best first: feasible models above target, then by predicted TPS desc
        results.sort(
            key=lambda a: (
                not a.feasible,
                -a.predicted_decode_tps_high,
            )
        )
        return results

    def _advise_one(
        self, gguf_path: str, n_ctx: int, target: float,
    ) -> Optional[ModelAdvice]:
        from local_llm_core.engine.pipelines.layer_streamer import read_gguf_meta

        p = Path(gguf_path)
        if not p.exists():
            return None
        meta = read_gguf_meta(p) or {}
        n_layer = int(meta.get("n_layer") or 0)
        n_kv = int(meta.get("n_kv_heads") or meta.get("n_head") or 0)
        head_dim = int(meta.get("head_dim") or 0)
        head_dim_v = int(meta.get("head_dim_v") or head_dim or 0)
        vocab = int(meta.get("vocab_size") or 0)
        n_embd = int(meta.get("n_embd") or 0)
        if n_layer <= 0 or n_kv <= 0 or head_dim <= 0:
            return None

        file_bytes = p.stat().st_size
        # The embedding table is read one row per token, not streamed.
        # Approx: vocab × n_embd × avg_bpw_for_embed (assume quant matches weights,
        # typically 4.5 for Q4_K_M). This shrinks the streamed fraction.
        embed_bytes = int(vocab * n_embd * 4.5 / 8) if vocab and n_embd else 0
        stream_bytes = max(file_bytes - embed_bytes, file_bytes // 2)

        # KV bytes per token from real GGUF metadata
        from local_llm_core.engine.inference.stream_guards import kv_bytes_per_token
        try:
            kv_bpt = int(kv_bytes_per_token(
                n_layer, n_kv, head_dim,
                head_dim_v or None,
                kv_quant_k=self._kv_quant,
            ))
        except Exception:
            return None

        kv_total_mb = int(kv_bpt * n_ctx / 1_048_576)
        if kv_total_mb > self._kv_budget_mb:
            return ModelAdvice(
                model_path=gguf_path,
                params_b_est=round(file_bytes / 1e9 / 0.56, 2),
                stream_bytes=stream_bytes,
                kv_bytes_per_token=kv_bpt,
                predicted_decode_tps_low=0.0,
                predicted_decode_tps_high=0.0,
                feasible=False,
                bottleneck="kv_capacity",
            )

        stream_gb = stream_bytes / 1e9
        if stream_gb <= 0:
            return None
        tps = self._bw * self._eta / stream_gb
        low = tps * 0.80
        high = tps * 1.00
        feasible = low >= target * 0.85          # allow 15% margin
        bottleneck = "feasible" if feasible else "memory_bandwidth"
        return ModelAdvice(
            model_path=gguf_path,
            params_b_est=round(file_bytes / 1e9 / 0.56, 2),
            stream_bytes=stream_bytes,
            kv_bytes_per_token=kv_bpt,
            predicted_decode_tps_low=round(low, 2),
            predicted_decode_tps_high=round(high, 2),
            feasible=feasible,
            bottleneck=bottleneck,
        )
