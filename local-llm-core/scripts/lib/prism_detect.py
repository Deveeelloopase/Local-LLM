#!/usr/bin/env python3
"""
scripts/lib/prism_detect.py — PRISM Model Classification Engine  v1.0.0
=========================================================================
Standalone, zero-dependency model classifier. Uses only Python stdlib
(struct, json, pathlib, sys, os, glob, hashlib, collections).

PRISM — Proactive Repository Intelligence for Smart Model-routing

Classification buckets:
  bf16        BF16/F32 safetensors — correct input for setup_env.py converter
  packed      Packed 1.58-bit (U8/I8 weights) — needs transformers pinned commit
  already_built  .gguf already present in model dir — skip build entirely
  gguf_file   Path provided IS a .gguf file — use directly
  unknown     Cannot determine — safest to attempt build and surface failure

Exit codes (also printed as first line to stdout for bash capture):
  0   Classification succeeded (print result to stdout)
  1   Usage error
  2   Path not found
  3   Cannot read config.json or safetensors

Algorithm — multi-layer evidence fusion:
  Layer 0  Direct .gguf file path → "gguf_file"          (instant)
  Layer 1  Existing ggml-model-*.gguf in dir → "already_built"  (glob)
  Layer 2  config.json quantization_config analysis       (JSON parse)
  Layer 3  Safetensors header dtype distribution          (struct unpack)
  Layer 4  config.json architecture string analysis       (string match)
  Layer 5  File size heuristic (packed models are tiny)   (stat)
  Layer 6  model_type field matching                      (JSON)
  Consensus: weighted vote — 3+ layers agreeing → confident result

Usage:
  python3 prism_detect.py <model_path_or_dir>
  python3 prism_detect.py <model_path_or_dir> --verbose
  python3 prism_detect.py <model_path_or_dir> --json
  python3 prism_detect.py <model_path_or_dir> --exit-code

Output (default): single line to stdout, one of:
  packed | bf16 | already_built | gguf_file | unknown
"""

from __future__ import annotations

import collections
import glob
import hashlib
import json
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Version ───────────────────────────────────────────────────────────────────
__version__ = "1.0.0"

# ── PRISM result constants ────────────────────────────────────────────────────
PACKED        = "packed"
BF16          = "bf16"
ALREADY_BUILT = "already_built"
GGUF_FILE     = "gguf_file"
UNKNOWN       = "unknown"

# ── Evidence weight per layer ─────────────────────────────────────────────────
# Higher = more reliable signal
LAYER_WEIGHTS = {
    "l0_direct_gguf":      100,  # direct .gguf file path — definitive
    "l1_built_gguf":       100,  # already-built output present — definitive
    "l2_quantization_cfg": 80,   # config.json quantization_config block
    "l3_safetensors_hdr":  90,   # safetensors header dtype distribution
    "l4_arch_string":      60,   # model architecture string analysis
    "l5_size_heuristic":   40,   # file size heuristic
    "l6_model_type":       70,   # model_type field
}

# ── Known dtype indicators ────────────────────────────────────────────────────
# U8/I8 in safetensors → packed ternary weights
PACKED_DTYPES  = {"U8", "I8", "INT8"}
# Standard floating-point dtypes → bf16/f32 training weights
FLOAT_DTYPES   = {"BF16", "F32", "F16", "FP32", "FP16", "FLOAT32", "FLOAT16"}

# ── Known packed model signatures in config.json ─────────────────────────────
PACKED_CONFIG_KEYS = {
    "quant_type", "bits", "pack_method", "weight_pack",
    "bitnet_mode", "use_packed_weights",
}
BITLINEAR_STRINGS = {
    "bitlinear", "bitnet", "ternary", "1.58bit", "1.58-bit",
    "trinary", "ternary_weight",
}


@dataclass
class PRISMEvidence:
    """Accumulated evidence from all layers for one classification."""
    layer:     str
    weight:    int
    verdict:   str
    detail:    str
    confident: bool = False


@dataclass
class PRISMResult:
    """Full PRISM classification result with evidence chain."""
    classification: str   = ""        # set after fuse_evidence()
    confidence:     float = 0.0       # 0.0 – 1.0
    evidence:       List[PRISMEvidence] = field(default_factory=list)
    model_path:     str = ""
    safetensors_files: List[str] = field(default_factory=list)
    dominant_dtype: Optional[str] = None
    dtype_distribution: Dict[str, int] = field(default_factory=dict)
    has_config:     bool = False
    config_model_type: Optional[str] = None
    config_architecture: Optional[str] = None   # architectures[0] from config.json
    gguf_files:     List[str] = field(default_factory=list)
    errors:         List[str] = field(default_factory=list)
    version:        str = __version__

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification":   self.classification,
            "confidence":       round(self.confidence, 3),
            "model_path":       self.model_path,
            "dominant_dtype":   self.dominant_dtype,
            "dtype_distribution": self.dtype_distribution,
            "has_config":       self.has_config,
            "config_model_type": self.config_model_type,
            "config_architecture": self.config_architecture,  # architectures[0] — used by bitnet_setup.sh
            "gguf_files":       self.gguf_files,
            "safetensors_files": self.safetensors_files,
            "errors":           self.errors,
            "evidence": [
                {
                    "layer":     e.layer,
                    "weight":    e.weight,
                    "verdict":   e.verdict,
                    "detail":    e.detail,
                    "confident": e.confident,
                }
                for e in self.evidence
            ],
            "version": self.version,
        }


# =============================================================================
# PRISM classifier
# =============================================================================
class PRISMClassifier:
    """
    Multi-layer evidence fusion classifier for BitNet model variants.
    Pure stdlib — no imports beyond what Python ships with.
    """

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"  [PRISM] {msg}", file=sys.stderr)

    # ── Layer 0: Direct .gguf file ────────────────────────────────────────────
    def _l0_direct_gguf(self, p: Path) -> Optional[PRISMEvidence]:
        if p.is_file() and p.suffix.lower() == ".gguf":
            return PRISMEvidence(
                layer="l0_direct_gguf",
                weight=LAYER_WEIGHTS["l0_direct_gguf"],
                verdict=GGUF_FILE,
                detail=f"Path is a .gguf file: {p.name}",
                confident=True,
            )
        return None

    # ── Layer 1: Already-built GGUF in directory ──────────────────────────────
    def _l1_built_gguf(self, p: Path, result: PRISMResult) -> Optional[PRISMEvidence]:
        if not p.is_dir():
            return None
        patterns = ["ggml-model-*.gguf", "ggml-*.gguf"]
        found = []
        for pat in patterns:
            found.extend(p.glob(pat))
        found = sorted(set(found), key=lambda f: f.stat().st_size, reverse=True)
        if found:
            result.gguf_files = [str(f) for f in found]
            return PRISMEvidence(
                layer="l1_built_gguf",
                weight=LAYER_WEIGHTS["l1_built_gguf"],
                verdict=ALREADY_BUILT,
                detail=f"{len(found)} built GGUF(s) found: {found[0].name}",
                confident=True,
            )
        # Also collect non-built .gguf files for informational purposes
        all_ggufs = list(p.glob("*.gguf"))
        if all_ggufs:
            result.gguf_files = [str(f) for f in all_ggufs]
        return None

    # ── Layer 2: config.json quantization_config ─────────────────────────────
    def _l2_quantization_cfg(
        self, p: Path, result: PRISMResult
    ) -> Optional[PRISMEvidence]:
        cfg_path = (p / "config.json") if p.is_dir() else (p.parent / "config.json")
        if not cfg_path.exists():
            return None
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as exc:
            result.errors.append(f"config.json parse error: {exc}")
            return None

        result.has_config = True
        result.config_model_type = cfg.get("model_type", "")

        # Capture architectures[0] — used by bitnet_setup.sh arch pre-check
        arch_list = cfg.get("architectures", [])
        if arch_list:
            result.config_architecture = arch_list[0]

        # Check quantization_config block
        qc = cfg.get("quantization_config", {})
        if isinstance(qc, dict) and qc:
            has_packed_key = bool(PACKED_CONFIG_KEYS & set(qc.keys()))
            cfg_str = json.dumps(cfg).lower()
            has_bitlinear = any(s in cfg_str for s in BITLINEAR_STRINGS)
            if has_packed_key and has_bitlinear:
                return PRISMEvidence(
                    layer="l2_quantization_cfg",
                    weight=LAYER_WEIGHTS["l2_quantization_cfg"],
                    verdict=PACKED,
                    detail=f"quantization_config has packed keys: "
                           f"{list(PACKED_CONFIG_KEYS & set(qc.keys()))}",
                )
            if has_packed_key:
                return PRISMEvidence(
                    layer="l2_quantization_cfg",
                    weight=LAYER_WEIGHTS["l2_quantization_cfg"] // 2,
                    verdict=PACKED,
                    detail=f"quantization_config present with keys: {list(qc.keys())[:5]}",
                )

        # Check torch_dtype field (bf16/float32 → training weights)
        torch_dtype = cfg.get("torch_dtype", "")
        if torch_dtype.lower() in ("bfloat16", "float32", "float16"):
            return PRISMEvidence(
                layer="l2_quantization_cfg",
                weight=LAYER_WEIGHTS["l2_quantization_cfg"] // 2,
                verdict=BF16,
                detail=f"torch_dtype={torch_dtype} — training-weight format",
            )

        return None

    # ── Layer 3: Safetensors header dtype distribution ────────────────────────
    def _l3_safetensors_hdr(
        self, p: Path, result: PRISMResult
    ) -> Optional[PRISMEvidence]:
        """
        Read the safetensors header (first 8 bytes = JSON length as uint64 LE,
        followed by JSON metadata).  Parse dtype for every tensor.
        No external dependencies — pure struct + json.
        """
        search_dir = p if p.is_dir() else p.parent
        st_files = sorted(
            search_dir.glob("*.safetensors"),
            key=lambda f: f.stat().st_size,
            reverse=True,
        )
        if not st_files:
            return None

        result.safetensors_files = [str(f) for f in st_files]
        dtype_counter: collections.Counter = collections.Counter()
        total_tensors = 0
        read_errors = 0

        # Read up to 3 shards for confidence (large sharded models)
        for st_file in st_files[:3]:
            d, n, err = self._read_safetensors_dtypes(st_file)
            dtype_counter.update(d)
            total_tensors += n
            if err:
                read_errors += 1
                result.errors.append(f"safetensors read: {err}")

        if not dtype_counter:
            return None

        result.dtype_distribution = dict(dtype_counter)

        # Dominant dtype = most common by tensor count
        dominant = dtype_counter.most_common(1)[0][0]
        result.dominant_dtype = dominant

        self._log(
            f"dtype distribution ({total_tensors} tensors): "
            + ", ".join(f"{dt}={n}" for dt, n in dtype_counter.most_common(5))
        )

        # Decision: U8/I8 presence → packed
        packed_count = sum(dtype_counter.get(d, 0) for d in PACKED_DTYPES)
        float_count  = sum(dtype_counter.get(d, 0) for d in FLOAT_DTYPES)
        packed_ratio = packed_count / max(total_tensors, 1)
        float_ratio  = float_count  / max(total_tensors, 1)

        if packed_ratio >= 0.3:
            return PRISMEvidence(
                layer="l3_safetensors_hdr",
                weight=LAYER_WEIGHTS["l3_safetensors_hdr"],
                verdict=PACKED,
                detail=(
                    f"{packed_count}/{total_tensors} tensors are packed dtype "
                    f"({dominant}), ratio={packed_ratio:.2f}"
                ),
                confident=packed_ratio >= 0.5,
            )
        if float_ratio >= 0.7:
            return PRISMEvidence(
                layer="l3_safetensors_hdr",
                weight=LAYER_WEIGHTS["l3_safetensors_hdr"],
                verdict=BF16,
                detail=(
                    f"Dominant dtype {dominant} ({float_count}/{total_tensors} tensors, "
                    f"ratio={float_ratio:.2f})"
                ),
                confident=float_ratio >= 0.9,
            )
        return None

    @staticmethod
    def _read_safetensors_dtypes(
        path: Path,
    ) -> Tuple[Dict[str, int], int, Optional[str]]:
        """
        Parse safetensors file header.  Returns (dtype_counts, n_tensors, error).
        Max header read: 100 MB (safeguard against corrupt files).
        """
        MAX_HDR = 100 * 1024 * 1024
        dtype_counts: Dict[str, int] = {}
        n = 0
        try:
            with open(path, "rb") as f:
                raw_len = f.read(8)
                if len(raw_len) < 8:
                    return {}, 0, "file too small"
                hdr_len = struct.unpack("<Q", raw_len)[0]
                if hdr_len > MAX_HDR:
                    return {}, 0, f"header too large ({hdr_len} bytes)"
                hdr_bytes = f.read(hdr_len)
                if len(hdr_bytes) < hdr_len:
                    return {}, 0, "truncated header"
            hdr = json.loads(hdr_bytes)
            for key, val in hdr.items():
                if key == "__metadata__":
                    continue
                if isinstance(val, dict):
                    dt = val.get("dtype", "UNKNOWN").upper()
                    dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
                    n += 1
            return dtype_counts, n, None
        except Exception as exc:
            return {}, 0, str(exc)

    # ── Layer 4: Architecture string analysis ─────────────────────────────────
    def _l4_arch_string(
        self, p: Path, result: PRISMResult
    ) -> Optional[PRISMEvidence]:
        cfg_path = (p / "config.json") if p.is_dir() else (p.parent / "config.json")
        if not cfg_path.exists():
            return None
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            return None

        # Check architectures list + model_type
        arch_list = cfg.get("architectures", [])
        model_type = cfg.get("model_type", "")
        combined = " ".join(arch_list + [model_type]).lower()

        if any(s in combined for s in ("bitnetforsequenceclassification",
                                        "bitnettextgeneration",
                                        "bitnetforcausallm")):
            return PRISMEvidence(
                layer="l4_arch_string",
                weight=LAYER_WEIGHTS["l4_arch_string"],
                verdict=PACKED,
                detail=f"BitNet architecture class found: {arch_list}",
            )
        if "llama" in combined or "mistral" in combined or "falcon" in combined:
            if "bitnet" not in combined and "1.58" not in combined:
                return PRISMEvidence(
                    layer="l4_arch_string",
                    weight=LAYER_WEIGHTS["l4_arch_string"] // 2,
                    verdict=BF16,
                    detail=f"Standard LLM architecture: {arch_list}",
                )
        return None

    # ── Layer 5: File-size heuristic ──────────────────────────────────────────
    def _l5_size_heuristic(
        self, p: Path, result: PRISMResult
    ) -> Optional[PRISMEvidence]:
        """
        Packed 1.58-bit 2B model is ~1.0 GB in safetensors.
        BF16 2B model is ~4.0 GB.
        This is an unreliable signal alone but useful as tie-breaker.
        """
        search_dir = p if p.is_dir() else p.parent
        st_files = list(search_dir.glob("*.safetensors"))
        if not st_files:
            return None

        total_size = sum(
            f.stat().st_size for f in st_files
            if f.is_file()
        )
        total_gb = total_size / (1024 ** 3)
        self._log(f"Total safetensors size: {total_gb:.2f} GB")

        if 0.1 < total_gb < 1.8:
            return PRISMEvidence(
                layer="l5_size_heuristic",
                weight=LAYER_WEIGHTS["l5_size_heuristic"],
                verdict=PACKED,
                detail=f"Small total size ({total_gb:.2f} GB) consistent with packed weights",
            )
        if total_gb > 3.0:
            return PRISMEvidence(
                layer="l5_size_heuristic",
                weight=LAYER_WEIGHTS["l5_size_heuristic"],
                verdict=BF16,
                detail=f"Large total size ({total_gb:.2f} GB) consistent with BF16 weights",
            )
        return None

    # ── Layer 6: model_type field ─────────────────────────────────────────────
    def _l6_model_type(
        self, p: Path, result: PRISMResult
    ) -> Optional[PRISMEvidence]:
        cfg_path = (p / "config.json") if p.is_dir() else (p.parent / "config.json")
        if not cfg_path.exists():
            return None
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            return None
        model_type = cfg.get("model_type", "").lower()
        if model_type in ("bitnet", "bitnet_b158"):
            return PRISMEvidence(
                layer="l6_model_type",
                weight=LAYER_WEIGHTS["l6_model_type"],
                verdict=PACKED,
                detail=f"model_type={model_type}",
            )
        if model_type in ("llama", "mistral", "falcon", "gpt2", "bloom", "opt"):
            return PRISMEvidence(
                layer="l6_model_type",
                weight=LAYER_WEIGHTS["l6_model_type"] // 2,
                verdict=BF16,
                detail=f"model_type={model_type} — standard architecture",
            )
        return None

    # ── Evidence fusion ───────────────────────────────────────────────────────
    def _fuse_evidence(
        self, evidence: List[PRISMEvidence]
    ) -> Tuple[str, float]:
        """
        Weighted vote across all evidence layers.
        Returns (classification, confidence).
        """
        if not evidence:
            return UNKNOWN, 0.0

        # Any definitive (confident=True) evidence wins immediately
        for e in evidence:
            if e.confident:
                return e.verdict, 1.0

        # Weighted vote
        votes: Dict[str, int] = {}
        total_weight = 0
        for e in evidence:
            votes[e.verdict] = votes.get(e.verdict, 0) + e.weight
            total_weight += e.weight

        if not votes or total_weight == 0:
            return UNKNOWN, 0.0

        winner = max(votes, key=lambda k: votes[k])
        winner_weight = votes[winner]
        confidence = winner_weight / total_weight

        # Require ≥50% weight agreement for a non-unknown result
        if confidence < 0.45:
            return UNKNOWN, confidence
        return winner, confidence

    # ── Main classify entry point ─────────────────────────────────────────────
    def classify(self, path_str: str) -> PRISMResult:
        """
        Run all PRISM layers and return a PRISMResult.
        """
        p = Path(path_str).expanduser().resolve()
        result = PRISMResult(model_path=str(p))

        if not p.exists():
            result.classification = UNKNOWN
            result.errors.append(f"Path does not exist: {p}")
            return result

        evidence: List[PRISMEvidence] = []

        # Run all layers
        layers = [
            self._l0_direct_gguf(p),
            self._l1_built_gguf(p, result),
            self._l2_quantization_cfg(p, result),
            self._l3_safetensors_hdr(p, result),
            self._l4_arch_string(p, result),
            self._l5_size_heuristic(p, result),
            self._l6_model_type(p, result),
        ]

        for ev in layers:
            if ev is not None:
                evidence.append(ev)
                self._log(
                    f"Layer {ev.layer}: verdict={ev.verdict} "
                    f"weight={ev.weight} confident={ev.confident} — {ev.detail}"
                )

        result.evidence = evidence
        classification, confidence = self._fuse_evidence(evidence)
        result.classification = classification
        result.confidence = confidence

        self._log(
            f"PRISM result: {classification} "
            f"(confidence={confidence:.2%}, "
            f"layers_fired={len(evidence)}/{len(layers)})"
        )
        return result


# =============================================================================
# CLI entry point
# =============================================================================
def _usage() -> None:
    print(
        f"Usage: prism_detect.py <model_path_or_dir> [--verbose] [--json] [--exit-code]\n"
        f"       prism_detect.py --version\n"
        f"\nReturns: packed | bf16 | already_built | gguf_file | unknown",
        file=sys.stderr,
    )


def main() -> int:
    args = sys.argv[1:]
    if not args or "--help" in args or "-h" in args:
        _usage()
        return 1
    if "--version" in args:
        print(f"prism_detect.py v{__version__}")
        return 0

    verbose    = "--verbose"   in args or "-v" in args
    as_json    = "--json"      in args
    use_exit   = "--exit-code" in args
    arch_only  = "--arch-only" in args   # NEW: print only architectures[0]

    path_args = [a for a in args if not a.startswith("-")]
    if not path_args:
        print("Error: no model path provided", file=sys.stderr)
        _usage()
        return 1

    model_path = path_args[0]

    classifier = PRISMClassifier(verbose=verbose)
    result = classifier.classify(model_path)

    if arch_only:
        # Used by bitnet_setup.sh: python3 prism_detect.py --arch-only /path
        print(result.config_architecture or "")
        return 0
    elif as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.classification)

    if result.errors and verbose:
        for err in result.errors:
            print(f"  [PRISM] WARNING: {err}", file=sys.stderr)

    if use_exit:
        # Exit codes for bash consumption
        exit_map = {
            ALREADY_BUILT: 10,
            GGUF_FILE:     11,
            BF16:          20,
            PACKED:        30,
            UNKNOWN:       40,
        }
        return exit_map.get(result.classification, 40)

    return 0


if __name__ == "__main__":
    sys.exit(main())
