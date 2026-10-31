from __future__ import annotations

"""
engine/utils/config_loader.py — Centralised Configuration Schemas (Engine)
===========================================================================
Library‑only; no file I/O.  Pydantic V2 models for all engine sub‑systems.
Each module imports the models it needs directly from this file.

Version: 1.0.0 – Added comprehensive BitNet configuration validation.
"""


from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Any, Dict, List, Literal, Optional


# -------------------------------------------------------------------------
# Speculative Decoding
# -------------------------------------------------------------------------

class SpeculativeConfig(BaseModel):
    """Speculative decoding parameters (used by BitNet and GGUF engines)."""
    enabled: bool = True
    draft_model_id: str = ""
    target_model_id: str = ""
    k: int = Field(default=8, ge=1, le=32)
    min_k: int = Field(default=3, ge=1)
    max_k: int = Field(default=16, ge=1)
    adaptive_k: bool = True
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    acceptance_target_high: float = Field(default=0.75, ge=0.0, le=1.0)
    acceptance_target_low: float = Field(default=0.50, ge=0.0, le=1.0)
    draft_timeout_s: float = Field(default=30.0, ge=1.0)
    verify_timeout_s: float = Field(default=60.0, ge=1.0)
    max_fallback_retries: int = Field(default=5, ge=1)
    enable_exact_verification: bool = False
    draft_repo: str = "Qwen/Qwen3-0.5B-GGUF"
    draft_filename: str = "Qwen3-0.5B-Instruct-Q4_K_M.gguf"
    draft_size_mb: int = Field(default=310, ge=1)
    vocab_size: int = Field(default=151936, ge=1)

    @model_validator(mode='after')
    def validate_k_bounds(self) -> 'SpeculativeConfig':
        if self.min_k > self.max_k:
            raise ValueError("min_k cannot exceed max_k")
        if self.k < self.min_k or self.k > self.max_k:
            raise ValueError(f"k must be between min_k ({self.min_k}) and max_k ({self.max_k})")
        return self


# -------------------------------------------------------------------------
# BitNet Configuration
# -------------------------------------------------------------------------

class BitNetProfileConfig(BaseModel):
    """Single BitNet model profile.

    v9.1 — Fixed defaults that were silently wrong:
        device             : "cpu"   (was "cuda" — BitNet ternary SIMD kernels are CPU-only)
        num_gpu_layers     : 0       (was -1     — -1 means "all layers on GPU"; 0 = CPU-only)
        use_flash_attention_2: False (was True   — requires flash_attn package, usually absent)
        dtype              : "bfloat16" (was "float16" — bfloat16 matches BitNet training dtype)
    Added:
        model_type         : Optional[str] — accepts the field from config.yaml models.profiles
                             without Pydantic validation errors (extra field passthrough).
        prompt_format      : Optional[str] — chat template hint (e.g. "phi3" for bitnet-b1.58)
    """
    model_config = {"extra": "ignore"}   # silently ignore unknown YAML keys (e.g. ram_required_mb)

    id: str
    path: str
    format: str = "bitnet"
    autoload: bool = False
    quantization: Literal["1bit", "4bit", "8bit", "mixed"] = "1bit"
    quantized: bool = True

    # BUG 3 FIX — was "cuda": BitNet ternary SIMD kernels run on CPU only
    device: str = "cpu"
    # BUG 3 FIX — was "float16": bfloat16 matches BitNet training dtype
    dtype: str = "bfloat16"
    # BUG 3 FIX — was -1 (all layers on GPU): 0 = CPU-only for BitNet weights
    num_gpu_layers: int = Field(default=0, ge=-1)

    context_length: int = Field(default=4096, ge=1)
    use_custom_kernels: bool = True
    kernel_backend: Optional[Literal["cuda", "rocm", "metal", "cpu"]] = None

    # BUG 3 FIX — was True: flash_attn may not be installed; False is safe default
    use_flash_attention_2: bool = False
    trust_remote_code: bool = True
    priority: int = Field(default=10, ge=0)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Extra passthrough fields present in config.yaml models.profiles
    model_type: Optional[str] = None          # "bitnet" or None
    prompt_format: Optional[str] = None       # "phi3", "llama3", etc.

    # tokenizer_path — separate directory for tokenizer when using a pre-built GGUF.
    # For profiles where path is a .gguf file (e.g. ggml-model-i2_s.gguf),
    # the safetensors directory can't be used for tokenizer loading.
    # Set this to the HuggingFace model directory (e.g. bitnet-b1.58-2B-4T)
    # so the tokenizer can be loaded independently of the GGUF file.
    # If None the loader will try model_name_or_path, then skip tokenizer
    # gracefully (llama-server handles tokenization internally).
    tokenizer_path: Optional[str] = None

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Enforce CPU for BitNet. Warn clearly if caller tried to set GPU."""
        import logging
        _log = logging.getLogger(__name__)
        if v in ("cuda", "mps", "rocm"):
            _log.warning(
                "BitNetProfileConfig: device=%r overridden to 'cpu'. "
                "BitNet 1.58-bit ternary SIMD kernels are CPU-only. "
                "GPU offload is only available for non-BitNet layers via bitnet.cpp.",
                v,
            )
            return "cpu"
        return v

    @field_validator("num_gpu_layers")
    @classmethod
    def validate_gpu_layers(cls, v: int) -> int:
        """Clamp to 0 for BitNet — ternary weights are always on CPU."""
        import logging
        _log = logging.getLogger(__name__)
        if v > 0:
            _log.warning(
                "BitNetProfileConfig: num_gpu_layers=%d overridden to 0. "
                "BitNet ternary weights cannot be offloaded to GPU. "
                "Use bitnet.cpp (bitnet_n_gpu_layers) for non-BitNet KV-cache GPU offload.",
                v,
            )
            return 0
        return v


class BitNetSystemConfig(BaseModel):
    """Global BitNet system settings – fully validated for production."""
    enabled: bool = True
    default_quantization: Literal["1bit", "4bit", "8bit", "mixed"] = "1bit"
    use_custom_kernels: bool = True
    auto_detect_kernel_backend: bool = True
    force_kernel_backend: Optional[Literal["cuda", "rocm", "metal", "cpu"]] = None
    memory_fraction: float = Field(default=0.9, ge=0.1, le=1.0)
    enable_speculative_decoding: bool = True
    enable_context_caching: bool = True
    max_loaded_models: int = Field(default=2, ge=1)
    fallback_to_cpu_on_oom: bool = True
    load_retries: int = Field(default=3, ge=1)
    default_model_id: Optional[str] = None

    # BitNet.cpp server settings (used by loader and backend)
    bitnet_cpp_port: int = Field(default=18080, ge=1, le=65535)
    bitnet_n_threads: int = Field(default=0, ge=0, description="0 = auto-detect")
    bitnet_n_gpu_layers: int = Field(default=0, ge=0, description="0 = CPU-only")
    bitnet_force_backend: Optional[Literal["transformers", "bitnet_cpp"]] = None

    # Advanced BitNet.cpp options (for tuning)
    bitnet_host: str = Field(default="127.0.0.1")
    bitnet_n_batch: int = Field(default=512, ge=1, description="Batch size for prompt processing")
    bitnet_n_ctx: int = Field(default=4096, ge=64, description="Context window size")
    bitnet_parallel: int = Field(default=1, ge=1, description="Number of parallel requests")

    @field_validator("bitnet_n_threads")
    @classmethod
    def validate_threads(cls, v: int) -> int:
        if v < 0:
            raise ValueError("bitnet_n_threads must be >= 0")
        return v

    @field_validator("bitnet_n_batch", "bitnet_n_ctx")
    @classmethod
    def positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("value must be >= 1")
        return v

    @model_validator(mode='after')
    def validate_port_conflict(self) -> 'BitNetSystemConfig':
        # No conflict validation needed here; can be extended
        return self


# -------------------------------------------------------------------------
# Throughput Optimizer & NVMe KV Cache
# -------------------------------------------------------------------------

class NVMeKVCacheConfig(BaseModel):
    """NVMe‑backed KV cache settings."""
    enabled: bool = True
    dir: str = "/tmp/llm_kv_cache"
    max_size_gb: float = Field(default=10.0, ge=0.1)
    max_entries: int = Field(default=0, ge=0)  # 0 = unlimited
    prefix_len: int = Field(default=512, ge=64)


class ThroughputOptimizerConfig(BaseModel):
    """Throughput optimizer section configuration."""
    enabled: bool = True
    max_batch_slots: int = Field(default=8, ge=1)
    queue_depth: int = Field(default=64, ge=1)
    kv_cache_priority_boost_enabled: bool = True
    kv_cache_priority_boost: int = Field(default=10, ge=0)
    speculative: SpeculativeConfig = Field(default_factory=SpeculativeConfig)
    kv_cache: NVMeKVCacheConfig = Field(default_factory=NVMeKVCacheConfig)


# -------------------------------------------------------------------------
# Multi‑GPU Configuration
# -------------------------------------------------------------------------

class MultiGPUConfig(BaseModel):
    """Multi‑GPU tensor splitting configuration."""
    enabled: bool = True
    strategy: Literal["auto", "even", "manual", "disabled"] = "auto"
    manual_ratios: Optional[List[float]] = None
    exclude_gpu_ids: List[int] = Field(default_factory=list)
    min_free_vram_mb: int = Field(default=512, ge=0)
    bandwidth_weight: float = Field(default=0.5, ge=0.0, le=1.0,
                                    description="0 = pure VRAM, 1 = pure bandwidth")

    @field_validator("manual_ratios")
    @classmethod
    def check_manual_ratios_positive(cls, v):
        if v is not None:
            if any(r <= 0 for r in v):
                raise ValueError("All manual_ratios must be > 0")
        return v


# -------------------------------------------------------------------------
# Async Weight Pipeline Configuration
# -------------------------------------------------------------------------

class AsyncWeightPipelineConfig(BaseModel):
    """Async weight pipeline (mmap prefetch) configuration."""
    enabled: bool = True
    prefetch_depth: int = Field(default=3, ge=1)
    release_behind: int = Field(default=2, ge=0)
    header_estimate_percent: float = Field(default=0.5, ge=0.0, le=100.0)
    max_header_bytes: int = Field(default=2 * 1024 * 1024, ge=1024)
    idle_sleep_ms: int = Field(default=5, ge=1)
    thread_join_timeout_s: float = Field(default=2.0, ge=0.1)


# -------------------------------------------------------------------------
# Inference Engine Configuration
# -------------------------------------------------------------------------

class InferenceConfig(BaseModel):
    """Inference engine configuration (section `inference`)."""
    max_batch_size: int = Field(default=4, ge=1)
    target_tps: float = Field(default=8.0, gt=0.0)

    rate_limit_rps: float = Field(default=10.0, gt=0.0)
    rate_limit_max_keys: int = Field(default=1000, ge=1)

    # 0 = no total wall-clock deadline. The engine's own stall detection
    # (first_token_timeout_floor_s + inter_token_stall_s) is authoritative;
    # this value bounds the WHOLE stream, which is almost never what a
    # long-running local model needs.
    default_timeout_s: float = Field(default=0.0, ge=0.0)

    # Operator backstop on total generated tokens per request. 0 = unlimited
    # (the default). Set to a positive integer only when you must bound
    # worst-case KV / memory cost at the cost of occasionally truncating a
    # legitimately long answer.
    max_total_tokens: int = Field(default=0, ge=0)

    context_safety_tokens: int = Field(default=256, ge=0)

    circuit_breaker_failure_threshold: int = Field(default=5, ge=1)
    circuit_breaker_recovery_s: float = Field(default=60.0, ge=1.0)

    autotune_interval_s: float = Field(default=300.0, ge=60.0)

    aimd_min_batch: int = Field(default=1, ge=1)
    aimd_interval_s: float = Field(default=5.0, ge=1.0)

    min_executor_workers: int = Field(default=4, ge=1)
    executor_max_queue: int = Field(default=64, ge=1)

    metrics_history_size: int = Field(default=500, ge=10)

    context_cache_bytes: int = Field(default=256 * 1024 * 1024, ge=1024 * 1024)
    context_cache_prefix_len: int = Field(default=512, ge=64)

    ngram_n: int = Field(default=4, ge=1)
    ngram_max_size: int = Field(default=50_000, ge=100)

    request_queue_maxsize: int = Field(default=2000, ge=10)

    first_token_timeout_floor_s: float = Field(default=120.0, ge=5.0)
    first_token_timeout_multiplier: float = Field(default=3.0, ge=1.0, le=10.0)
    # Post-first-token gap that triggers a stall error.
    inter_token_stall_s: float = Field(default=60.0, ge=5.0)
    # EWMA smoothing factor for the prefill-throughput tracker.
    prefill_ewma_alpha: float = Field(default=0.05, gt=0.0, le=0.5)
    # Below this many tokens of response headroom, refuse generation
    # instead of silently clamping to 1 token (Issue 6).
    min_response_tokens: int = Field(default=64, ge=1)

    # ─────────────────────────────────────────────────────────────────
    # Generation policy (config-driven; 0 = unlimited, -1 = auto)
    # ─────────────────────────────────────────────────────────────────
    # Client-facing default when the caller sends max_tokens<=0.  0 = no
    # token-count cap; termination comes from EOS / stop / loop guard /
    # stall detection, all of which are hardware-independent.
    default_max_tokens: int = Field(default=0, ge=0)
    # -1 = auto (server computes per-request from n_ctx & segment_size ×
    # max_segments).  0 = always segmented.  N>0 = segmented only if the
    # caller asks for > N.  Never a cap; a *routing* decision.
    long_request_threshold_tokens: int = Field(default=-1, ge=-1)
    # 0 = unlimited (recommended).  Legacy callers who set a positive
    # value keep the old behaviour; nothing is hard-coded.
    segment_size: int = Field(default=512, ge=64)
    max_segments: int = Field(default=0, ge=0)
    health_check_interval_segments: int = Field(default=1, ge=1)
    default_stop: List[str] = Field(default_factory=list)

    # ─────────────────────────────────────────────────────────────────
    # Streaming / heartbeat (Fix Pack core)
    # ─────────────────────────────────────────────────────────────────
    # Interruptible, observable prompt evaluation.  Set false to restore
    # the exact legacy prefill path (one uninterruptible llm() call).
    chunked_prefill: bool = True
    # Control-frame interval while the model is silent. 0 = off.
    heartbeat_interval_s: float = Field(default=5.0, ge=0.0)
    # Chunked-prefill tuning — all adaptive from measured prefill tps.
    chunked_prefill_min_tokens: int = Field(default=256, ge=64)
    chunked_prefill_chunk_min: int = Field(default=32, ge=8)
    chunked_prefill_chunk_max: int = Field(default=256, ge=32)
    chunked_prefill_chunk_seconds: float = Field(default=3.0, gt=0.0)

    # ─────────────────────────────────────────────────────────────────
    # Anti-repetition sampling (production-grade, 5 knobs)
    # ─────────────────────────────────────────────────────────────────
    presence_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    frequency_penalty: float = Field(default=0.2, ge=0.0, le=2.0)
    repeat_penalty: float = Field(default=1.05, ge=1.0, le=2.0)
    # When temperature is 0 (greedy decode) penalties are a no-op in
    # llama.cpp anyway, but passing them causes silent drift on some
    # builds.  Keep True unless you have a reason.
    penalty_apply_only_when_sampling: bool = True

    # ─────────────────────────────────────────────────────────────────
    # Loop guards (semantic termination; the actual safety net)
    # ─────────────────────────────────────────────────────────────────
    loop_guard_enabled: bool = True
    loop_guard_min_cover: int = Field(default=800, ge=64)
    loop_guard_window: int = Field(default=8000, ge=512)
    loop_guard_check_every: int = Field(default=48, ge=8)
    loop_guard_line_min_chars: int = Field(default=40, ge=8)
    loop_guard_line_repeats: int = Field(default=3, ge=2)
    loop_guard_heading_min_chars: int = Field(default=12, ge=4)
    loop_guard_heading_repeats: int = Field(default=3, ge=2)
    loop_guard_shingle_n: int = Field(default=8, ge=3)
    loop_guard_shingle_window: int = Field(default=6000, ge=256)
    loop_guard_shingle_ratio: float = Field(default=0.45, ge=0.1, le=1.0)
    loop_guard_shingle_min_words: int = Field(default=600, ge=64)

    # ─────────────────────────────────────────────────────────────────
    # Second-layer guard for "asks user then answers itself"
    # ─────────────────────────────────────────────────────────────────
    solicitation_guard_enabled: bool = True
    solicitation_grace_tokens: int = Field(default=64, ge=1)
    solicitation_min_arm_tokens: int = Field(default=256, ge=1)
    # Empty = use builtin families.  Each entry is a Python regex
    # (case-insensitive) matched against the tail of the last completed
    # line.  Extend without code change.
    solicitation_families: List[str] = Field(default_factory=list)


# -------------------------------------------------------------------------
# Layer Streamer Configuration
# -------------------------------------------------------------------------

class LayerStreamerConfig(BaseModel):
    """Fractional layer pipeline / OOM recovery settings."""
    vram_overhead_mb: int = Field(default=80, ge=0)
    layer_overcommit_factor: float = Field(default=1.05, ge=1.0, le=2.0)
    oom_max_attempts: int = Field(default=5, ge=1, le=20)
    oom_layer_reduction_factor: float = Field(default=0.75, ge=0.1, lt=1.0)
    ctx_reduction_steps: List[int] = Field(default=[1, 2, 4])
    min_context: int = Field(default=256, ge=64)
    # Sliding window for llama.cpp's penalty samplers.  Default 64 is far
    # too short to suppress paragraph-level loops; 2048 reaches most of a
    # typical reply.  Set at Llama() construction time (not per request).
    last_n_tokens_size: int = Field(default=2048, ge=64, le=8192)
    prompt_lookup_enabled: bool = False  # <— new, default False
    prompt_lookup_num_tokens: int = Field(2, ge=1, le=16)
    n_threads_batch: int = Field(0, ge=0)
    model_config = {"extra": "allow"}  # <— log warning on unknown keys


# -------------------------------------------------------------------------
# Drive Pipeline Configuration (NVMe‑backed inference)
# -------------------------------------------------------------------------

class ContextCacheConfig(BaseModel):
    max_bytes: int = Field(default=256 * 1024 * 1024, ge=1024 * 1024)
    prefix_len: int = Field(default=512, ge=64)


class LayerPrefetcherConfig(BaseModel):
    prefetch_depth: int = Field(default=3, ge=1)
    max_cache_bytes: int = Field(default=2 * 1024**3, ge=1024**2)
    max_workers: int = Field(default=2, ge=1)


class CPUThrottlerConfig(BaseModel):
    thread_ratio: float = Field(default=0.50, ge=0.1, le=1.0)
    thread_limit: int = Field(default=0, ge=0)


class ConcurrencyConfig(BaseModel):
    initial_capacity: int = Field(default=1, ge=1)


class ThroughputControllerConfig(BaseModel):
    target_tps: float = Field(default=8.0, gt=0.0)
    min_batch: int = Field(default=1, ge=1)
    max_batch: int = Field(default=32, ge=1)
    window_sec: float = Field(default=300.0, ge=10.0)


class NVMeAdvisorConfig(BaseModel):
    ram_threshold: float = Field(default=0.70, ge=0.1, le=1.0)


class MemoryPressureConfig(BaseModel):
    vram_target_pct: float = Field(default=0.95, ge=0.5, le=0.99)
    ram_target_pct: float = Field(default=0.75, ge=0.5, le=0.95)


class SysctlConfig(BaseModel):
    dirty_bytes: int = Field(default=256 * 1024 * 1024, ge=0)
    dirty_background_bytes: int = Field(default=64 * 1024 * 1024, ge=0)
    vfs_cache_pressure: int = Field(default=50, ge=0)
    numa_balancing: int = Field(default=0, ge=0, le=1)


class DrivePipelineConfig(BaseModel):
    """Top‑level drive pipeline configuration."""
    target_vram_pct: float = Field(default=0.95, ge=0.5, le=0.99)
    target_ram_pct: float = Field(default=0.75, ge=0.5, le=0.95)
    target_tps: float = Field(default=8.0, gt=0.0)
    cpu_thread_ratio: float = Field(default=0.50, ge=0.1, le=1.0)
    context_cache: ContextCacheConfig = Field(default_factory=ContextCacheConfig)
    layer_prefetcher: LayerPrefetcherConfig = Field(default_factory=LayerPrefetcherConfig)
    cpu_throttler: CPUThrottlerConfig = Field(default_factory=CPUThrottlerConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    throughput_controller: ThroughputControllerConfig = Field(default_factory=ThroughputControllerConfig)
    nvme_advisor: NVMeAdvisorConfig = Field(default_factory=NVMeAdvisorConfig)
    memory_pressure: MemoryPressureConfig = Field(default_factory=MemoryPressureConfig)
    sysctl: SysctlConfig = Field(default_factory=SysctlConfig)
    use_madvise: bool = True
    use_direct_io: bool = False
    model_dir: str = "./models"
    swap_dir: str = "/tmp/llm_swap"
    cpu_thread_limit: int = Field(default=0, ge=0)


# -------------------------------------------------------------------------
# Scheduler Configuration
# -------------------------------------------------------------------------

class PreemptionConfig(BaseModel):
    persist_dir: Optional[str] = None


class SchedulerConfig(BaseModel):
    """Scheduler configuration (section `scheduler`)."""
    aimd_target_tps: float = Field(default=7.0, gt=0.0)
    aimd_min_batch: int = Field(default=1, ge=1)
    aimd_max_batch: int = Field(default=4, ge=1)
    aimd_interval_s: float = Field(default=5.0, gt=0.0)
    aimd_hysteresis: int = Field(default=2, ge=1)
    aimd_high_margin: float = Field(default=0.95, gt=0.0, le=1.0)
    aimd_low_margin: float = Field(default=0.85, gt=0.0, le=1.0)
    max_queue_per_priority: int = Field(default=64, ge=1)
    max_total_queue: int = Field(default=256, ge=1)
    default_timeout_s: float = Field(default=300.0, gt=0.0)
    prune_interval_s: float = Field(default=10.0, gt=0.0)
    fairness_weights: Dict[str, int] = Field(
        default_factory=lambda: {"low": 1, "normal": 2, "high": 4, "urgent": 8}
    )
    preemption: PreemptionConfig = Field(default_factory=PreemptionConfig)

    @model_validator(mode='after')
    def check_bounds(self) -> 'SchedulerConfig':
        if self.aimd_min_batch > self.aimd_max_batch:
            raise ValueError("aimd_min_batch cannot exceed aimd_max_batch")
        if self.aimd_low_margin >= self.aimd_high_margin:
            raise ValueError("aimd_low_margin must be < aimd_high_margin")
        return self


# -------------------------------------------------------------------------
# Kernels Configuration
# -------------------------------------------------------------------------

class KernelsConfig(BaseModel):
    """Custom compute kernels configuration."""
    enabled: bool = True
    auto_select: bool = True
    force_backend: Optional[Literal["cuda", "hip", "metal", "vulkan", "cpu"]] = None
    cpu_optimizer: Dict[str, Any] = Field(default_factory=dict)


# -------------------------------------------------------------------------
# Model Lifecycle Configuration
# -------------------------------------------------------------------------

class ModelLifecycleConfig(BaseModel):
    """Model lifecycle management configuration."""
    max_loaded_models: int = Field(default=8, ge=1)
    idle_timeout_s: float = Field(default=1800.0, ge=0.0)
    vram_target_utilization: float = Field(default=0.90, ge=0.1, le=1.0)
    ram_target_utilization: float = Field(default=0.75, ge=0.1, le=1.0)
    eviction_policy: Literal["lru", "priority"] = "lru"
    auto_evict_on_pressure: bool = True
    check_interval_s: float = Field(default=15.0, ge=1.0)
    autoload_all: bool = False


# -------------------------------------------------------------------------
# Models Configuration (profile definitions)
# -------------------------------------------------------------------------

class ModelProfileConfig(BaseModel):
    """A single model profile entry."""
    id: str
    path: str
    format: Literal["gguf", "bitnet"] = "gguf"
    description: str = ""
    autoload: bool = False
    quantization: str = "auto"
    context_length: int = Field(default=0, ge=0)
    num_gpu_layers: int = Field(default=-1, ge=-1)
    n_threads: int = Field(default=0, ge=0)
    use_mlock: bool = False
    rope_scaling: float = Field(default=1.0, gt=0.0)
    gpu_split: Optional[List[float]] = None
    priority: int = Field(default=0, ge=0)
    suppress_thinking: Optional[bool] = None
    tags: List[str] = Field(default_factory=list)
    device: Literal["cuda", "cpu", "mps"] = "cuda"

    @field_validator("path")
    @classmethod
    def validate_path_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("path cannot be empty")
        return v


class ModelsConfig(BaseModel):
    """Container for model profiles."""
    autoload_all: bool = False
    allowed_dir: str = Field(default="/usr/share/llm-models")
    autoload_timeout_s: float = Field(default=900.0, ge=30.0)
    warmup_enabled: bool = True
    warmup_prompt: str = "Hello"
    warmup_tokens: int = Field(default=10, ge=1)
    warmup_timeout_s: float = Field(default=30.0, ge=5.0)
    profiles: List[ModelProfileConfig] = Field(default_factory=list)

    @field_validator("allowed_dir")
    @classmethod
    def validate_allowed_dir(cls, v: str) -> str:
        if not v:
            raise ValueError("allowed_dir cannot be empty")
        return v

#------------------------------------------------------------------------------
# concurrency pool configuration
#------------------------------------------------------------------------------
class ConcurrencyPoolConfig(BaseModel):
    """Concurrency pool configuration (server-side, forwarded to pool)."""
    model_config = {"extra": "allow"}

    mode: str = Field(default="process_pool",
                      description="process_pool | single_process | micro_batch")
    initial_workers: int = Field(default=1, ge=1)
    max_workers: int = Field(default=0, ge=0, description="0 = MemoryLedger decides")
    heartbeat_s: float = Field(default=0.5, gt=0.0)
    respawn_backoff: List[float] = Field(default_factory=lambda: [1, 2, 4, 8, 30])
    max_respawns_per_5min: int = Field(default=3, ge=1)
    global_queue_max: int = Field(default=256, ge=1)
    ram_refuse_pct: float = Field(default=90.0, ge=50.0, le=99.0)
    target_min_tps_per_worker: float = Field(default=1.0, gt=0.0)
    os_reserve_mb: int = Field(default=1024, ge=0)
    runtime_per_worker_mb: int = Field(default=300, ge=64)
    affinity_prefix_weight: float = Field(default=1.0, ge=0.0)
    affinity_thermal_weight: float = Field(default=0.5, ge=0.0)
    affinity_staleness_weight: float = Field(default=1.0, ge=0.0)
    throttle_start_c: float = Field(default=85.0, ge=60.0)
    micro_batch_enabled: bool = Field(default=False)

