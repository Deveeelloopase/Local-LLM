#!/usr/bin/env python3
"""
local_llm_client/core/examples.py

Production‑ready example suite covering every integration pattern.

Run any example:
    python -m local_llm_client.core.examples <example_name>

Available examples:
    sync              – synchronous non‑streaming generation
    stream            – synchronous streaming generation
    async             – asynchronous non‑streaming generation
    async_stream      – asynchronous streaming generation
    batch             – batch generation through adapter
    agent             – adapter with adaptation
    security_agent    – production pattern for security analysis
    models_list       – list loaded models
    models_load       – load a model (requires server with model path)
    models_unload     – unload a model
    models_hotswap    – hot‑swap two models
    metrics           – display server metrics
    health            – check server health
    keys_create       – create an API key (requires master key)
    keys_list         – list API keys
    keys_stats        – show statistics for a key
    all               – run a comprehensive integration test

Configuration:
    Set environment variables (LLM_BASE_URL, LLM_MODEL, etc.) or edit the
    ClientConfig defaults below.  The config is automatically built from
    environment variables when available.
"""

import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------------------------
# Client imports (the production‑grade versions you have just hardened)
# ------------------------------------------------------------------------------
from local_llm_client.core import (
    LLMCloudClient,
    AsyncLLMCloudClient,
    ClientConfig,
    GenerationResult,
)
from local_llm_client.core.adapter import LocalLLMCloudAdapter
from local_llm_client.core.exceptions import (
    LLMCloudError,
    ModelNotLoadedError,
    RateLimitError,
    AuthError,
    ConnectionError,
    CircuitOpenError,
)


# ------------------------------------------------------------------------------
# Default configuration – can be fully overridden via environment variables
# ------------------------------------------------------------------------------
def build_config(overrides: Optional[Dict[str, Any]] = None) -> ClientConfig:
    """Construct a ClientConfig from environment variables with optional overrides."""
    cfg = ClientConfig()
    cfg.update_from_env()   # populates from LLM_* env vars
    if overrides:
        cfg.apply_overrides(**overrides)
    return cfg


# ------------------------------------------------------------------------------
# Helper: print a separator
# ------------------------------------------------------------------------------
def header(title: str) -> None:
    print(f"\n{'═' * 60}\n  {title}\n{'═' * 60}")


# ------------------------------------------------------------------------------
# 1. Synchronous generate (non‑streaming)
# ------------------------------------------------------------------------------
def example_sync() -> None:
    """Basic synchronous generation."""
    header("SYNCHRONOUS GENERATE")
    client = LLMCloudClient(build_config())
    try:
        if not client.is_healthy():
            print("[SYNC] Server not healthy – check if it's running.")
            return
        result = client.generate(
            prompt="What is the capital of France?",
            system_prompt="You are a helpful assistant.",
        )
        print(f"[SYNC] Response: {result}")
    except LLMCloudError as e:
        print(f"[SYNC] Error: {e}")
    finally:
        client.close()


# ------------------------------------------------------------------------------
# 2. Synchronous streaming
# ------------------------------------------------------------------------------
def example_stream() -> None:
    """Streaming – tokens appear word‑by‑word as generated."""
    header("SYNCHRONOUS STREAMING")
    client = LLMCloudClient(build_config())
    try:
        if not client.is_healthy():
            print("[STREAM] Server not healthy.")
            return
        print("[STREAM] ", end="", flush=True)
        t0 = time.perf_counter()
        for chunk in client.stream(
            "Write a two‑paragraph story about a robot learning to cook.",
            max_tokens=0,
        ):
            print(chunk, end="", flush=True)
        elapsed = time.perf_counter() - t0
        print(f"\n[STREAM] Done in {elapsed:.1f}s")
    except LLMCloudError as e:
        print(f"\n[STREAM] Error: {e}")
    finally:
        client.close()


# ------------------------------------------------------------------------------
# 3. Asynchronous generate
# ------------------------------------------------------------------------------
async def example_async() -> None:
    """Asynchronous non‑streaming generation."""
    header("ASYNC GENERATE")
    async with AsyncLLMCloudClient(build_config()) as client:
        try:
            if not await client.is_healthy():
                print("[ASYNC] Server not healthy.")
                return
            result = await client.generate(
                prompt="Explain how neural networks work in 3 sentences.",
            )
            print(f"[ASYNC] Response: {result}")
        except LLMCloudError as e:
            print(f"[ASYNC] Error: {e}")


# ------------------------------------------------------------------------------
# 4. Asynchronous streaming
# ------------------------------------------------------------------------------
async def example_async_stream() -> None:
    """Asynchronous streaming generation."""
    header("ASYNC STREAMING")
    async with AsyncLLMCloudClient(build_config()) as client:
        try:
            if not await client.is_healthy():
                print("[ASYNC-STREAM] Server not healthy.")
                return
            print("[ASYNC-STREAM] ", end="", flush=True)
            t0 = time.perf_counter()
            async for chunk in client.stream(
                "List five interesting facts about the moon.",
                max_tokens=200,
            ):
                print(chunk, end="", flush=True)
            elapsed = time.perf_counter() - t0
            print(f"\n[ASYNC-STREAM] Done in {elapsed:.1f}s")
        except LLMCloudError as e:
            print(f"\n[ASYNC-STREAM] Error: {e}")


# ------------------------------------------------------------------------------
# 5. Batch generate (sequential via adapter)
# ------------------------------------------------------------------------------
async def example_batch() -> None:
    """Batch generation – multiple prompts processed sequentially."""
    header("BATCH GENERATION")
    adapter = LocalLLMCloudAdapter(build_config())
    try:
        if not await adapter.initialize():
            print("[BATCH] Failed to initialize adapter.")
            return
        prompts: List[str] = [
            "What is Python?",
            "What is machine learning?",
            "What is a REST API?",
            "What is Docker?",
        ]
        t0 = time.perf_counter()
        results = await adapter.batch_generate(prompts, max_tokens=200)
        elapsed = time.perf_counter() - t0
        for prompt, text in zip(prompts, results):
            print(f"[BATCH] Q: {prompt}")
            print(f"[BATCH] A: {text[:120]}…\n")
        print(f"[BATCH] All {len(prompts)} prompts completed in {elapsed:.1f}s")
    except LLMCloudError as e:
        print(f"[BATCH] Error: {e}")
    finally:
        adapter.shutdown()


# ------------------------------------------------------------------------------
# 6. AI agent adapter with context adaptation
# ------------------------------------------------------------------------------
async def example_agent() -> None:
    """
    Drop‑in adapter for ai_agent.py – demonstrates token budget management
    and prompt compression.
    """
    header("ADAPTER (AGENT PATTERN)")
    adapter = LocalLLMCloudAdapter(build_config())
    try:
        if not await adapter.initialize():
            print("[AGENT] Failed to initialize adapter.")
            return
        print(f"[AGENT] Healthy: {adapter.is_healthy()}")

        # Simple generate
        text = await adapter.generate("Tell me a joke about programming.")
        print(f"[AGENT] {text}")

        # Generate with adaptation
        text, meta = await adapter.generate_with_adaptation(
            "Summarise the history of artificial intelligence in 4 bullet points."
        )
        print(f"[AGENT] {text}")
        print(f"[AGENT] Metadata: tps={meta.get('tps', 0):.1f} "
              f"tokens={meta.get('tokens_generated', 0)}")

        # Generate full result object
        gen_result = await adapter.generate_full(
            "What is the meaning of life?"
        )
        print(f"[AGENT] Full result: tokens={gen_result.tokens_generated} ttft={gen_result.ttft_ms}ms tps={gen_result.tps:.1f}")

        # Metrics
        metrics = adapter.get_metrics()
        print(f"[AGENT] Adapter metrics: {metrics}")
    except LLMCloudError as e:
        print(f"[AGENT] Error: {e}")
    finally:
        adapter.shutdown()


# ------------------------------------------------------------------------------
# 7. Security‑agent production pattern
# ------------------------------------------------------------------------------
async def example_security_agent() -> None:
    """Full production workflow for a security analysis agent."""
    header("SECURITY AGENT")
    adapter = LocalLLMCloudAdapter(build_config())
    try:
        if not await adapter.initialize():
            print("[SECURITY] Server not available – aborting.")
            return

        # 1. Check queue depth
        try:
            qs = adapter._sync_client.queue_status()
            queue_depth = qs.get("queue_depth", 0)
            gate_waiters = qs.get("gate_waiters", 0)
            est_wait = qs.get("estimated_wait_s", 0)
            print(f"[SECURITY] Queue depth: {queue_depth}  Gate waiters: {gate_waiters}  Est. wait: {est_wait:.0f}s")
        except Exception:
            pass

        # 2. Check current model info
        try:
            models = adapter._sync_client.models()
            print(f"[SECURITY] Loaded models: {[m.get('id') for m in models]}")
        except Exception:
            pass

        # 3. Stream a threat‑analysis prompt
        prompt = (
            "You are a cybersecurity analyst. Analyse the following log excerpt "
            "and identify any suspicious patterns, potential threats, or anomalies. "
            "Provide a structured threat assessment.\n\n"
            "LOG:\n"
            "2024-01-15 03:42:17 WARN  Failed login attempt from 192.168.1.105 user=admin\n"
            "2024-01-15 03:42:19 WARN  Failed login attempt from 192.168.1.105 user=root\n"
            "2024-01-15 03:42:21 WARN  Failed login attempt from 192.168.1.105 user=admin\n"
            "2024-01-15 03:42:45 INFO  Successful login from 192.168.1.105 user=backup\n"
            "2024-01-15 03:43:12 INFO  File access: /etc/passwd (user=backup)\n"
            "2024-01-15 03:43:15 INFO  File access: /etc/shadow (user=backup)\n"
        )
        print("[SECURITY] Analysing threat…")
        full_response: list[str] = []
        try:
            loop = asyncio.get_running_loop()
            chunks = await loop.run_in_executor(
                None, list, adapter.stream_sync(prompt, max_tokens=0)
            )
            for chunk in chunks:
                print(chunk, end="", flush=True)
                full_response.append(chunk)
        except ModelNotLoadedError:
            print("\n[SECURITY] Model was unloaded during analysis")
        except RateLimitError as e:
            print(f"\n[SECURITY] Rate limited – retry after {e.retry_after:.0f}s")
        except Exception as e:
            print(f"\n[SECURITY] Streaming error: {e}")
        else:
            total_chars = sum(len(c) for c in full_response)
            print(f"\n[SECURITY] Analysis complete. Total chars: {total_chars}")

    except LLMCloudError as e:
        print(f"[SECURITY] Error: {e}")
    finally:
        adapter.shutdown()


# ------------------------------------------------------------------------------
# 8. Model management (list, load, unload, hot‑swap)
# ------------------------------------------------------------------------------
def example_models_list() -> None:
    """List all models currently loaded on the server."""
    header("MODELS – LIST")
    client = LLMCloudClient(build_config())
    try:
        models = client.models()
        if not models:
            print("No models loaded.")
            return
        for m in models:
            print(f"  {m.get('id','?'):20s} {m.get('model_type','gguf'):8s} "
                  f"GPU layers: {m.get('gpu_layers',0):3d}  VRAM: {m.get('vram_mb',0):5d} MB")
    except LLMCloudError as e:
        print(f"Error: {e}")
    finally:
        client.close()


def example_models_load() -> None:
    """Load a model (interactive prompts for path and ID)."""
    header("MODELS – LOAD")
    model_id = input("Model ID: ").strip()
    model_path = input("Model path (GGUF file or BitNet dir): ").strip()
    if not model_id or not model_path:
        print("Both model ID and path are required.")
        return
    client = LLMCloudClient(build_config())
    try:
        resp = client.load_model(
            model_id=model_id,
            model_path=model_path,
            model_type="auto",
        )
        print(f"Response: {resp}")
    except LLMCloudError as e:
        print(f"Error: {e}")
    finally:
        client.close()


def example_models_unload() -> None:
    """Unload a model by ID."""
    header("MODELS – UNLOAD")
    model_id = input("Model ID to unload: ").strip()
    if not model_id:
        print("Model ID is required.")
        return
    client = LLMCloudClient(build_config())
    try:
        resp = client.unload_model(model_id)
        print(f"Response: {resp}")
    except LLMCloudError as e:
        print(f"Error: {e}")
    finally:
        client.close()


def example_models_hotswap() -> None:
    """Hot‑swap two models (interactive)."""
    header("MODELS – HOT‑SWAP")
    old_id = input("Old model ID: ").strip()
    new_id = input("New model ID: ").strip()
    new_path = input("New model path: ").strip()
    if not old_id or not new_id or not new_path:
        print("All fields are required.")
        return
    client = LLMCloudClient(build_config())
    try:
        resp = client.hot_swap(
            old_id=old_id,
            new_id=new_id,
            new_path=new_path,
        )
        print(f"Response: {resp}")
    except LLMCloudError as e:
        print(f"Error: {e}")
    finally:
        client.close()


# ------------------------------------------------------------------------------
# 9. Server metrics and health
# ------------------------------------------------------------------------------
def example_metrics() -> None:
    """Display server performance metrics."""
    header("SERVER METRICS")
    client = LLMCloudClient(build_config())
    try:
        m = client.metrics()
        engine = m.get("engine", {})
        vram = m.get("vram", {})
        system = m.get("system", {})
        print(f"  TPS: {engine.get('avg_tps', 0):.2f}  TTFT: {engine.get('avg_ttft_ms', 0):.0f} ms")
        print(f"  Requests: {engine.get('total_requests', 0)}  Tokens: {engine.get('total_tokens', 0)}")
        print(f"  Queue depth: {engine.get('queue_depth', 0)}  Circuit: {engine.get('circuit_breaker', {}).get('state', 'closed')}")
        used_mb = vram.get('used_bytes', 0) >> 20
        total_mb = vram.get('total_bytes', 0) >> 20
        util = vram.get('utilization_pct', 0)
        print(f"  VRAM: {used_mb} / {total_mb} MB ({util:.1f}%)")
        print(f"  CPU: {system.get('cpu_pct', 0):.1f}%  RAM: {system.get('ram_pct', 0):.1f}%")
    except LLMCloudError as e:
        print(f"Error: {e}")
    finally:
        client.close()


def example_health() -> None:
    """Check server health and component status."""
    header("SERVER HEALTH")
    client = LLMCloudClient(build_config())
    try:
        h = client.health()
        print(f"Status: {h.get('status', 'unknown')}")
        checks = h.get("checks", {})
        for component, state in checks.items():
            icon = "✓" if state == "ok" else ("⚠" if state == "degraded" else "✗")
            print(f"  {icon} {component}: {state}")
    except LLMCloudError as e:
        print(f"Error: {e}")
    finally:
        client.close()


# ------------------------------------------------------------------------------
# 10. API Key management (requires master key)
# ------------------------------------------------------------------------------
def example_keys_create() -> None:
    """Create a new API key."""
    header("KEYS – CREATE")
    key_id = input("New key ID: ").strip()
    if not key_id:
        print("Key ID is required.")
        return
    rps = input("Rate limit (requests per second, default 10): ").strip() or "10"
    client = LLMCloudClient(build_config({"master_key": os.environ.get("LLM_MASTER_KEY", "")}))
    try:
        resp = client.create_key(
            key_id=key_id,
            rate_limit_rps=float(rps),
            description="CLI example key",
        )
        print(f"Key created! Plain key: {resp.get('plain_key', 'N/A')}")
        print("Store this key – it will not be shown again.")
    except LLMCloudError as e:
        print(f"Error: {e}")
    finally:
        client.close()


def example_keys_list() -> None:
    """List all API keys (active only by default)."""
    header("KEYS – LIST")
    client = LLMCloudClient(build_config({"master_key": os.environ.get("LLM_MASTER_KEY", "")}))
    try:
        keys = client.list_keys(active_only=True)
        if not keys:
            print("No keys found.")
            return
        for k in keys:
            active = "✅" if k.get("is_active") else "❌"
            print(f"  {k['id']:20s} {active} RPS: {k.get('rate_limit_rps', 0):5.1f} "
                  f"Requests: {k.get('total_requests', 0):5d}")
    except LLMCloudError as e:
        print(f"Error: {e}")
    finally:
        client.close()


def example_keys_stats() -> None:
    """Show detailed statistics for a specific API key."""
    header("KEYS – STATS")
    key_id = input("Key ID: ").strip()
    if not key_id:
        print("Key ID is required.")
        return
    client = LLMCloudClient(build_config({"master_key": os.environ.get("LLM_MASTER_KEY", "")}))
    try:
        stats = client.key_stats(key_id)
        print(f"Key: {stats.get('key_id', key_id)}")
        print(f"  Created: {stats.get('created_at', '?')}")
        print(f"  Expires: {stats.get('expires_at', 'never')}")
        print(f"  Rate limit: {stats.get('rate_limit_rps', 0)} RPS / {stats.get('rate_limit_capacity', 0)} burst")
        print(f"  Active: {stats.get('is_active', False)}")
        print(f"  Total requests: {stats.get('total_requests', 0)}")
        print(f"  Total tokens: {stats.get('total_tokens', 0)}")
        print(f"  Last used: {stats.get('last_used', 'never')}")
        print(f"  Last IP: {stats.get('last_ip', 'unknown')}")
    except LLMCloudError as e:
        print(f"Error: {e}")
    finally:
        client.close()


# ------------------------------------------------------------------------------
# 11. Comprehensive integration test
# ------------------------------------------------------------------------------
async def example_all() -> None:
    """Run a comprehensive set of operations to validate the whole stack."""
    header("COMPREHENSIVE INTEGRATION TEST")
    cfg = build_config()
    print(f"Server: {cfg.base_url}")
    print(f"Default model: {cfg.model}")

    # 1. Health
    client = LLMCloudClient(cfg)
    try:
        h = client.health()
        print(f"  Health: {h.get('status', 'unknown')}")
    except Exception as e:
        print(f"  Health check failed: {e}")
        return
    finally:
        client.close()

    # 2. List models
    client = LLMCloudClient(cfg)
    try:
        models = client.models()
        print(f"  Loaded models: {len(models)}")
        for m in models:
            print(f"    - {m.get('id','?')} ({m.get('model_type','gguf')})")
    except Exception as e:
        print(f"  Model list failed: {e}")
    finally:
        client.close()

    # 3. Queue status
    client = LLMCloudClient(cfg)
    try:
        qs = client.queue_status()
        print(f"  Queue depth: {qs.get('queue_depth', 0)}")
    except Exception as e:
        print(f"  Queue status failed: {e}")
    finally:
        client.close()

    # 4. Metrics
    client = LLMCloudClient(cfg)
    try:
        m = client.metrics()
        print(f"  Avg TPS: {m.get('engine', {}).get('avg_tps', 0):.2f}")
    except Exception as e:
        print(f"  Metrics failed: {e}")
    finally:
        client.close()

    # 5. Simple generation (synchronous)
    client = LLMCloudClient(cfg)
    try:
        result = client.generate("2+2=", max_tokens=5)
        print(f"  Generate test: {result.strip()}")
    except Exception as e:
        print(f"  Generate test failed: {e}")
    finally:
        client.close()

    # 6. Streaming generation
    client = LLMCloudClient(cfg)
    try:
        print("  Stream test: ", end="", flush=True)
        for chunk in client.stream("Hello", max_tokens=10):
            print(chunk, end="", flush=True)
        print()
    except Exception as e:
        print(f"\n  Stream test failed: {e}")
    finally:
        client.close()

    # 7. Adapter test
    adapter = LocalLLMCloudAdapter(cfg)
    try:
        init_ok = await adapter.initialize()
        print(f"  Adapter init: {init_ok}")
        if init_ok:
            text = await adapter.generate("Say OK")
            print(f"  Adapter generate: {text.strip()}")
    except Exception as e:
        print(f"  Adapter test failed: {e}")
    finally:
        adapter.shutdown()

    print("\n✅ Integration test complete.")


# ------------------------------------------------------------------------------
# Command‑line dispatcher
# ------------------------------------------------------------------------------
_EXAMPLES = {
    "sync":               ("Synchronous non‑streaming generation", example_sync),
    "stream":             ("Synchronous streaming generation", example_stream),
    "async":              ("Asynchronous non‑streaming generation", lambda: asyncio.run(example_async())),
    "async_stream":       ("Asynchronous streaming generation", lambda: asyncio.run(example_async_stream())),
    "batch":              ("Batch generation via adapter", lambda: asyncio.run(example_batch())),
    "agent":              ("Adapter with context adaptation", lambda: asyncio.run(example_agent())),
    "security_agent":     ("Production security‑agent pattern", lambda: asyncio.run(example_security_agent())),
    "models_list":        ("List loaded models", example_models_list),
    "models_load":        ("Load a model (interactive)", example_models_load),
    "models_unload":      ("Unload a model", example_models_unload),
    "models_hotswap":     ("Hot‑swap two models", example_models_hotswap),
    "metrics":            ("Display server metrics", example_metrics),
    "health":             ("Check server health", example_health),
    "keys_create":        ("Create an API key (requires master key)", example_keys_create),
    "keys_list":          ("List API keys", example_keys_list),
    "keys_stats":         ("Show key statistics", example_keys_stats),
    "all":                ("Run comprehensive integration test", lambda: asyncio.run(example_all())),
}


def main() -> None:
    if len(sys.argv) < 2:
        print("Available examples:")
        for name, (desc, _) in sorted(_EXAMPLES.items()):
            print(f"  {name:20s} – {desc}")
        print("\nUsage: python -m local_llm_client.core.examples <example>")
        sys.exit(1)

    choice = sys.argv[1]
    if choice not in _EXAMPLES:
        print(f"Unknown example: {choice!r}")
        print(f"Available: {', '.join(sorted(_EXAMPLES))}")
        sys.exit(1)

    _, runner = _EXAMPLES[choice]
    try:
        runner()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Unhandled error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
