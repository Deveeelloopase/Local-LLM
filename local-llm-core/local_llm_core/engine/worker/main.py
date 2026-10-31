"""
Worker child process. Owns one Llama(), wraps it in a minimal InferenceEngine,
serves requests over TCP loopback to the parent.

Invoked by the parent as:
    python -m local_llm_core.engine.worker.main

All configuration arrives via environment variables set by the parent:
    LLM_WORKER_PORT        127.0.0.1:<port>   — parent is listening
    LLM_WORKER_SECRET      hex secret, one per worker
    LLM_WORKER_CONFIG      base64(json) of {
                              "model_id": str,
                              "model_path": str,
                              "load_kwargs": {...},
                              "engine_config": {...},
                              "runtime": {...}
                           }

Reuses InferenceEngine so all guard logic (loop guard, solicitation guard,
chunked prefill, EWMA prefill tracker, stall detection) is identical to the
single-process path.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import signal
import sys
import time
from typing import Any, Dict

from local_llm_core.accel.protocol import read_frame, write_frame

log = logging.getLogger(__name__)


async def _load_engine(cfg: Dict[str, Any]) -> Any:
    from local_llm_core.engine.gguf.model_loader import ModelConfig, ModelLoader
    from local_llm_core.engine.inference.inference_engine import (
        InferenceEngine, InferenceRequest,
    )
    from local_llm_core.memory.vram_manager import VRAMManager
    from local_llm_core.memory.ram_optimizer import RAMOptimizer
    from local_llm_core.memory.swap_handler import SwapHandler
    from local_llm_core.memory.hybrid_manager import HybridMemoryManager
    from local_llm_core.hal.hardware_detector import HardwareDetector
    import psutil

    det = HardwareDetector(config=cfg.get("hardware_config") or {})
    sys_info = det.detect_all()

    mem_cfg = cfg.get("memory_config") or {}
    total_vram = max(
        sum(g.vram_total_mb for g in sys_info.gpus if getattr(g, "supported", True)) * 1024 * 1024,
        1,
    )
    ram_target = float(mem_cfg.get("ram_utilization_target", 0.75))
    vram_target = float(mem_cfg.get("vram_utilization_target", 0.95))

    vram_mgr = VRAMManager(
        total_vram_bytes=total_vram,
        config={"memory": {"vram_target_utilization": vram_target}},
    )
    ram_mgr = RAMOptimizer(
        max_ram_mb=int(sys_info.ram_total_mb * ram_target),
        compression=bool(mem_cfg.get("compression_enabled", True)),
    )
    swap_mgr = SwapHandler(config={"memory": {
        "ssd_swap_path": mem_cfg.get("ssd_swap_path", "/var/lib/local-llm/swap"),
        "ssd_swap_max_gb": mem_cfg.get("ssd_swap_max_gb", 32),
    }})
    hybrid = HybridMemoryManager(
        vram_bytes=total_vram,
        ram_bytes=sys_info.ram_total_mb * 1024 * 1024,
        config={"memory": {}},
    )

    loader = ModelLoader(
        vram_mgr, ram_mgr, swap_mgr, sys_info.gpus,
        cfg, hybrid=hybrid, system_info=sys_info,
    )

    model_id = cfg["model_id"]
    model_path = cfg["model_path"]
    mc = ModelConfig(
        model_path=model_path,
        model_id=model_id,
        **({k: v for k, v in (cfg.get("load_kwargs") or {}).items()
            if k in {
                "quantization", "context_length", "num_gpu_layers",
                "n_threads", "use_mmap", "use_mlock", "rope_scaling", "gpu_split",
            }}),
    )
    await loader.load(mc)

    engine_cfg = cfg.get("engine_config") or {}
    engine = InferenceEngine(
        loader,
        config=engine_cfg,
        max_batch_size=1,              # single-slot worker
        target_tps=float(engine_cfg.get("target_tps", 3.0)),
        thermal_state=None,
        kernels_enabled=False,
    )
    engine.start_background_tasks()
    return engine


def _deserialize_request(payload: dict) -> Any:
    from local_llm_core.engine.inference.inference_engine import InferenceRequest
    return InferenceRequest(
        request_id=payload["request_id"],
        model_id=payload["model_id"],
        prompt=payload["prompt"],
        max_tokens=int(payload.get("max_tokens", 0) or 0),
        temperature=float(payload.get("temperature", 0.7)),
        top_p=float(payload.get("top_p", 0.95)),
        top_k=int(payload.get("top_k", 40)),
        stop=list(payload.get("stop") or []),
        stream=True,
        timeout_s=float(payload.get("timeout_s", 0.0) or 0.0),
        # parent decides via payload. Default True means think
        # tags are stripped before tokens leave the worker — matches the
        # single-process engine's behaviour exactly.
        strip_think=bool(payload.get("strip_think", True)),
        root_id=payload.get("root_id") or "",
        grammar=payload.get("grammar"),
        response_format=payload.get("response_format"),
    )


async def _serve_requests(
    engine: Any,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Main loop. Serial: one request in flight, no local queue."""
    current_task: asyncio.Task | None = None
    current_rid: str | None = None

    async def _run_one(req_payload: dict) -> None:
        nonlocal current_rid
        rid = req_payload["request_id"]
        current_rid = rid
        ir = _deserialize_request(req_payload)
        try:
            async for token in engine.generate_stream(ir):
                if token:
                    await write_frame(
                        writer,
                        {"t": "chunk", "rid": rid},
                        token.encode("utf-8", "replace"),
                    )
            await write_frame(writer, {"t": "done", "rid": rid}, b"")
        except asyncio.CancelledError:
            await write_frame(writer, {"t": "cancelled", "rid": rid}, b"")
            raise
        except Exception as exc:
            log.exception("worker generate error for %s", rid)
            await write_frame(
                writer, {"t": "error", "rid": rid},
                str(exc)[:500].encode("utf-8", "replace"),
            )
        finally:
            current_rid = None
            # Report ready with the resident prefix so the parent can score
            # subsequent requests for prefix affinity.
            prefix_ids = _resident_prefix_ids(engine, ir.model_id)
            await write_frame(writer, {
                "t": "ready",
                "resident_prefix_len": len(prefix_ids),
                "resident_prefix_head": prefix_ids[:4096],  # 4096 tokens ≈ 32 KB
            }, b"")

    while True:
        try:
            header, body = await read_frame(reader)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            log.info("parent closed connection")
            return
        except Exception:
            log.exception("bad frame")
            return
        t = header.get("t")
        if t == "submit":
            if current_task is not None and not current_task.done():
                await write_frame(writer, {
                    "t": "error", "rid": header.get("rid", "?"),
                }, b"worker busy")
                continue
            payload = json.loads(body.decode("utf-8"))
            current_task = asyncio.create_task(_run_one(payload))
        elif t == "cancel":
            rid = header.get("rid")
            if current_rid == rid and current_task is not None:
                current_task.cancel()
        elif t == "shutdown":
            if current_task is not None and not current_task.done():
                current_task.cancel()
            return
        elif t == "ping":
            await write_frame(writer, {"t": "pong"}, b"")


def _resident_prefix_ids(engine: Any, model_id: str) -> list:
    """Best-effort: read the live KV prefix the worker holds after the last call."""
    try:
        model = engine._loaded_model_obj(model_id)
        llm = getattr(model, "llm", model)
        ids = getattr(llm, "_input_ids", None)
        if ids is None:
            return []
        return [int(x) for x in ids[:4096]]
    except Exception:
        return []


async def _amain() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [worker %(process)d] %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    port = int(os.environ["LLM_WORKER_PORT"])
    secret = os.environ["LLM_WORKER_SECRET"]
    cfg = json.loads(base64.b64decode(os.environ["LLM_WORKER_CONFIG"]).decode("utf-8"))

    log.info("worker starting: model=%s port=%d", cfg.get("model_id"), port)

    engine = await _load_engine(cfg)
    log.info("worker: model loaded, connecting to parent")

    reader, writer = await asyncio.open_connection("127.0.0.1", port)

    # Handshake: send secret, expect OK from parent
    await write_frame(writer, {"t": "hello", "secret": secret, "pid": os.getpid()}, b"")
    hdr, _ = await read_frame(reader)
    if hdr.get("t") != "hello_ok":
        log.error("handshake failed: %r", hdr)
        return 2

    log.info("worker: handshake ok, entering serve loop")
    try:
        await _serve_requests(engine, reader, writer)
    finally:
        try:
            await engine.shutdown()
        except Exception:
            pass
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
    return 0


def main() -> None:
    try:
        rc = asyncio.run(_amain())
    except KeyboardInterrupt:
        rc = 0
    sys.exit(rc)


if __name__ == "__main__":
    main()
