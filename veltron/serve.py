"""API server entry point.

    VELTRON_CHECKPOINT=checkpoints/micro-pretrain python -m veltron.serve

Environment:
    VELTRON_CHECKPOINT   checkpoint directory (required for generation endpoints)
    VELTRON_RAG_INDEX    retrieval index directory (default models/rag_index)
    VELTRON_DEVICE       auto | cuda | directml | cpu
    VELTRON_DTYPE        auto | fp32 | fp16 | bf16
    VELTRON_HOST         default 127.0.0.1 (loopback: the API has no auth)
    VELTRON_PORT         default 8000
    VELTRON_RATE_LIMIT   requests per window per client IP (default 60)
"""

from __future__ import annotations

import argparse
import os
import sys

from .utils.logging_utils import setup_logging


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m veltron.serve")
    ap.add_argument("--checkpoint", default=os.environ.get("VELTRON_CHECKPOINT", ""))
    ap.add_argument("--tokenizer", default=os.environ.get("VELTRON_TOKENIZER", ""))
    ap.add_argument("--rag-index", default=os.environ.get("VELTRON_RAG_INDEX", "models/rag_index"))
    ap.add_argument("--device", default=os.environ.get("VELTRON_DEVICE", "auto"))
    ap.add_argument("--dtype", default=os.environ.get("VELTRON_DTYPE", "auto"))
    ap.add_argument("--host", default=os.environ.get("VELTRON_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("VELTRON_PORT", "8000")))
    ap.add_argument("--log-level", default=os.environ.get("VELTRON_LOG_LEVEL", "INFO"))
    ap.add_argument("--no-chat", action="store_true", help="API only, no /chat page")
    args = ap.parse_args(argv)

    setup_logging(level=args.log_level)

    os.environ["VELTRON_CHECKPOINT"] = args.checkpoint
    os.environ["VELTRON_RAG_INDEX"] = args.rag_index
    os.environ["VELTRON_DEVICE"] = args.device
    os.environ["VELTRON_DTYPE"] = args.dtype
    if args.tokenizer:
        os.environ["VELTRON_TOKENIZER"] = args.tokenizer

    from .serving.app import app
    if not args.no_chat:
        from .chatbot.server import mount_chatbot

        mount_chatbot(app)

    if args.host not in ("127.0.0.1", "localhost") and not os.environ.get("VELTRON_AUTH_REQUIRED"):
        print(
            f"WARNING: binding to {args.host} exposes an unauthenticated inference API.\n"
            "         VeltronLM ships no authentication; put it behind a reverse proxy "
            "with TLS and auth, or keep it on loopback.",
            file=sys.stderr,
        )

    import uvicorn

    print(f"VeltronLM serving on http://{args.host}:{args.port}")
    print(f"  chat UI   http://{args.host}:{args.port}/chat")
    print(f"  API docs  http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


if __name__ == "__main__":
    sys.exit(main())
