"""Inference CLI: generate, interactive chat, and one-shot grounded questions.

    python -m veltron.inference.cli generate --checkpoint <dir> --prompt "hello"
    python -m veltron.inference.cli chat --checkpoint <dir> --rag
    python -m veltron.inference.cli ask --checkpoint <dir> "How long is the warranty?"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ..utils.logging_utils import setup_logging

DEFAULT_INDEX = "models/rag_index"


def _load(args: argparse.Namespace):
    from ..inference.engine import load_engine
    from ..inference.generator import GenerationConfig

    engine = load_engine(checkpoint=args.checkpoint, tokenizer_dir=args.tokenizer or None,
                         device=args.device, dtype=args.dtype,
                         quantize_bits=args.quantize)
    cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        min_p=args.min_p,
        greedy=args.greedy,
        seed=args.seed,
    )
    return engine, cfg


def _maybe_rag(engine, args):
    from ..rag.pipeline import RAGConfig, RAGPipeline

    pipe = RAGPipeline(generator=engine.generator, cfg=RAGConfig())
    index = Path(args.rag_index)
    if (index / "chunks.json").exists():
        pipe.load_index(index)
    else:
        pipe.build_index(index)
    return pipe


def _cmd_generate(args: argparse.Namespace) -> int:
    engine, cfg = _load(args)
    print(f"model: {engine.info.model_name} "
          f"({engine.info.parameters_billions:.4f}B) on {engine.info.device} "
          f"[{engine.info.dtype}]")
    if not args.prompt and not args.interactive:
        print("error: --prompt is required unless --interactive is set", file=sys.stderr)
        return 2

    if args.prompt:
        res = engine.generate(args.prompt, cfg)
        print("\n" + "=" * 80)
        print(res.text)
        print("=" * 80)
        print(f"{res.prompt_tokens} prompt + {res.completion_tokens} generated tokens  "
              f"{res.tokens_per_second:.2f} tok/s  TTFT {res.time_to_first_token*1000:.0f} ms  "
              f"finish={res.finish_reason}")

    if args.interactive:
        return _chat_loop(engine, cfg, None, stream=args.stream)
    return 0


def _chat_loop(engine, cfg, rag, stream: bool = True) -> int:
    print("\nInteractive mode. 'exit' to quit, 'reset' to clear history, 'stats' for tokens.")
    history: list[dict[str, str]] = []
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("exit", "quit"):
            return 0
        if line == "reset":
            history = []
            engine.generator.reset_cache()
            print("  history cleared")
            continue
        if line == "stats":
            print(f"  messages={len(history)}")
            continue

        t0 = time.perf_counter()
        if rag is not None:
            res = rag.answer(line, include_debug=args_debug(0))
            text = res.answer
            if res.citations:
                print(f"  [{len(res.citations)} sources, "
                      f"escalated={res.escalated}, refused={res.refused}]")
            for c in res.citations:
                print(f"    [{c['index']}] {c['title']} > {c['section_path']}")
        elif stream:
            buf = []
            for delta in engine.stream(line, cfg):
                sys.stdout.write(delta)
                sys.stdout.flush()
                buf.append(delta)
            print()
            text = "".join(buf)
        else:
            text = engine.generate(line, cfg).text
        history.append({"role": "user", "content": line})
        history.append({"role": "assistant", "content": text})
        print(f"  ({time.perf_counter() - t0:.2f}s)")


def args_debug(_i: int = 0) -> bool:
    return False


def _cmd_chat(args: argparse.Namespace) -> int:
    engine, cfg = _load(args)
    rag = _maybe_rag(engine, args) if args.rag else None
    print(f"model: {engine.info.model_name} on {engine.info.device}")
    print(f"retrieval: {'ON (grounded)' if rag else 'OFF (raw generation)'}")
    print("This model is UNTRAINED unless a checkpoint was loaded; output will be incoherent.")
    return _chat_loop(engine, cfg, rag, stream=not rag)


def _cmd_ask(args: argparse.Namespace) -> int:
    engine, cfg = _load(args)
    rag = _maybe_rag(engine, args)
    question = " ".join(args.question)
    res = rag.answer(question, include_debug=args.verbose)
    if args.json:
        print(res.to_json())
        return 0
    print(f"\nQ: {question}\n")
    print(f"A: {res.answer}\n")
    if res.citations:
        print("Sources:")
        for c in res.citations:
            print(f"  [{c['index']}] {c['title']} > {c['section_path']}  "
                  f"(score {c['score']:.3f})")
    flags = []
    if res.escalated:
        flags.append(f"escalated ({res.escalation.get('reason')})")
    if res.refused:
        flags.append("refused")
    if res.synthetic_data_notice:
        flags.append("synthetic knowledge base")
    if flags:
        print("\nFlags: " + ", ".join(flags))
    print(f"\ncategory: {res.classification.get('category')}  "
          f"confidence: {res.classification.get('confidence')}  "
          f"latency: {res.latency_seconds*1000:.0f} ms")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m veltron.inference.cli")
    ap.add_argument("--checkpoint", default="checkpoints/micro-pretrain")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--quantize", type=int, choices=(4, 8), default=None)
    ap.add_argument("--rag-index", default=DEFAULT_INDEX)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=None)

    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate", help="generate from a prompt")
    p.add_argument("--prompt", default="")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--stream", action="store_true")
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("chat", help="interactive chat")
    p.add_argument("--rag", action="store_true", help="ground answers in the knowledge base")
    p.set_defaults(func=_cmd_chat)

    p = sub.add_parser("ask", help="one-shot grounded question")
    p.add_argument("question", nargs="+")
    p.add_argument("--json", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=_cmd_ask)

    args = ap.parse_args(argv)
    setup_logging(level="WARNING")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
