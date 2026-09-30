"""Post-training CLI: build SFT data, run instruction tuning, build preferences, run DPO.

    python -m veltron.finetune.cli build-data
    python -m veltron.finetune.cli train --base checkpoints/micro-pretrain/step-.../
    python -m veltron.finetune.cli build-preferences --checkpoint <dir>
    python -m veltron.finetune.cli dpo --init <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from veltron.utils.logging_utils import setup_logging

SFT_DATA = "datasets/sft/sft.jsonl"
PREF_DATA = "datasets/alignment/preferences.jsonl"


def _load_documents(limit: int = 900) -> list[dict]:
    docs: list[dict] = []
    for path in sorted(Path("datasets/raw").glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if len(docs) >= limit:
                    return docs
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return docs


def cmd_build_data(args) -> int:
    from veltron.finetuning.data import build_sft_dataset, save_sft_dataset
    from veltron.rag.ingest import knowledge_base_chunks

    chunks = knowledge_base_chunks(max_chars=1100, overlap_chars=150)
    docs = _load_documents()
    print(f"knowledge base: {len(chunks)} chunks from {len({c.doc_id for c in chunks})} docs")
    print(f"corpus documents for general instructions: {len(docs)}")

    examples = build_sft_dataset(chunks=chunks, documents=docs, general_n=args.general,
                                 per_chunk=args.per_chunk, seed=args.seed)
    path = save_sft_dataset(examples, args.out)
    meta = json.loads(Path(args.out).with_suffix(".meta.json").read_text(encoding="utf-8"))
    print("\n" + json.dumps(meta, indent=2))

    print("\nsample examples:")
    for ex in examples[:2]:
        print(f"\n  [{ex.category}] {ex.messages[1]['content'][:120]!r}")
        print(f"    -> {ex.messages[-1]['content'][:220]!r}")
    for ex in [e for e in examples if e.expects_refusal][:1]:
        print(f"\n  [refusal] {ex.messages[1]['content'][:120]!r}")
        print(f"    -> {ex.messages[-1]['content'][:220]!r}")
    print(f"\nwrote {path}")
    return 0


def cmd_train(args) -> int:
    from veltron.finetuning.trainer import SFTConfig, SFTTrainer

    cfg = SFTConfig(
        run_name=args.run_name,
        base_checkpoint=args.base,
        model_config=args.model,
        tokenizer_dir=args.tokenizer,
        data_path=args.data,
        output_dir=args.out,
        max_seq_len=args.max_seq_len,
        micro_batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_steps=args.max_steps,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(cfg.as_dict(), indent=2, default=str))
    summary = SFTTrainer(cfg).train()
    print("\n" + json.dumps(summary, indent=2, default=str))
    return 0


def _best_checkpoint(run_dir: str | Path) -> Path | None:
    from veltron.training.checkpoint import CheckpointManager

    p = Path(run_dir)
    if p.is_dir() and (p / "metadata.json").exists():
        return p
    if p.is_dir():
        best = CheckpointManager(p).best_valid()
        if best:
            return best
    return None


def cmd_build_preferences(args) -> int:
    from veltron.alignment.data import (
        build_dataset_pairs,
        build_preference_pairs,
        save_preference_pairs,
    )
    from veltron.finetuning.data import load_sft_dataset
    from veltron.rag.ingest import knowledge_base_chunks

    chunks = knowledge_base_chunks(max_chars=1100, overlap_chars=150)
    generator = None
    if args.checkpoint:
        from veltron.inference.engine import load_engine

        ckpt = _best_checkpoint(args.checkpoint)
        if ckpt is None:
            print(f"error: no checkpoint found under {args.checkpoint}", file=sys.stderr)
            return 2
        print(f"loading generator from {ckpt}")
        generator = load_engine(checkpoint=str(ckpt), tokenizer_dir=args.tokenizer,
                                device=args.device)

    pairs = build_preference_pairs(chunks, generator=generator, per_chunk=args.per_chunk,
                                  seed=args.seed)
    if Path(args.sft_data).exists():
        pairs += build_dataset_pairs(load_sft_dataset(args.sft_data), generator=generator,
                                    seed=args.seed)
    if not pairs:
        print("error: no preference pairs generated", file=sys.stderr)
        return 1
    path = save_preference_pairs(pairs, args.out)
    meta = json.loads(Path(args.out).with_suffix(".meta.json").read_text(encoding="utf-8"))
    print("\n" + json.dumps(meta, indent=2))
    for pr in pairs[:3]:
        print(f"\n  failure_mode={pr.failure_mode}")
        print(f"    Q:       {pr.prompt[:110]!r}")
        print(f"    chosen:  {pr.chosen[:130]!r}")
        print(f"    rejected:{pr.rejected[:130]!r}")
    print(f"\nwrote {path}")
    return 0


def cmd_dpo(args) -> int:
    from veltron.alignment.trainer import DPOConfig, DPOTrainer

    init = _best_checkpoint(args.init)
    if init is None:
        print(f"error: no checkpoint under {args.init}", file=sys.stderr)
        return 2
    cfg = DPOConfig(
        run_name=args.run_name,
        init_checkpoint=str(init),
        model_config=args.model,
        tokenizer_dir=args.tokenizer,
        data_path=args.data,
        output_dir=args.out,
        max_steps=args.max_steps,
        lr=args.lr,
        beta=args.beta,
        micro_batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(cfg.as_dict(), indent=2, default=str))
    summary = DPOTrainer(cfg).train()
    print("\n" + json.dumps(summary, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m veltron.finetune.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-data", help="generate the SFT dataset")
    p.add_argument("--out", default=SFT_DATA)
    p.add_argument("--general", type=int, default=500)
    p.add_argument("--per-chunk", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.set_defaults(func=cmd_build_data)

    p = sub.add_parser("train", help="run instruction tuning")
    p.add_argument("--base", default="", help="base checkpoint directory")
    p.add_argument("--data", default=SFT_DATA)
    p.add_argument("--run-name", default="sft-support")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--model", default="micro")
    p.add_argument("--tokenizer", default="models/tok-mini-32k")
    p.add_argument("--max-seq-len", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("build-preferences", help="generate preference pairs")
    p.add_argument("--checkpoint", default="", help="checkpoint to sample chosen answers from")
    p.add_argument("--sft-data", default=SFT_DATA)
    p.add_argument("--out", default=PREF_DATA)
    p.add_argument("--per-chunk", type=int, default=3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tokenizer", default="models/tok-mini-32k")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_build_preferences)

    p = sub.add_parser("dpo", help="run preference optimisation")
    p.add_argument("--init", required=True, help="SFT checkpoint to align")
    p.add_argument("--data", default=PREF_DATA)
    p.add_argument("--run-name", default="dpo-support")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--model", default="micro")
    p.add_argument("--tokenizer", default="models/tok-mini-32k")
    p.add_argument("--max-seq-len", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_dpo)

    args = ap.parse_args(argv)
    setup_logging(level="INFO")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
