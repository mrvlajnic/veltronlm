"""Unified command-line interface.

    veltron info                      environment, backends, registry, project status
    veltron config                    print the model registry
    veltron param-count               exact parameter accounting for every tier
    veltron devices                   compute backends and measured throughput
    veltron tokens                    train the tokenizers
    veltron data acquire              download the corpus
    veltron data build                build a versioned dataset
    veltron train                     pretraining
    veltron finetune build-data       build the SFT dataset
    veltron finetune train            instruction tuning
    veltron finetune preferences      build preference pairs
    veltron finetune dpo              preference optimisation
    veltron index                     build the RAG index
    veltron generate                  generate text
    veltron chat                      interactive support chat
    veltron ask                       one-shot grounded question
    veltron evaluate                  run benchmarks
    veltron serve                     run the API + chat UI
    veltron checkpoints               audit checkpoints
    veltron report                    render the project status report
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SUBCOMMANDS = {
    "info": ("veltron.info", "Project and environment status"),
    "config": ("veltron.train", "Model registry"),
    "param-count": ("scripts.verify_param_count", None),
    "devices": ("veltron.train", None),
    "tokens": (None, "Train the tokenizers"),
    "data": (None, "Data pipeline"),
    "train": ("veltron.train", "Pretraining"),
    "finetune": ("veltron.finetuning.cli", "SFT and DPO"),
    "index": (None, "Build the RAG index"),
    "generate": (None, "Generate text from a checkpoint"),
    "chat": (None, "Interactive support chat"),
    "ask": (None, "One-shot grounded question"),
    "evaluate": ("veltron.evaluate", "Run benchmarks"),
    "serve": ("veltron.serve", "Run the API and chat UI"),
    "checkpoints": (None, "Audit checkpoints"),
    "report": (None, "Render the status report"),
}


def _run(module: str, args: list[str]) -> int:
    if module.startswith("scripts."):
        path = module.split(".", 1)[1] + ".py"
        if not Path(path).exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 2
        return subprocess.run([sys.executable, path] + args).returncode
    return subprocess.run([sys.executable, "-m", module] + args).returncode


def _info() -> int:
    import torch

    from . import __version__
    from .data.sources import licence_report
    from .model.config import registry_table
    from .utils.device import summary

    print("=" * 84)
    print(f"VeltronLM {__version__}")
    print("=" * 84)

    print("\n-- compute backends --")
    for b in summary()["backends"]:
        print(f"  {b['kind']:10} {b['name']:24} {b['total_memory_gb']:7.2f} GB  "
              f"fp16={b['supports_fp16']} bf16={b['supports_bf16']}")

    print("\n-- python / torch --")
    print(f"  python {sys.version.split()[0]}   torch {torch.__version__}")

    print("\n-- model registry --")
    for r in registry_table():
        flag = "trainable here" if r["trainable_here"] else "inference only"
        print(f"  {r['key']:9} {r['name']:24} {r['params']:>13,}  {r['params_B']:>7.4f}B  {flag}")

    print("\n-- data sources --")
    for row in licence_report():
        print(f"  {row['key']:20} {row['license']:24} commercial={row['commercial_use']}")

    print("\n-- artefacts --")
    for label, path in (
        ("dataset", "datasets/dataset-v1/manifest.json"),
        ("tokenizer (mini)", "models/tok-mini-32k/tokenizer.json"),
        ("tokenizer (4B)", "models/tok-4b-64k/tokenizer.json"),
        ("RAG index", "models/rag_index/chunks.json"),
        ("SFT data", "datasets/sft/sft.jsonl"),
        ("preferences", "datasets/alignment/preferences.jsonl"),
    ):
        p = Path(path)
        extra = ""
        if p.name == "manifest.json" and p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            extra = f"  ({m['totals']['train_tokens']:,} train tokens)"
        if p.name == "chunks.json" and p.exists():
            extra = f"  ({len(json.loads(p.read_text(encoding='utf-8')))} chunks)"
        print(f"  {label:18} {'present' if p.exists() else 'MISSING':8} {p}{extra}")

    runs = sorted(Path("checkpoints").glob("*/summary.json")) if Path("checkpoints").exists() else []
    print("\n-- completed runs --")
    if not runs:
        print("  none yet")
    for r in runs:
        s = json.loads(r.read_text(encoding="utf-8"))
        print(f"  {s.get('run_name', r.parent.name):22} steps={s.get('steps_completed', s.get('steps', '-'))} "
              f"val_loss={s.get('final_val_loss', s.get('final_val_perplexity', '-'))}")
    return 0


def _tokens(args: list[str]) -> int:
    return subprocess.run([sys.executable, "scripts/train_tokenizers.py"] + args).returncode


def _data(args: list[str]) -> int:
    if not args or args[0] in ("-h", "--help"):
        print("usage: veltron data {acquire|build} [options]")
        print("  acquire  download the corpus into datasets/raw/")
        print("  build    build a versioned, tokenized, sharded dataset")
        return 0
    sub, rest = args[0], args[1:]
    if sub == "acquire":
        return subprocess.run([sys.executable, "scripts/acquire_data.py"] + rest).returncode
    if sub == "build":
        return subprocess.run([sys.executable, "scripts/build_dataset.py"] + rest).returncode
    print(f"error: unknown data subcommand {sub!r}", file=sys.stderr)
    return 2


def _index(args: list[str]) -> int:
    code = """
from veltron.rag.ingest import knowledge_base_chunks
from veltron.rag.pipeline import RAGConfig, RAGPipeline
cfg = RAGConfig()
pipe = RAGPipeline(cfg=cfg)
chunks = knowledge_base_chunks(cfg.chunk_chars, cfg.overlap_chars)
r = pipe.build_index(%r, chunks)
print('index:', r.stats())
"""
    out = args[0] if args else "models/rag_index"
    ns: dict = {}
    exec(code % repr(out), {"__name__": "__main__"}, ns)  # noqa: S102
    return 0


def _generate(args: list[str]) -> int:
    from .inference.cli import main as gen_main

    return gen_main(["generate"] + args)


def _chat(args: list[str]) -> int:
    from .inference.cli import main as gen_main

    return gen_main(["chat"] + args)


def _ask(args: list[str]) -> int:
    from .inference.cli import main as gen_main

    return gen_main(["ask"] + args)


def _checkpoints(args: list[str]) -> int:
    return subprocess.run([sys.executable, "scripts/verify_checkpoints.py"] + args).returncode


def _report(args: list[str]) -> int:
    return subprocess.run([sys.executable, "scripts/generate_report.py"] + args).returncode


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("commands:")
        for name, (_mod, desc) in SUBCOMMANDS.items():
            print(f"  {name:14} {desc or ''}")
        return 0
    cmd, args = argv[0], argv[1:]

    handlers = {
        "info": _info,
        "tokens": _tokens,
        "data": _data,
        "index": _index,
        "generate": _generate,
        "chat": _chat,
        "ask": _ask,
        "checkpoints": _checkpoints,
        "report": _report,
    }
    if cmd in handlers:
        return handlers[cmd](args)

    if cmd in ("config", "param-count", "devices"):
        module = SUBCOMMANDS[cmd][0]
        extra = ["--list-configs"] if cmd == "config" else (
            ["--devices"] if cmd == "devices" else [])
        return _run(module, extra + args)
    if cmd == "train":
        return _run("veltron.train", args)
    if cmd == "finetune":
        return _run("veltron.finetuning.cli", args)
    if cmd == "evaluate":
        return _run("veltron.evaluate", args)
    if cmd == "serve":
        return _run("veltron.serve", args)

    print(f"error: unknown command {cmd!r}", file=sys.stderr)
    print("run `veltron --help` for the command list", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
