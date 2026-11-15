"""Batch generation on the eval sets with the base model or a LoRA adapter (greedy decoding).

    python -m src.predict --tag base --files eval/val_trained.jsonl eval/val_heldout.jsonl
    python -m src.predict --tag ft --adapter runs/qwen3-4b-lora-r16-v2/best --files eval/val_trained.jsonl

Writes results/eval/<tag>/<file name>.jsonl, one row per chunk: {id, text, n_prompt, n_gen, truncated, seconds}.
The script is resumable: ids already in the output file are skipped, so a crash or Ctrl-C loses at most one batch.
Prompts are rendered exactly as training rendered them (apply_chat_template + generation prompt), via mlx-lm's
own tokenizer wrapper.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/cuad_v1"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def prompt_tokens(tokenizer, messages: list[dict]) -> list[int]:
    # same call as mlx_lm.tuner.datasets.ChatDataset.process uses for the prompt part of a training example
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="output folder name, e.g. base or ft")
    ap.add_argument("--adapter", type=Path, help="LoRA adapter dir (omit for the base model)")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--files", nargs="+", required=True, help="eval files relative to data/processed/cuad_v1")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, help="only the first N rows of each file (for smoke tests)")
    ap.add_argument("--out-root", type=Path, default=ROOT / "results/eval")
    a = ap.parse_args()

    out_dir = a.out_root / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load(a.model, adapter_path=str(a.adapter) if a.adapter else None)
    (out_dir / "predict_info.json").write_text(json.dumps(
        {"model": a.model, "adapter": str(a.adapter) if a.adapter else None, "decoding": "greedy",
         "max_tokens": a.max_tokens, "batch_size": a.batch_size}, indent=1))

    for f in a.files:
        rows = [json.loads(l) for l in open(DATA / f, encoding="utf-8")][: a.limit]
        out = out_dir / Path(f).name
        done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
        todo = [(r["id"], prompt_tokens(tokenizer, r["messages"])) for r in rows if r["id"] not in done]
        todo.sort(key=lambda t: len(t[1]))  # similar lengths per batch, so less padding
        print(f"{f}: {len(rows)} rows, {len(done)} done, {len(todo)} to go", flush=True)
        t_file = time.time()
        for b in range(0, len(todo), a.batch_size):
            batch = todo[b: b + a.batch_size]
            t0 = time.time()
            resp = batch_generate(model, tokenizer, [p for _, p in batch], max_tokens=a.max_tokens,
                                  return_token_ids=True)
            dt = time.time() - t0
            with open(out, "a", encoding="utf-8") as fh:
                for (rid, p), text, ids in zip(batch, resp.texts, resp.token_ids):
                    fh.write(json.dumps({"id": rid, "text": text, "n_prompt": len(p), "n_gen": len(ids),
                                         "truncated": len(ids) >= a.max_tokens,
                                         "seconds": round(dt / len(batch), 2)}, ensure_ascii=False) + "\n")
            n = b + len(batch)
            eta = (time.time() - t_file) / n * (len(todo) - n)
            print(f"  {n}/{len(todo)}  batch {dt:.1f}s  peak {mx.get_peak_memory() / 2**30:.1f}GB  "
                  f"eta {eta / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
