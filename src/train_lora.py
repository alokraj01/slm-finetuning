"""LoRA fine-tuning of a chat model on the CUAD extraction dataset with MLX-LM.

    python -m src.train_lora --config configs/qwen3-4b-lora-r16.yaml [--override iters=20 run_name=smoke]

A thin wrapper around `mlx_lm` (pinned 0.32.2) that adds what the stock CLI lacks:
  * a fail-fast check that no example exceeds `max_seq_length`. mlx-lm would otherwise silently truncate, cutting off
    the end of the excerpt or, worse, the answer;
  * a fixed validation subset evaluated in full, so validation loss is comparable across evaluations (mlx-lm samples
    `val_batches` randomly each time);
  * metrics.jsonl (train/val loss, LR, throughput, memory) via a TrainingCallback. Note that
    `mlx_lm.lora.run()` discards custom callbacks, so we call `train_model` directly;
  * run_info.json (versions, git commit, data hash, hardware, timings) and a `best/` adapter chosen by validation loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import mlx.core as mx
import numpy as np
import yaml
from mlx_lm import __version__ as MLX_LM_VERSION
from mlx_lm.lora import CONFIG_DEFAULTS, train_model
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.utils import load
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]


class JsonlMetrics(TrainingCallback):
    """Writes metrics.jsonl and keeps best/adapters.safetensors = the exact weights with the lowest val loss.

    mlx-lm runs validation at the top of iteration N (weights after N-1 steps, reported as iteration N-1) but saves
    its own checkpoints after step N. Snapshotting here avoids that off-by-one between the score and the weights."""

    def __init__(self, path: Path, model, best_dir: Path):
        self.path, self.model, self.best_dir = path, model, best_dir
        self.t0 = time.time()
        self.best = None  # (val_loss, iteration)

    def _write(self, kind: str, info: dict):
        row = {"kind": kind, "wall_s": round(time.time() - self.t0, 1),
               **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in info.items()}}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if kind == "val":
            print(f"[val] it={info['iteration']} val_loss={info['val_loss']:.4f}", flush=True)
        else:
            print(f"[train] it={info['iteration']} loss={info['train_loss']:.4f} lr={info['learning_rate']:.2e} "
                  f"tok/s={info['tokens_per_second']:.0f} peak_mem={info['peak_memory']:.1f}GB", flush=True)

    def on_train_loss_report(self, info: dict):
        self._write("train", info)

    def on_val_loss_report(self, info: dict):
        self._write("val", info)
        loss, it = float(info["val_loss"]), int(info["iteration"])
        if it > 0 and (self.best is None or loss < self.best[0]):  # it == 0 is the untrained baseline
            self.best = (loss, it)
            self.best_dir.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(self.best_dir / "adapters.safetensors"),
                                dict(tree_flatten(self.model.trainable_parameters())))
            (self.best_dir / "best.json").write_text(json.dumps({"iteration": it, "val_loss": loss}))
            print(f"[best] saved weights from iteration {it} (val_loss={loss:.4f})", flush=True)


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def parse_override(items: list[str]) -> dict:
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def prepare_data(cfg: dict, run_dir: Path) -> Path:
    """run_dir/data/{train,valid}.jsonl: train linked to the built dataset, valid = fixed seeded subset."""
    src = ROOT / cfg["dataset_dir"]
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_dst = data_dir / "train.jsonl"
    if not train_dst.exists():
        train_dst.symlink_to(src / "train.jsonl")
    valid = (src / "valid.jsonl").read_text().splitlines()
    rng = random.Random(cfg["seed"])
    keep = sorted(rng.sample(range(len(valid)), min(cfg["valid_subset"], len(valid))))
    (data_dir / "valid.jsonl").write_text("\n".join(valid[i] for i in keep) + "\n")
    return data_dir


def check_lengths(dataset, max_len: int, name: str) -> dict:
    # ChatDataset[i] is the raw example; .process() renders the chat template -> (tokens, prompt_offset)
    lens = [len(dataset.process(dataset[i])[0]) for i in range(len(dataset))]
    over = sum(l > max_len for l in lens)
    if over:
        raise SystemExit(f"{over} {name} examples exceed max_seq_length={max_len} (longest {max(lens)}). "
                         "Raise max_seq_length; refusing to train on truncated examples.")
    return {"n": len(lens), "max": max(lens), "mean": round(float(np.mean(lens)), 1), "total_tokens": int(sum(lens))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--override", nargs="*", default=[], help="key=value pairs (YAML-parsed), e.g. iters=20")
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    cfg.update(parse_override(a.override))
    run_dir = ROOT / "runs" / cfg["run_name"]
    if (run_dir / "metrics.jsonl").exists():
        raise SystemExit(f"{run_dir} already has metrics.jsonl; pick a new run_name (runs are never overwritten)")
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); mx.random.seed(cfg["seed"])
    data_dir = prepare_data(cfg, run_dir)

    # mlx-lm argument namespace = its defaults + our config (the same resolution `mlx_lm lora` applies)
    args = dict(CONFIG_DEFAULTS)
    args.update({k: v for k, v in cfg.items() if k in CONFIG_DEFAULTS})
    args.update({"train": True, "data": str(data_dir), "adapter_path": str(run_dir / "adapters")})
    args = types.SimpleNamespace(**args)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(f"Loading {args.model} ...", flush=True)
    model, tokenizer = load(args.model)
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    lengths = {"train": check_lengths(train_set, args.max_seq_length, "train"),
               "valid": check_lengths(valid_set, args.max_seq_length, "valid")}
    print(f"lengths OK: {lengths}", flush=True)

    info = {
        "run_name": cfg["run_name"], "started": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": git_commit(),
        "python": sys.version.split()[0], "mlx": mx.__version__, "mlx_lm": MLX_LM_VERSION,
        "platform": platform.platform(), "device": mx.device_info().get("device_name"),
        "memory_gb": round(mx.device_info().get("memory_size", 0) / 2**30, 1),
        "train_sha256_16": sha16(ROOT / cfg["dataset_dir"] / "train.jsonl"),
        "valid_subset_sha256_16": sha16(data_dir / "valid.jsonl"), "lengths": lengths,
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=1))

    t0 = time.time()
    cb = JsonlMetrics(run_dir / "metrics.jsonl", model, run_dir / "best")
    train_model(args, model, train_set, valid_set, cb)
    info["train_seconds"] = round(time.time() - t0, 1)
    info["peak_memory_gb"] = round(mx.get_peak_memory() / 2**30, 2)
    if cb.best:  # make best/ loadable as an adapter dir (mlx_lm.load(model, adapter_path=run_dir/"best"))
        shutil.copy(run_dir / "adapters" / "adapter_config.json", run_dir / "best" / "adapter_config.json")
        info["best"] = {"iteration": cb.best[1], "val_loss": cb.best[0]}
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=1))
    print(f"done in {info['train_seconds']}s; best={info.get('best')}", flush=True)


if __name__ == "__main__":
    main()
