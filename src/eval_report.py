"""Join predictions with gold labels and score them at chunk and contract level.

    python -m src.eval_report --split test     # -> results/eval/summary_test.{json,md}

Predictions come from `src.predict` (results/eval/<tag>/<split>_<kind>.jsonl). Partial prediction files are fine:
  * chunk-level metrics use the chunks that have a prediction;
  * contract-level metrics use only the contracts whose chunks *all* have a prediction.
Coverage is reported alongside, so a partial score is never mistaken for a final one.

Scoring rules (see CLAUDE.md):
  * header fields are scored at contract level only, because CUAD offsets point at the first occurrence of a
    repeated string;
  * at chunk level, a category whose gold span is cut by the chunk boundary is not scored;
  * a JSON parse failure counts as [] for every category, i.e. as misses.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.cuad_data import CLAUSE_CATEGORIES, HEADER_CATEGORIES, HELDOUT_CATEGORIES
from src.eval_metrics import dedupe, parse_output, score, verbatim_filter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/cuad_v1"
RESULTS = ROOT / "results/eval"
TAGS = {"base": "base-qwen3-4b", "ft": "qwen3-4b-lora-r16-v2"}  # the main comparison: base vs the chosen r16 model
# rank ablation: same recipe, only the LoRA rank differs (alpha = 2r). Prediction tag == run name.
ABLATION = {"base": "base-qwen3-4b", "r4": "qwen3-4b-lora-r4-v1", "r16": "qwen3-4b-lora-r16-v2"}  # r8 dropped (time)
GROUPS = {"header": HEADER_CATEGORIES, "clause": CLAUSE_CATEGORIES, "heldout": HELDOUT_CATEGORIES}
CHUNK_GROUPS = {"clause": CLAUSE_CATEGORIES, "heldout": HELDOUT_CATEGORIES}  # no header at chunk level


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def excerpt_of(row: dict) -> str:
    u = row["messages"][1]["content"]
    return u[u.index("<excerpt>\n") + len("<excerpt>\n"): u.rindex("\n</excerpt>")]


def load(split: str, kind: str, tag: str) -> list[dict]:
    """Eval rows joined with the predictions of `tag`. Each row gets:
      pred      {category: [quotes]} (all [] if the output was not parseable)
      verbatim  the same, keeping only quotes that are exact substrings of the excerpt
      dropped   the non-verbatim quotes, i.e. the hallucination signal
      flags     parse flags; text, n_gen, truncated and seconds come from the generation
    Rows without a prediction get has_pred=False."""
    preds = {p["id"]: p for p in _jsonl(RESULTS / tag / f"{split}_{kind}.jsonl")}
    out = []
    for r in _jsonl(DATA / f"eval/{split}_{kind}.jsonl"):
        p = preds.get(r["id"])
        row = {k: r[k] for k in ("id", "contract", "type", "chunk", "n_chunks", "categories", "gold", "cut")}
        row["excerpt"] = excerpt_of(r)
        row["has_pred"] = p is not None
        if p:
            parsed, flags = parse_output(p["text"], r["categories"])
            pred = parsed or {c: [] for c in r["categories"]}
            ver, dropped = {}, {}
            for c, qs in pred.items():
                ver[c], dropped[c] = verbatim_filter(qs, row["excerpt"])
            row.update(text=p["text"], pred=pred, verbatim=ver, dropped=dropped, flags=flags,
                       n_gen=p["n_gen"], truncated=p["truncated"], seconds=p["seconds"])
        out.append(row)
    return out


def health(rows: list[dict]) -> dict:
    """Output-format and hallucination health of one prediction set."""
    done = [r for r in rows if r["has_pred"]]
    n = len(done)
    if not n:
        return {"chunks": 0, "coverage": 0.0}
    n_quotes = sum(len(q) for r in done for q in r["pred"].values())
    n_bad = sum(len(q) for r in done for q in r["dropped"].values())
    return {"chunks": n, "coverage": n / len(rows),
            "strict_json": sum(r["flags"]["strict_json"] for r in done) / n,
            "json_ok": sum(r["flags"]["json_ok"] for r in done) / n,
            "all_keys": sum(r["flags"]["keys_ok"] for r in done) / n,
            "truncated": sum(r["truncated"] for r in done) / n,
            "quotes": n_quotes,
            "verbatim_rate": 1 - n_bad / n_quotes if n_quotes else None,
            "mean_gen_tokens": sum(r["n_gen"] for r in done) / n,
            "s_per_chunk": sum(r["seconds"] for r in done) / n}


def chunk_scores(rows: list[dict], field: str = "pred") -> dict:
    pairs = [(c, r[field][c], r["gold"][c]) for r in rows if r["has_pred"]
             for c in r["categories"] if c not in r["cut"] and c not in HEADER_CATEGORIES]
    return score(pairs, CHUNK_GROUPS)


def merge_contracts(rows: list[dict], field: str = "pred") -> dict[str, dict[str, list[str]]]:
    """{contract: {category: deduplicated quotes from all its chunks}}, only for fully predicted contracts."""
    by_c = defaultdict(list)
    for r in rows:
        by_c[r["contract"]].append(r)
    out = {}
    for c, rs in by_c.items():
        if all(r["has_pred"] for r in rs):
            out[c] = {cat: dedupe([q for r in rs for q in r[field][cat]]) for cat in rs[0]["categories"]}
    return out


def contract_scores(rows: list[dict], gold_contract: dict[str, dict], field: str = "pred") -> dict:
    merged = merge_contracts(rows, field)
    pairs = [(cat, qs, gold_contract[c][cat]) for c, cats in merged.items() for cat, qs in cats.items()]
    return {**score(pairs, GROUPS), "contracts": len(merged)}


def evaluate(split: str, tags: dict[str, str] | None = None) -> dict:
    gold_contract = {g["contract"]: g["gold"] for g in _jsonl(DATA / f"eval/{split}_gold_contract.jsonl")}
    res = {}
    for name, tag in (tags or TAGS).items():
        res[name] = {}
        for kind in ("trained", "heldout"):
            rows = load(split, kind, tag)
            res[name][kind] = {"health": health(rows),
                               "chunk": chunk_scores(rows), "chunk_verbatim": chunk_scores(rows, "verbatim"),
                               "contract": contract_scores(rows, gold_contract),
                               "contract_verbatim": contract_scores(rows, gold_contract, "verbatim")}
    return res


def adapter_params(path: Path) -> int | None:
    """Number of trainable LoRA parameters, read from the safetensors header (no weights loaded)."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
    n = 0
    for k, v in header.items():
        if k != "__metadata__":
            size = 1
            for d in v["shape"]:
                size *= d
            n += size
    return n


def ablation_table(res: dict, tags: dict[str, str]) -> list[dict]:
    """One row per model: training cost and best val loss (from runs/), plus the test metrics that matter."""
    out = []
    for name, tag in tags.items():
        run = ROOT / "runs" / tag
        info = json.loads((run / "run_info.json").read_text()) if (run / "run_info.json").exists() else {}
        g = lambda kind, level, group, key="F1": res[name][kind][level]["per_group"].get(group, {}).get(key)
        h = res[name]["trained"]["health"]
        out.append({
            "model": name, "rank": None if name == "base" else int(name[1:]),
            "trainable_params_M": (adapter_params(run / "best/adapters.safetensors") or 0) / 1e6 or None,
            "train_h": info.get("train_seconds", 0) / 3600 or None,
            "best_val_loss": info.get("best", {}).get("val_loss"),
            "header_F1": g("trained", "contract", "header"), "clause_F1": g("trained", "contract", "clause"),
            "clause_F1_chunk": g("trained", "chunk", "clause"),
            "heldout_F1": g("heldout", "contract", "heldout"), "heldout_F1_chunk": g("heldout", "chunk", "heldout"),
            "clause_false_alarm": g("trained", "chunk", "clause", "false_alarm_rate"),
            "strict_json": h.get("strict_json"), "verbatim_rate": h.get("verbatim_rate"),
            "coverage_trained": h.get("coverage"), "coverage_heldout": res[name]["heldout"]["health"].get("coverage"),
        })
    return out


def headline_table(res: dict) -> list[dict]:
    """One row per (level, group): F1/P/R for base and fine-tuned."""
    out = []
    for level in ("contract", "chunk"):
        for group, kind in (("header", "trained"), ("clause", "trained"), ("heldout", "heldout")):
            if level == "chunk" and group == "header":
                continue
            row = {"level": level, "group": group}
            for m in ("base", "ft"):
                s = res[m][kind][level]["per_group"].get(group, {})
                row.update({f"{m}_F1": s.get("F1"), f"{m}_P": s.get("P"), f"{m}_R": s.get("R"),
                            f"{m}_false_alarm": s.get("false_alarm_rate")})
            out.append(row)
    return out


def to_markdown(res: dict, split: str) -> str:
    f = lambda v: "–" if v is None else f"{v:.3f}"
    lines = [f"# Evaluation summary: {split}\n", "## Span F1 (Jaccard >= 0.5), base vs fine-tuned\n",
             "| level | group | base F1 | FT F1 | Δ F1 | base P / R | FT P / R |", "|---|---|---|---|---|---|---|"]
    for r in headline_table(res):
        d = None if r["base_F1"] is None or r["ft_F1"] is None else r["ft_F1"] - r["base_F1"]
        lines.append(f"| {r['level']} | {r['group']} | {f(r['base_F1'])} | {f(r['ft_F1'])} | "
                     f"{'–' if d is None else f'{d:+.3f}'} | {f(r['base_P'])} / {f(r['base_R'])} | "
                     f"{f(r['ft_P'])} / {f(r['ft_R'])} |")
    lines += ["\n## Output health\n", "| model | set | chunks | coverage | strict JSON | JSON (lenient) | "
              "verbatim quotes | truncated | gen tokens | s/chunk |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("base", "ft"):
        for kind in ("trained", "heldout"):
            h = res[m][kind]["health"]
            if not h["chunks"]:
                lines.append(f"| {m} | {kind} | 0 | 0 | | | | | | |"); continue
            lines.append(f"| {m} | {kind} | {h['chunks']} | {h['coverage']:.0%} | {h['strict_json']:.1%} | "
                         f"{h['json_ok']:.1%} | {f(h['verbatim_rate'])} | {h['truncated']:.1%} | "
                         f"{h['mean_gen_tokens']:.0f} | {h['s_per_chunk']:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    res = evaluate(a.split)
    (RESULTS / f"summary_{a.split}.json").write_text(json.dumps(res, indent=1))
    md = to_markdown(res, a.split)
    (RESULTS / f"summary_{a.split}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
