"""Build the CUAD fine-tuning dataset (category-conditioned clause extraction).

    python -m src.build_dataset            # from 02_slm_finetuning/, defaults below

Pipeline:
  1. Load CUAD_v1.json. Normalize whitespace and map every gold span to the normalized text.
  2. Split contracts (not chunks) into train/val/test: seeded, stratified by contract type.
  3. Chunk each contract at paragraph/sentence boundaries (~size chars, >= overlap chars of overlap).
  4. Gold per chunk: spans lying wholly inside the chunk. A span that the chunk cuts is "cut" for that chunk.
  5. Training examples (train and val, for loss monitoring): one per chunk. The requested category set is
     every positive category plus random negatives, k ~ U[k_min, 16], in shuffled order. Categories cut in that
     chunk are never requested, so no example asks for a clause that is only half visible. Chunks with no
     positive trained category are kept with probability `neg_keep`.
  6. Eval sets (val and test): every chunk, requesting (a) all 16 trained categories in canonical order and
     (b) the 3 held-out categories. Plus contract-level gold for scoring merged predictions.
  7. Write a data card and stats.json next to the data.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from src.cuad_data import (HELDOUT_CATEGORIES, TRAIN_CATEGORIES, build_messages, chunk_text, load_definitions,
                           map_span, normalize_whitespace, write_jsonl)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/cuad_v1/CUAD_v1"
OUT = ROOT / "data/processed/cuad_v1"
ALL_CATS = TRAIN_CATEGORIES + HELDOUT_CATEGORIES


# ---------------------------------------------------------------------------------------------------------------
def norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.casefold())


def contract_types() -> dict[str, str]:
    """Contract type from the PDF folder layout full_contract_pdf/Part_*/<Type>/..."""
    out = {}
    for p in (RAW / "full_contract_pdf").rglob("*"):
        if p.is_file() and p.suffix.lower() == ".pdf":
            t = p.relative_to(RAW / "full_contract_pdf").parts[1].replace("_", " ")
            t = re.sub(r"\b(Agreements?|Filing)\b", "", t, flags=re.I)
            out[norm_key(p.stem)] = re.sub(r"\s+", " ", t).strip(" -").title()
    return out


def lookup_type(title: str, types: dict[str, str]) -> str:
    k = norm_key(title)
    if k in types:
        return types[k]
    m = difflib.get_close_matches(k, types.keys(), n=1, cutoff=0.9)
    return types[m[0]] if m else "Unknown"


def load_contracts() -> list[dict]:
    raw = json.load(open(RAW / "CUAD_v1.json", encoding="utf-8"))
    types = contract_types()
    contracts = []
    for d in raw["data"]:
        p = d["paragraphs"][0]
        text, pos_map = normalize_whitespace(p["context"])
        spans = defaultdict(list)  # category -> [(start, end)] in normalized text
        for q in p["qas"]:
            cat = q["id"].split("__")[-1]
            if cat not in ALL_CATS:
                continue
            for a in q["answers"]:
                s = a["answer_start"]
                m = map_span(text, pos_map, s, s + len(a["text"]))
                if m and m not in spans[cat]:
                    spans[cat].append(m)
        for cat in spans:
            spans[cat].sort()
        contracts.append({"contract": d["title"], "type": lookup_type(d["title"], types), "text": text,
                          "orig_chars": len(p["context"]), "spans": dict(spans)})
    return contracts, raw


def split_contracts(contracts: list[dict], seed: int, val: float, test: float) -> dict[str, str]:
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for c in contracts:
        by_type[c["type"]].append(c["contract"])
    split = {}
    for t in sorted(by_type):
        names = sorted(by_type[t])
        rng.shuffle(names)
        n_te = max(1, round(len(names) * test))
        n_va = max(1, round(len(names) * val)) if len(names) >= 3 else 0
        for i, n in enumerate(names):
            split[n] = "test" if i < n_te else "val" if i < n_te + n_va else "train"
    return split


def chunk_gold(c: dict, cs: int, ce: int, cats: list[str]) -> tuple[dict[str, list[str]], list[str], list]:
    """Quotes wholly inside [cs, ce) per category (deduplicated, in reading order), the categories cut there,
    and the exact annotated spans as [category, start, end] relative to the chunk (for inspection only: the
    training target is quote text, so it does not depend on which occurrence the annotator's offset points to)."""
    gold, cut, offsets = {}, [], []
    for cat in cats:
        quotes = []
        for s, e in c["spans"].get(cat, []):
            if s >= cs and e <= ce:
                offsets.append([cat, s - cs, e - cs])
                q = c["text"][s:e]
                if q not in quotes:
                    quotes.append(q)
            elif s < ce and e > cs:
                cut.append(cat)
        gold[cat] = quotes
    return gold, sorted(set(cut)), offsets


# ---------------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=6000, help="max chunk chars (~1.5k tokens)")
    ap.add_argument("--overlap", type=int, default=800, help="min overlap chars between consecutive chunks")
    ap.add_argument("--neg-keep", type=float, default=0.5, help="keep-probability for chunks with no positive")
    ap.add_argument("--k-min", type=int, default=4, help="min categories requested per training example")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    contracts, raw = load_contracts()
    defs = load_definitions(raw)
    split = split_contracts(contracts, args.seed, args.val, args.test)
    rng = random.Random(args.seed)

    train_rows, train_meta, val_rows, val_meta = [], [], [], []
    eval_rows = {("val", "trained"): [], ("val", "heldout"): [], ("test", "trained"): [], ("test", "heldout"): []}
    gold_rows = {"val": [], "test": []}
    st = Counter()
    req_count, pos_count = Counter(), Counter()
    chunk_lens, prompt_chars, target_chars = [], [], []
    spans_total = spans_never_whole = 0

    for c in sorted(contracts, key=lambda x: x["contract"]):
        sp = split[c["contract"]]
        chunks = chunk_text(c["text"], args.size, args.overlap)
        st[f"chunks_{sp}"] += len(chunks)
        chunk_lens += [e - s for s, e in chunks]
        for cat in TRAIN_CATEGORIES:
            for s, e in c["spans"].get(cat, []):
                spans_total += 1
                spans_never_whole += not any(s >= cs and e <= ce for cs, ce in chunks)

        for ci, (cs, ce) in enumerate(chunks):
            excerpt = c["text"][cs:ce]
            cid = f"{c['contract']}#{ci}"
            meta_base = {"id": cid, "contract": c["contract"], "type": c["type"], "chunk": ci,
                         "n_chunks": len(chunks), "chars": [cs, ce]}

            if sp in ("train", "val"):
                gold, cut, offsets = chunk_gold(c, cs, ce, TRAIN_CATEGORIES)
                avail = [k for k in TRAIN_CATEGORIES if k not in cut]
                pos = [k for k in avail if gold[k]]
                keep = bool(pos) or rng.random() <= args.neg_keep
                st[f"neg_chunks_dropped_{sp}"] += not keep
            if sp in ("train", "val") and keep:
                st[f"cut_exclusions_{sp}"] += len(cut)
                k = max(rng.randint(args.k_min, len(TRAIN_CATEGORIES)), len(pos))
                neg = [x for x in avail if not gold[x]]
                req = pos + rng.sample(neg, min(k - len(pos), len(neg)))
                rng.shuffle(req)
                target = {x: gold[x] for x in req}
                msgs = build_messages(excerpt, req, defs, target)
                for x in req:  # invariant: every target quote is an exact substring of the excerpt
                    assert all(q in excerpt for q in target[x]), (cid, x)
                row, meta = {"messages": msgs}, {**meta_base, "requested": req,
                                                 "positive": [x for x in req if target[x]], "cut": cut,
                                                 "gold_offsets": [o for o in offsets if o[0] in req]}
                if sp == "train":
                    train_rows.append(row); train_meta.append(meta)
                    req_count.update(req); pos_count.update(meta["positive"])
                    prompt_chars.append(sum(len(m["content"]) for m in msgs[:2]))
                    target_chars.append(len(msgs[2]["content"]))
                else:
                    val_rows.append(row); val_meta.append(meta)

            if sp in ("val", "test"):
                for kind, cats in (("trained", TRAIN_CATEGORIES), ("heldout", HELDOUT_CATEGORIES)):
                    gold, cut, offsets = chunk_gold(c, cs, ce, cats)
                    eval_rows[(sp, kind)].append({**meta_base, "categories": cats,
                                                  "messages": build_messages(excerpt, cats, defs),
                                                  "gold": gold, "cut": cut, "gold_offsets": offsets})

        if sp in ("val", "test"):
            gold_rows[sp].append({"contract": c["contract"], "type": c["type"], "n_chunks": len(chunks),
                                  "gold": {cat: [c["text"][s:e] for s, e in c["spans"].get(cat, [])]
                                           for cat in ALL_CATS}})

    # ---- write ------------------------------------------------------------------------------------------------
    out = args.out
    files = {
        "train.jsonl": train_rows, "valid.jsonl": val_rows,  # chat format only (MLX-LM / TRL compatible)
        "train_meta.jsonl": train_meta, "valid_meta.jsonl": val_meta,
        "eval/val_trained.jsonl": eval_rows[("val", "trained")], "eval/val_heldout.jsonl": eval_rows[("val", "heldout")],
        "eval/test_trained.jsonl": eval_rows[("test", "trained")], "eval/test_heldout.jsonl": eval_rows[("test", "heldout")],
        "eval/val_gold_contract.jsonl": gold_rows["val"], "eval/test_gold_contract.jsonl": gold_rows["test"],
    }
    counts = {name: write_jsonl(out / name, rows) for name, rows in files.items()}
    (out / "splits.json").write_text(json.dumps(dict(sorted(split.items())), indent=1))
    (out / "categories.json").write_text(json.dumps(
        {"train": TRAIN_CATEGORIES, "heldout": HELDOUT_CATEGORIES, "definitions": {c: defs[c] for c in ALL_CATS}},
        indent=1, ensure_ascii=False))

    # ---- stats ------------------------------------------------------------------------------------------------
    split_counts = Counter(split.values())
    test_pos = {cat: sum(1 for g in gold_rows["test"] if g["gold"][cat]) for cat in ALL_CATS}
    val_pos = {cat: sum(1 for g in gold_rows["val"] if g["gold"][cat]) for cat in ALL_CATS}
    orig_chars = sum(c["orig_chars"] for c in contracts)
    norm_chars = sum(len(c["text"]) for c in contracts)
    sha = hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest()[:16]
    stats = {
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "contracts": dict(split_counts), "files": counts, "train_sha256_16": sha,
        "whitespace_normalization": {"orig_chars": orig_chars, "norm_chars": norm_chars,
                                     "saved": round(1 - norm_chars / orig_chars, 4)},
        "chunks": {"total": len(chunk_lens), **{k: v for k, v in st.items() if k.startswith("chunks_")},
                   "median_chars": int(statistics.median(chunk_lens)), "max_chars": max(chunk_lens)},
        "trained_spans_never_whole_in_any_chunk": f"{spans_never_whole}/{spans_total}",
        "neg_chunks_dropped": {k: v for k, v in st.items() if k.startswith("neg_")},
        "cut_category_exclusions": {k: v for k, v in st.items() if k.startswith("cut_")},
        "train_examples": {"n": len(train_rows),
                           "all_empty": sum(1 for m in train_meta if not m["positive"]),
                           "median_prompt_chars": int(statistics.median(prompt_chars)),
                           "max_prompt_chars": max(prompt_chars),
                           "approx_median_prompt_tokens": int(statistics.median(prompt_chars) / 4),
                           "approx_max_total_tokens": int(max(p + t for p, t in zip(prompt_chars, target_chars)) / 4),
                           "median_target_chars": int(statistics.median(target_chars))},
        "train_per_category": {cat: {"requested": req_count[cat], "positive": pos_count[cat]} for cat in TRAIN_CATEGORIES},
        "contract_level_positives": {"val": val_pos, "test": test_pos},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    write_data_card(out, stats, defs)
    print(json.dumps({k: stats[k] for k in ("contracts", "files", "chunks", "train_examples",
                                            "trained_spans_never_whole_in_any_chunk", "whitespace_normalization")}, indent=1))


def write_data_card(out: Path, s: dict, defs: dict) -> None:
    p, te = s["params"], s["train_examples"]
    rows = "\n".join(
        f"| {c} | {'header' if c in TRAIN_CATEGORIES[:4] else 'clause' if c in TRAIN_CATEGORIES else '**held-out**'} | "
        f"{s['train_per_category'].get(c, {}).get('requested', 0)} | {s['train_per_category'].get(c, {}).get('positive', 0)} | "
        f"{s['contract_level_positives']['val'][c]} | {s['contract_level_positives']['test'][c]} |"
        for c in TRAIN_CATEGORIES + HELDOUT_CATEGORIES)
    card = f"""# Data card: CUAD v1 → category-conditioned clause extraction

_Generated by `python -m src.build_dataset`. Do not edit by hand; re-run the builder._

## Source
- **CUAD v1** (Contract Understanding Atticus Dataset), The Atticus Project. Zenodo record 4595826,
  DOI 10.5281/zenodo.4595826, **CC BY 4.0**. Hendrycks et al., NeurIPS 2021 Datasets & Benchmarks (arXiv:2103.06268).
- 510 commercial contracts from SEC EDGAR filings, with expert span labels for 41 clause categories.
- Labels are taken from `CUAD_v1.json` (exact character offsets). `master_clauses.csv` is not used for spans,
  because it merges non-contiguous fragments with `<omitted>`.

## Cleaning
- Whitespace normalization: runs of spaces, tabs and Unicode spaces → one space; whitespace containing one newline → `\\n`;
  two or more newlines → `\\n\\n`. Gold spans are re-mapped through an offset map, so every target quote is an
  exact substring of the model input (this is asserted for every example). Characters saved: {s['whitespace_normalization']['saved']:.1%}.
- No other text edits. Redactions (`[***]`), page headers and PDF artifacts are kept, because they will be present at inference time too.

## Task format
Chat format (`system` / `user` / `assistant`). The user turn lists the **requested categories with their CUAD
definitions**, then the contract excerpt. The assistant turn is a JSON object with exactly those keys, in request
order, each mapping to a list of verbatim quotes, or `[]` when absent.

## Chunking
Contracts are split into windows of ≤ {p['size']} chars (≈{p['size']//4} tokens) that end at paragraph or sentence
boundaries, with ≥ {p['overlap']} chars of overlap. Total chunks: {s['chunks']['total']} (median {s['chunks']['median_chars']} chars).
Trained-category spans that never appear whole in any chunk: {s['trained_spans_never_whole_in_any_chunk']}.

## Split (by contract, never by chunk)
Seeded ({p['seed']}) and stratified by contract type: train {s['contracts'].get('train')} / val {s['contracts'].get('val')} /
test {s['contracts'].get('test')} contracts. The assignment is in `splits.json`.

## Training examples (`train.jsonl`, `valid.jsonl`)
- One example per chunk. Requested categories = every positive trained category in the chunk, plus random negatives up
  to k ~ U[{p['k_min']}, {len(TRAIN_CATEGORIES)}], in shuffled order. This teaches the model to read the request list, not a fixed schema.
- A category whose span is **cut** by the chunk boundary is not requested for that chunk. This avoids teaching
  "half a clause = absent". Exclusions: {s['cut_category_exclusions']}.
- Chunks with no positive trained category are kept with probability {p['neg_keep']}. Dropped: {s['neg_chunks_dropped']}.
- Train: **{te['n']} examples** ({te['all_empty']} all-empty), median prompt ≈{te['approx_median_prompt_tokens']} tokens,
  max prompt + target ≈{te['approx_max_total_tokens']} tokens (approximate, chars / 4). `valid.jsonl`: {s['files']['valid.jsonl']} examples.
- `*_meta.jsonl` files are line-aligned with the training files and record the contract, chunk offsets, and requested, positive and cut categories.

## Evaluation sets (`eval/`)
- `{{val,test}}_trained.jsonl`: every chunk of each val/test contract, requesting all {len(TRAIN_CATEGORIES)} trained categories in canonical order.
- `{{val,test}}_heldout.jsonl`: the same chunks, requesting only the {len(HELDOUT_CATEGORIES)} **held-out** categories
  ({', '.join(HELDOUT_CATEGORIES)}). These are never requested in training. They test generalization versus memorization.
- `{{val,test}}_gold_contract.jsonl`: all gold quotes per contract, for scoring merged (contract-level) predictions.
- Chunk-level gold excludes spans cut by that chunk (listed under `cut`). Contract-level gold includes everything.

## Per-category counts
| Category | Group | Train: requested | Train: positive | Val contracts w/ label | Test contracts w/ label |
|---|---|---|---|---|---|
{rows}

## Known limitations
- Token counts are approximate (chars / 4) until the base model's tokenizer is fixed.
- The labels inherit CUAD's annotation conventions (for example, Parties includes defined-term aliases such as "Seller").
- The contracts are US public-company SEC exhibits (1994–2020); private or non-US supplier contracts may differ.
"""
    (out / "DATA_CARD.md").write_text(card)


if __name__ == "__main__":
    main()
