"""Render training/eval examples as readable Markdown for manual inspection.

    python -m src.show_examples                                  # 5 random train examples
    python -m src.show_examples --category "Cap On Liability"    # only examples where this category is positive
    python -m src.show_examples --empty --n 3                    # all-empty examples
    python -m src.show_examples --id "<contract>#4"              # one specific chunk (id from *_meta.jsonl)
    python -m src.show_examples --file eval/test_trained.jsonl   # eval rows (gold field instead of assistant)

Writes data/processed/cuad_v1/inspect.md. Each annotated span is wrapped at its exact CUAD offset as
**【Category】quote**. Other occurrences of the same string are not highlighted. Note that the training target is the
quote *text*, so the model is right to output "Company" whichever occurrence the annotator pointed at.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data/processed/cuad_v1"


def load(file: str) -> list[tuple[dict, dict]]:
    """[(row, meta)]. Training files carry meta in a line-aligned sidecar; eval files carry it inline."""
    rows = [json.loads(l) for l in open(D / file, encoding="utf-8")]
    side = D / file.replace(".jsonl", "_meta.jsonl")
    metas = [json.loads(l) for l in open(side, encoding="utf-8")] if side.exists() else rows
    return list(zip(rows, metas))


def answer_of(row: dict) -> dict[str, list[str]]:
    if "gold" in row:
        return row["gold"]
    return json.loads(row["messages"][2]["content"])


def excerpt_of(row: dict) -> str:
    u = row["messages"][1]["content"]
    return u[u.index("<excerpt>\n") + len("<excerpt>\n"): u.rindex("\n</excerpt>")]


def highlight(text: str, offsets: list[list]) -> tuple[str, list[str]]:
    """Wrap each annotated span (exact offsets from the builder) as **【Category】text**.
    A span nested inside or overlapping an already-wrapped span is not wrapped again; it is listed in the notes."""
    out, pos, notes = [], 0, []
    for cat, s, e in sorted(offsets, key=lambda o: (o[1], -o[2])):
        if s < pos:
            notes.append(f"【{cat}】 \"{text[s:e]}\" (chars {s}–{e}, inside/overlapping the previous highlight)")
            continue
        out += [text[pos:s], f"**【{cat}】", text[s:e], "**"]
        pos = e
    out.append(text[pos:])
    return "".join(out), notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="train.jsonl")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--category", help="only examples where this category has a gold quote")
    ap.add_argument("--empty", action="store_true", help="only examples with no gold quote at all")
    ap.add_argument("--id", help="a specific chunk id, e.g. 'CONTRACT#3'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=D / "inspect.md")
    a = ap.parse_args()

    pairs = load(a.file)
    if a.id:
        pairs = [p for p in pairs if p[1]["id"] == a.id]
    if a.category:
        pairs = [p for p in pairs if answer_of(p[0]).get(a.category)]
    if a.empty:
        pairs = [p for p in pairs if not any(answer_of(p[0]).values())]
    random.Random(a.seed).shuffle(pairs)
    pairs = pairs[: a.n]

    md = [f"# Inspect: `{a.file}`, {len(pairs)} example(s)\n"]
    for row, meta in pairs:
        ans = answer_of(row)
        cats = meta.get("requested") or meta.get("categories")
        text, notes = highlight(excerpt_of(row), meta.get("gold_offsets", []))
        md += [f"\n---\n## {meta['id']}",
               f"- **Type:** {meta['type']} · **chunk** {meta['chunk'] + 1}/{meta['n_chunks']} · chars {meta['chars']}",
               f"- **Requested ({len(cats)}):** {', '.join(cats)}",
               f"- **Positive:** {', '.join(c for c in cats if ans.get(c)) or '— none (all empty)'}",
               f"- **Cut by chunk boundary (not requested / not scored):** {', '.join(meta.get('cut', [])) or '—'}",
               "\n### Target answer\n```json", json.dumps(ans, indent=1, ensure_ascii=False), "```",
               "\n### Excerpt (annotated spans highlighted at their exact CUAD offsets)\n",
               text.replace("\n", "  \n")]
        if notes:
            md += ["\n**Nested/overlapping spans not highlighted inline:**"] + [f"- {n}" for n in notes]
    a.out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {a.out} ({len(pairs)} examples)")


if __name__ == "__main__":
    main()
