"""Shared CUAD preprocessing: categories, whitespace normalization, chunking and the prompt format.

Training-data construction and inference must both use these functions. If they ever diverge, the model is
evaluated on inputs shaped differently from the ones it was trained on.
"""
from __future__ import annotations

import json
import re
from bisect import bisect_right
from pathlib import Path

# ---------------------------------------------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------------------------------------------
# Header fields are easy (mostly in the first 5% of a contract); they are reported separately as a control group.
HEADER_CATEGORIES = ["Document Name", "Parties", "Agreement Date", "Effective Date"]
CLAUSE_CATEGORIES = [
    "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal", "Governing Law",
    "Termination For Convenience", "Anti-Assignment", "Change Of Control", "Cap On Liability",
    "Uncapped Liability", "Minimum Commitment", "Exclusivity", "Audit Rights",
]
TRAIN_CATEGORIES = HEADER_CATEGORIES + CLAUSE_CATEGORIES
# Never requested during training. Used only to test whether the model learned the general skill
# ("find the clause matching this definition") or just the 16 trained labels.
HELDOUT_CATEGORIES = ["Insurance", "Post-Termination Services", "Non-Compete"]
assert not set(TRAIN_CATEGORIES) & set(HELDOUT_CATEGORIES)


def load_definitions(cuad_json: dict) -> dict[str, str]:
    """Category -> annotation guideline, taken from the 'Details:' part of each CUAD question."""
    defs = {}
    for qa in cuad_json["data"][0]["paragraphs"][0]["qas"]:
        cat = qa["id"].split("__")[-1]
        detail = qa["question"].split("Details:", 1)[1]
        defs[cat] = re.sub(r"\s+", " ", detail).strip()
    return defs


# ---------------------------------------------------------------------------------------------------------------
# Whitespace normalization (with an offset map, so gold spans stay exact substrings)
# ---------------------------------------------------------------------------------------------------------------
# A run of whitespace containing a newline becomes "\n" (one newline) or "\n\n" (two or more newlines).
# A run of horizontal whitespace (spaces, tabs, NBSP and other Unicode spaces) becomes a single " ".
_WS_RUN = re.compile(r"[^\S\n]*\n\s*|[^\S\n]+")


def normalize_whitespace(text: str) -> tuple[str, list[int]]:
    """Return (normalized_text, pos_map), where pos_map[i] is the index in normalized_text of original char i."""
    out: list[str] = []
    pos_map = [0] * len(text)
    n = last = 0
    for m in _WS_RUN.finditer(text):
        s, e = m.span()
        for i in range(last, s):
            pos_map[i] = n + (i - last)
        out.append(text[last:s])
        n += s - last
        run = m.group()
        rep = ("\n\n" if run.count("\n") > 1 else "\n") if "\n" in run else " "
        for i in range(s, e):
            pos_map[i] = n
        out.append(rep)
        n += len(rep)
        last = e
    for i in range(last, len(text)):
        pos_map[i] = n + (i - last)
    out.append(text[last:])
    return "".join(out), pos_map


def map_span(norm: str, pos_map: list[int], start: int, end: int) -> tuple[int, int] | None:
    """Map an original [start, end) span to normalized offsets, trimmed of edge whitespace."""
    ns, ne = pos_map[start], pos_map[end - 1] + 1
    seg = norm[ns:ne]
    if not seg.strip():
        return None
    return ns + (len(seg) - len(seg.lstrip())), ns + len(seg.rstrip())


# ---------------------------------------------------------------------------------------------------------------
# Chunking: windows of ~`size` chars that end on natural boundaries, with >= `overlap` chars of overlap
# ---------------------------------------------------------------------------------------------------------------
_PARAGRAPH = re.compile(r"\n\n")
_SENTENCE = re.compile(r"[.;:!?][\"'”’)\]]*\s")
_LINE = re.compile(r"\n")


def _break_points(text: str) -> tuple[list[int], list[int]]:
    """Candidate cut positions and their rank (3 = paragraph, 2 = sentence end, 1 = line end)."""
    rank: dict[int, int] = {}
    for pat, r in ((_PARAGRAPH, 3), (_SENTENCE, 2), (_LINE, 1)):
        for m in pat.finditer(text):
            rank[m.end()] = max(rank.get(m.end(), 0), r)
    pos = sorted(rank)
    return pos, [rank[p] for p in pos]


def _best_break(pos: list[int], rank: list[int], lo: int, hi: int) -> int | None:
    """Highest-ranked break in (lo, hi]; ties go to the farthest one."""
    i, j = bisect_right(pos, lo), bisect_right(pos, hi)
    if i >= j:
        return None
    best = max(rank[i:j])
    return next(pos[k] for k in range(j - 1, i - 1, -1) if rank[k] == best)


def _space_break(text: str, lo: int, hi: int) -> int:
    p = text.rfind(" ", lo + 1, hi)
    return p + 1 if p > lo else hi


def chunk_text(text: str, size: int = 6000, overlap: int = 800) -> list[tuple[int, int]]:
    """Split text into [start, end) windows of at most `size` chars that cut at paragraph or sentence boundaries
    where possible. Consecutive windows share at least `overlap` chars, so a clause cut at one boundary appears
    whole in the neighbouring window (unless it is longer than the overlap)."""
    n = len(text)
    if n <= size:
        return [(0, n)]
    pos, rank = _break_points(text)
    chunks, start = [], 0
    while True:
        if n - start <= size:
            chunks.append((start, n))
            return chunks
        end = _best_break(pos, rank, start + int(size * 0.8), start + size) or _space_break(text, start, start + size)
        chunks.append((start, end))
        nxt = _best_break(pos, rank, max(start, end - 2 * overlap), end - overlap)
        if nxt is None or nxt <= start:
            nxt = _space_break(text, start, end - overlap)
        start = nxt if start < nxt < end else end - overlap


# ---------------------------------------------------------------------------------------------------------------
# Prompt format (category-conditioned: the request lists categories with their definitions)
# ---------------------------------------------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a contract-review assistant for a procurement team. You extract clauses from contract excerpts. "
    "Quotes must be copied verbatim from the excerpt. A category that is not present gets an empty list. "
    "Output JSON only."
)


def build_user_prompt(excerpt: str, categories: list[str], definitions: dict[str, str]) -> str:
    cat_lines = "\n".join(f"- {c}: {definitions[c]}" for c in categories)
    return (
        "Extract the following clause categories from the contract excerpt.\n\n"
        f"Categories:\n{cat_lines}\n\n"
        "Return a JSON object whose keys are exactly these category names, in this order. Each value is a list of "
        "verbatim quotes from the excerpt; use [] if the category is absent from the excerpt.\n\n"
        f"<excerpt>\n{excerpt}\n</excerpt>"
    )


def build_messages(excerpt: str, categories: list[str], definitions: dict[str, str],
                   target: dict[str, list[str]] | None = None) -> list[dict]:
    """Chat messages. With `target`, append the assistant answer (training); without it, an inference prompt."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(excerpt, categories, definitions)}]
    if target is not None:
        assert list(target) == categories, "target keys must follow the requested order"
        msgs.append({"role": "assistant", "content": json.dumps(target, ensure_ascii=False)})
    return msgs


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    k = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            k += 1
    return k
