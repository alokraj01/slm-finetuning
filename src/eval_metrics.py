"""Scoring for the category-conditioned extraction task.

The functions here are pure, with no model dependency, so they are unit-tested and shared by the notebook and the CLI.

Matching follows the CUAD paper's evaluation: a predicted quote matches a gold quote when the Jaccard similarity of
their normalized word sets is >= 0.5. Two additions:
  * matching is one-to-one (greedy, highest Jaccard first), so a prediction repeated three times counts once,
    not three times;
  * quotes are deduplicated by normalized text before matching. CUAD labels a repeated string such as "Company" at
    several offsets, and chunk overlap makes the model emit the same quote twice.
"""
from __future__ import annotations

import json
import re
import string
from collections import defaultdict

JACCARD_THRESHOLD = 0.5
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)
_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.S)


# --- parsing --------------------------------------------------------------------------------------------------------

def parse_output(text: str, categories: list[str]) -> tuple[dict[str, list[str]] | None, dict]:
    """Model text -> ({category: [quotes]} restricted to `categories`, flags).

    flags:
      strict_json  the raw text is a JSON object (what the fine-tuned model was trained to emit)
      json_ok      a JSON object after stripping a ```json fence (lenient, so a base model is not scored 0 just
                   for wrapping its answer in a fence)
      keys_ok      every requested category is present as a key
      extra_keys   keys that were not requested
    A non-list value is coerced: a string becomes [string], and null or anything else becomes []."""
    flags = {"strict_json": False, "json_ok": False, "keys_ok": False, "extra_keys": []}
    obj = _loads_obj(text)
    if obj is not None:
        flags["strict_json"] = True
    else:
        m = _FENCE.match(text)
        obj = _loads_obj(m.group(1)) if m else None
    if obj is None:
        return None, flags
    flags["json_ok"] = True
    flags["keys_ok"] = all(c in obj for c in categories)
    flags["extra_keys"] = [k for k in obj if k not in categories]
    out = {}
    for c in categories:
        v = obj.get(c, [])
        if isinstance(v, str):
            v = [v]
        out[c] = [q for q in v if isinstance(q, str) and q.strip()] if isinstance(v, list) else []
    return out, flags


def _loads_obj(s: str):
    try:
        o = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None
    return o if isinstance(o, dict) else None


# --- text matching --------------------------------------------------------------------------------------------------

def normalize(s: str) -> str:
    """SQuAD/CUAD answer normalization: lowercase, drop punctuation and articles, collapse whitespace."""
    return " ".join(_ARTICLES.sub(" ", s.lower().translate(_PUNCT)).split())


def jaccard(a: str, b: str) -> float:
    sa, sb = set(normalize(a).split()), set(normalize(b).split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def dedupe(quotes: list[str]) -> list[str]:
    seen, out = set(), []
    for q in quotes:
        k = normalize(q)
        if k and k not in seen:
            seen.add(k)
            out.append(q)
    return out


def match(pred: list[str], gold: list[str], thr: float = JACCARD_THRESHOLD) -> tuple[int, list[tuple[str, str]]]:
    """One-to-one greedy matching by Jaccard. Returns (true positives, matched (pred, gold) pairs)."""
    pairs = sorted(((jaccard(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gold)), reverse=True)
    used_p, used_g, matched = set(), set(), []
    for s, i, j in pairs:
        if s < thr:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i); used_g.add(j); matched.append((pred[i], gold[j]))
    return len(matched), matched


# --- aggregation ----------------------------------------------------------------------------------------------------

class Counts:
    """Micro-averaged span P/R/F1 plus presence-level counts for one bucket (a category or a group)."""

    def __init__(self):
        self.tp = self.n_pred = self.n_gold = 0
        self.items = self.gold_pos = self.pred_pos = self.both_pos = 0

    def add(self, pred: list[str], gold: list[str]):
        pred, gold = dedupe(pred), dedupe(gold)
        tp, _ = match(pred, gold)
        self.tp += tp; self.n_pred += len(pred); self.n_gold += len(gold)
        self.items += 1
        self.gold_pos += bool(gold); self.pred_pos += bool(pred); self.both_pos += bool(gold) and bool(pred)

    def __iadd__(self, o: "Counts"):
        for k in vars(self):
            setattr(self, k, getattr(self, k) + getattr(o, k))
        return self

    def summary(self) -> dict:
        if not self.items:  # nothing scored (e.g. predictions not generated yet), so don't report a fake 1.0
            return {"P": None, "R": None, "F1": None, "tp": 0, "n_pred": 0, "n_gold": 0, "items": 0,
                    "gold_pos": 0, "false_alarm_rate": None, "miss_rate": None}
        p = self.tp / self.n_pred if self.n_pred else (1.0 if not self.n_gold else 0.0)
        r = self.tp / self.n_gold if self.n_gold else (1.0 if not self.n_pred else 0.0)
        f = 2 * p * r / (p + r) if p + r else 0.0
        neg = self.items - self.gold_pos
        return {"P": p, "R": r, "F1": f, "tp": self.tp, "n_pred": self.n_pred, "n_gold": self.n_gold,
                "items": self.items, "gold_pos": self.gold_pos,
                # how often the model returns quotes where gold is []; this is the "confidently wrong" rate
                "false_alarm_rate": (self.pred_pos - self.both_pos) / neg if neg else None,
                # how often the model returns [] where gold has a quote
                "miss_rate": (self.gold_pos - self.both_pos) / self.gold_pos if self.gold_pos else None}


def score(pairs: list[tuple[str, list[str], list[str]]], groups: dict[str, list[str]]) -> dict:
    """pairs = [(category, pred, gold)] -> {"per_category": {...}, "per_group": {...}}."""
    per_cat: dict[str, Counts] = defaultdict(Counts)
    for cat, pred, gold in pairs:
        per_cat[cat].add(pred, gold)
    per_group = {}
    for g, cats in groups.items():
        tot = Counts()
        for c in cats:
            if c in per_cat:
                tot += per_cat[c]
        per_group[g] = tot.summary()
    return {"per_category": {c: per_cat[c].summary() for c in per_cat}, "per_group": per_group}


def verbatim_filter(quotes: list[str], excerpt: str) -> tuple[list[str], list[str]]:
    """Split quotes into (verbatim substrings of the excerpt, non-verbatim). The non-verbatim ones are the
    hallucination signal, and in production they would be dropped and flagged."""
    ok, bad = [], []
    for q in quotes:
        (ok if q in excerpt else bad).append(q)
    return ok, bad
