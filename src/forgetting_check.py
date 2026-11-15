"""Catastrophic-forgetting check: the same general prompts with the LoRA adapter off (base) and on.

    python -I src/forgetting_check.py --download                       # MMLU + GSM8K test parquet -> data/raw/forgetting/
    python -I src/forgetting_check.py --tag base
    python -I src/forgetting_check.py --tag r16 --adapter runs/qwen3-4b-lora-r16-v2/best
    python -I src/forgetting_check.py --report base r16                # -> results/forgetting/summary.{json,md}

There are three probes, all decoded greedily with the same chat template; only the adapter differs:
  * MMLU: 285 questions (5 per subject, seeded); knowledge; answer-letter exact match.
  * GSM8K: 100 problems (seeded); multi-step reasoning; final-number exact match.
  * Format probe: 8 everyday prompts with no extraction request; does the model still answer in prose, or has it
    collapsed into the trained `{category: [quotes]}` JSON? This is the most likely damage from 3k training examples
    that all share one output format, and benchmarks would not show it.
The base-vs-adapter comparison is paired: same items, so the report counts flips (right->wrong, wrong->right) and gives
an exact McNemar p-value instead of only two accuracies.

Self-contained (no project imports), so it runs with `python -I` like the other scripts that read downloaded data.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/forgetting"
OUT = ROOT / "results/forgetting"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SEED = 0
SOURCES = {"mmlu": ("cais/mmlu", "all/test-00000-of-00001.parquet"),
           "gsm8k": ("openai/gsm8k", "main/test-00000-of-00001.parquet")}

PROBES = [
    ("summarise", "Summarise this in two sentences:\n\nPhotosynthesis is the process by which green plants, algae and "
     "some bacteria convert light energy into chemical energy. Using sunlight, they turn carbon dioxide and water "
     "into glucose and release oxygen as a by-product. The process takes place mainly in the chloroplasts, which "
     "contain the pigment chlorophyll, and it is the primary source of the oxygen in Earth's atmosphere."),
    ("email", "Write a short, polite email to a supplier asking to move next Tuesday's delivery by one week."),
    ("explain_legal", "In plain English and in three sentences, what does a limitation-of-liability clause in a "
     "contract do?"),
    ("contract_question", "Read this clause and tell me in one sentence whether the supplier can terminate the "
     "agreement early.\n\n\"Either party may terminate this Agreement for convenience upon ninety (90) days' prior "
     "written notice to the other party.\""),
    ("supplier_risks", "List three risks of relying on a single supplier for a critical component."),
    ("code", "Write a Python function that returns True if a string is a palindrome, ignoring case and spaces."),
    ("reasoning", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the "
     "ball cost? Explain briefly."),
    ("translate", "Translate into French: \"The meeting has been moved to Thursday afternoon.\""),
]


# --- data --------------------------------------------------------------------------------------------------------

def download() -> None:
    from huggingface_hub import HfApi, hf_hub_download
    RAW.mkdir(parents=True, exist_ok=True)
    info = {}
    for name, (repo, path) in SOURCES.items():
        sha = HfApi().dataset_info(repo).sha
        hf_hub_download(repo, path, repo_type="dataset", revision=sha, local_dir=RAW / name)
        info[name] = {"repo": repo, "file": path, "revision": sha}
        print(f"{name}: {repo}@{sha[:10]} {path}")
    (RAW / "sources.json").write_text(json.dumps(info, indent=1))


def load_items() -> list[dict]:
    """Seeded subsets as {task, id, prompt, gold, max_tokens}."""
    import pandas as pd
    items = []
    mmlu = pd.read_parquet(RAW / "mmlu" / SOURCES["mmlu"][1])
    for subject, g in sorted(mmlu.groupby("subject")):
        for idx in random.Random(f"{SEED}-{subject}").sample(list(g.index), 5):
            r = mmlu.loc[idx]
            opts = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", r["choices"]))
            items.append({"task": "mmlu", "id": f"mmlu-{idx}", "max_tokens": 8, "gold": "ABCD"[int(r["answer"])],
                          "prompt": f"The following is a multiple-choice question about {subject.replace('_', ' ')}. "
                                    f"Answer with the letter of the correct option only.\n\n{r['question']}\n{opts}"})
    gsm = pd.read_parquet(RAW / "gsm8k" / SOURCES["gsm8k"][1])
    for idx in sorted(random.Random(SEED).sample(range(len(gsm)), 100)):
        r = gsm.loc[idx]
        items.append({"task": "gsm8k", "id": f"gsm8k-{idx}", "max_tokens": 512,
                      "gold": r["answer"].split("####")[-1].strip().replace(",", ""),
                      "prompt": "Solve the problem. Think step by step, then give the final answer on the last line "
                                f"as 'Answer: <number>'.\n\n{r['question']}"})
    for pid, prompt in PROBES:
        items.append({"task": "probe", "id": f"probe-{pid}", "max_tokens": 300, "gold": None, "prompt": prompt})
    return items


# --- scoring -----------------------------------------------------------------------------------------------------

def json_like(text: str) -> bool:
    s = text.strip()
    return s.startswith(("{", "[", "```json"))


def correct(task: str, text: str, gold: str | None) -> bool | None:
    if task == "mmlu":
        m = re.match(r"\s*\(?([ABCD])\b", text)
        return bool(m) and m.group(1) == gold
    if task == "gsm8k":
        m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text) or re.findall(r"-?[\d,]*\.?\d+", text)
        if not m:
            return False
        try:
            return abs(float(m[-1].replace(",", "")) - float(gold)) < 1e-6
        except ValueError:
            return False
    return None


def mmlu_letter(text: str) -> str | None:
    """The answer letter, also when it is wrapped as {"answer": "B"} (format drift, not a knowledge error)."""
    m = re.search(r'"answer"\s*:\s*"\(?([ABCD])', text) or re.match(r"\s*\(?([ABCD])\b", text)
    return m.group(1) if m else None


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs (b = right->wrong, c = wrong->right)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# --- generation --------------------------------------------------------------------------------------------------

def generate(tag: str, adapter: Path | None, batch_size: int) -> None:
    import mlx.core as mx
    from mlx_lm import batch_generate, load
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load(BASE_MODEL, adapter_path=str(adapter) if adapter else None)
    items = load_items()
    out = OUT / f"{tag}.jsonl"
    out.unlink(missing_ok=True)
    t_all = time.time()
    for task in ("mmlu", "gsm8k", "probe"):
        todo = [(it, tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], add_generation_prompt=True,
                                             return_dict=False)) for it in items if it["task"] == task]
        todo.sort(key=lambda t: len(t[1]))
        for b in range(0, len(todo), batch_size):
            batch = todo[b: b + batch_size]
            resp = batch_generate(model, tok, [p for _, p in batch], max_tokens=batch[0][0]["max_tokens"])
            with open(out, "a", encoding="utf-8") as fh:
                for (it, _), text in zip(batch, resp.texts):
                    fh.write(json.dumps({"task": task, "id": it["id"], "text": text, "gold": it["gold"],
                                         "correct": correct(task, text, it["gold"]), "json_like": json_like(text)},
                                        ensure_ascii=False) + "\n")
        print(f"{tag} {task}: {len(todo)} done, {time.time() - t_all:.0f}s, "
              f"peak {mx.get_peak_memory() / 2**30:.1f}GB", flush=True)
    (OUT / f"{tag}_info.json").write_text(json.dumps(
        {"model": BASE_MODEL, "adapter": str(adapter) if adapter else None, "decoding": "greedy",
         "batch_size": batch_size, "seed": SEED, "seconds": round(time.time() - t_all)}, indent=1))


# --- report ------------------------------------------------------------------------------------------------------

def report(a: str, b: str) -> None:
    rows = {t: {r["id"]: r for r in map(json.loads, open(OUT / f"{t}.jsonl", encoding="utf-8"))} for t in (a, b)}
    summary = {}
    for task in ("mmlu", "gsm8k"):
        ids = sorted(i for i, r in rows[a].items() if r["task"] == task)
        ca = [rows[a][i]["correct"] for i in ids]
        cb = [rows[b][i]["correct"] for i in ids]
        lost = sum(x and not y for x, y in zip(ca, cb))
        gained = sum(y and not x for x, y in zip(ca, cb))
        summary[task] = {"n": len(ids), f"acc_{a}": sum(ca) / len(ids), f"acc_{b}": sum(cb) / len(ids),
                         "right_to_wrong": lost, "wrong_to_right": gained, "mcnemar_p": mcnemar_p(lost, gained)}
    # Diagnosis: separate "lost the knowledge" from "answers in the wrong format".
    mm = sorted(i for i, r in rows[a].items() if r["task"] == "mmlu")
    letter = {t: {i: mmlu_letter(rows[t][i]["text"]) for i in mm} for t in (a, b)}
    answered = [i for i in mm if letter[b][i]]
    summary["mmlu_diagnosis"] = {
        f"format_{b}": {"prose": sum(not rows[b][i]["json_like"] for i in mm),
                        "json_with_answer": sum(rows[b][i]["json_like"] and '"answer"' in rows[b][i]["text"] for i in mm),
                        "json_without_answer": sum(rows[b][i]["json_like"] and '"answer"' not in rows[b][i]["text"]
                                                   for i in mm)},
        f"acc_{b}_reading_json_answer": sum(letter[b][i] == rows[b][i]["gold"] for i in mm) / len(mm),
        f"items_{b}_gave_a_letter": len(answered),
        f"acc_{a}_on_those": sum(letter[a][i] == rows[a][i]["gold"] for i in answered) / len(answered),
        f"acc_{b}_on_those": sum(letter[b][i] == rows[b][i]["gold"] for i in answered) / len(answered)}
    gs = [i for i, r in rows[a].items() if r["task"] == "gsm8k"]
    lost = [i for i in gs if rows[a][i]["correct"] and not rows[b][i]["correct"]]
    summary["gsm8k_diagnosis"] = {**{f"no_final_answer_line_{t}": sum("Answer:" not in rows[t][i]["text"] for i in gs)
                                     for t in (a, b)},
                                  "lost_items_without_final_answer_line": sum("Answer:" not in rows[b][i]["text"]
                                                                              for i in lost), "lost_items": len(lost)}
    probe_ids = sorted(i for i, r in rows[a].items() if r["task"] == "probe")
    summary["format_probe"] = {"n": len(probe_ids),
                               **{f"json_like_{t}": sum(rows[t][i]["json_like"] for i in probe_ids) for t in (a, b)}}
    summary["json_like_on_benchmarks"] = {t: sum(r["json_like"] for r in rows[t].values() if r["task"] != "probe")
                                          for t in (a, b)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    lines = [f"# Forgetting check: {a} vs {b} (adapter off vs on, greedy, same prompts)\n",
             f"| benchmark | n | {a} | {b} | Δ | right→wrong | wrong→right | McNemar p |", "|---|---|---|---|---|---|---|---|"]
    for task in ("mmlu", "gsm8k"):
        s = summary[task]
        lines.append(f"| {task} | {s['n']} | {s[f'acc_{a}']:.1%} | {s[f'acc_{b}']:.1%} | "
                     f"{(s[f'acc_{b}'] - s[f'acc_{a}']) * 100:+.1f} pt | {s['right_to_wrong']} | {s['wrong_to_right']} | "
                     f"{s['mcnemar_p']:.2f} |")
    fp = summary["format_probe"]
    lines += [f"\nFormat probe: answers that came back as JSON instead of prose: {a} {fp[f'json_like_{a}']}/{fp['n']}, "
              f"{b} {fp[f'json_like_{b}']}/{fp['n']}. JSON-like answers on the benchmark items: "
              f"{a} {summary['json_like_on_benchmarks'][a]}, {b} {summary['json_like_on_benchmarks'][b]}.\n",
              "## Diagnosis\n", "```", json.dumps({k: summary[k] for k in ("mmlu_diagnosis", "gsm8k_diagnosis")},
                                                  indent=1), "```\n",
              "## Format probe, side by side\n"]
    for i in probe_ids:
        lines.append(f"### {i}\n")
        for t in (a, b):
            txt = rows[t][i]["text"].strip().replace("\n", " ")
            lines.append(f"- **{t}:** {txt[:500]}{'…' if len(txt) > 500 else ''}")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--tag")
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--report", nargs=2, metavar=("A", "B"))
    x = ap.parse_args()
    if x.download:
        download()
    if x.tag:
        generate(x.tag, x.adapter, x.batch_size)
    if x.report:
        report(*x.report)


if __name__ == "__main__":
    main()
