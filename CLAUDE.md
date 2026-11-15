# CLAUDE.md — Track B (SLM Fine-Tuning)

This folder is the **Track B: SLM Fine-Tuning** attempt of the Avathon AI/ML Hiring Challenge.
The brief is `Avathon_AI_Hiring_Challenge (3).docx`. The parent repo (`../`) contains the
separate Track A (agentic, S1 supply chain) work. Keep the two apart; the challenge forbids
combining tracks.

## Track B requirements (from the brief)
- Base model ≤ 13B params; PEFT only (LoRA / QLoRA / DoRA / IA³). Justify rank, alpha and target modules.
- Document the data pipeline (source, cleaning, formatting, split, sample count). The brief rates this as important as the training code.
- At least one automatic metric, plus a base-vs-fine-tuned qualitative comparison on the same prompt.
- A reproducible training script with every hyperparameter logged.
- The fine-tuned SLM must be the primary inference model. A larger LLM may be used only for data synthesis or eval scoring.
- Rubric: reasoning beats accuracy. Algorithm selection is 30%, so every choice needs a named, rejected alternative.
- Strong extras: catastrophic-forgetting check, hallucination detection, deployment story.

## Scenario and task
- Scenario **S2: Gen AI for Enterprise Documents**, framed as supplier-contract risk review.
- Task (**category-conditioned**): the prompt lists the requested categories with their CUAD definitions, followed by a
  contract chunk. The output is JSON `{category: [verbatim quotes]}` in request order, with `[]` when a category is absent.
  The model learns the skill of finding the clause that matches a definition, not a fixed schema.
- Categories (decided 2026-11-14; source of truth is `src/cuad_data.py`):
  - **16 trained**, in two groups. Header (control): Document Name, Parties, Agreement Date, Effective Date.
    Clause: Expiration Date, Renewal Term, Notice Period To Terminate Renewal, Governing Law, Termination For Convenience,
    Anti-Assignment, Change Of Control, Cap On Liability, Uncapped Liability, Minimum Commitment, Exclusivity, Audit Rights.
  - **3 held-out** (never requested in training; test only, as evidence for generalization vs memorization):
    Insurance, Post-Termination Services, Non-Compete.
  - The other 22 are excluded by four filters: too rare (<10%); defined relative to another category;
    IP/licensing domain; scope. In production they would be covered by category-conditioned prompting,
    adapter-off fallback, per-family adapters, or keyword flags plus human review.
- Chunking: ≤6,000 chars with ≥800 overlap, cut at paragraph/sentence boundaries. A category whose span is cut by
  a chunk is not requested for that chunk in training ("half a clause" is never labeled absent).
- Metrics: span-level P/R/F1 (Jaccard overlap, as in the CUAD paper) and JSON validity, reported per category and per
  group. There are two levels: **chunk** (diagnostic) and **contract** (merge chunk outputs, deduplicate; this is the headline).
  Hallucination check: every extracted quote must be a verbatim substring of the input.
- **Header fields are scored at contract level only. Clause fields are scored at both levels** (decided 2026-11-14).
  Reason: CUAD labels a short repeated string (e.g. "Company", a date) once, and its `answer_start` points at the
  *first* occurrence in 90% of such cases (an artifact of how the dataset was packaged). A later chunk that also
  contains the string therefore has chunk-level gold `[]`, so a correct extraction would count as a false positive.
  Clause spans are unique sentences, so chunk-level scoring is fair for them.
- Known CUAD label noise (for the error-analysis section): (1) one-off span-edge slips, e.g. Document Name
  `"STRATEGIC ALLIANCE AGREEMENT is"` (1 of 521); (2) offsets point at the first occurrence of repeated strings (90% of 2,218),
  so trust the quote text, not the position; (3) generic role aliases ("Company", "Seller") are labeled as Parties in only
  37% of contracts. That inconsistency caps the achievable Parties score. Original labels are kept, not cleaned.
- Split **by contract, never by chunk**: seeded and stratified by type, 350/80/80 (`data/processed/cuad_v1/splits.json`).
- Rejected datasets (for the write-up): LEDGAR (a DeBERTa classifier would win, which weakens the SLM case),
  ContractNLI (NDA-only, 17 fixed hypotheses), synthetic S1 emails (circular evaluation),
  S4 forecasting (a poor fit for an SLM).

## Fine-tuning dataset: CUAD v1 (reference)
- **Name:** Contract Understanding Atticus Dataset (CUAD) v1, by The Atticus Project
- **Source (downloaded from):** Zenodo record 4595826, https://zenodo.org/records/4595826
  - DOI: 10.5281/zenodo.4595826
  - File: `CUAD_v1.zip` (105,883,672 bytes, md5 `c38f490a984420b8a62600db401fafd5`), downloaded 2026-11-14
- **Other mirrors:** https://www.atticusprojectai.org/cuad · https://github.com/TheAtticusProject/cuad ·
  https://huggingface.co/datasets/theatticusproject/cuad-qa
- **License:** CC BY 4.0. Attribution is required in the README and write-up.
- **Paper:** Hendrycks, D., Burns, C., Chen, A., Ball, S. *CUAD: An Expert-Annotated NLP Dataset for Legal
  Contract Review.* NeurIPS 2021 Datasets and Benchmarks Track. arXiv:2103.06268.

```bibtex
@article{hendrycks2021cuad,
  title   = {CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review},
  author  = {Dan Hendrycks and Collin Burns and Anya Chen and Spencer Ball},
  journal = {NeurIPS Datasets and Benchmarks},
  year    = {2021}
}
```

### Contents (verified locally)
Location: `data/raw/cuad_v1/CUAD_v1/`
- `CUAD_v1.json`: SQuAD-2.0-style QA. 510 contracts, 41 categories, 20,910 questions
  (6,702 with answers, 13,823 answer spans). Each question id has the form `<contract>__<Category>`.
- `full_contract_txt/`: 510 plain-text contracts. This is the input text for chunking.
- `full_contract_pdf/` (Part_I–III): the original EDGAR PDFs. These are not needed for training.
- `master_clauses.csv`: one row per contract with clause text and answers for each category.
- `label_group_xlsx/`: label reports for each category group.

CUAD_v1.json ships with no train/test split. The split used in the original paper is in the GitHub repo's
`data.zip` (`train_separate_questions.json` / `test.json`). Either use that split or make a seeded,
contract-level split, and document which one was used.

## Environment and notebooks
- Track B has its own venv, `02_slm_finetuning/.venv` (Python 3.12, managed with uv; pins in `requirements.txt`). It is
  registered as the Jupyter kernel `track-b-slm`. Do not use the Track A `../.venv`.
- `notebooks/01_cuad_eda.ipynb`: step-by-step EDA of CUAD. Its Takeaways cell records the data-driven design decisions
  (spans come from the JSON, not the CSV; chunks of about 1.5k tokens; 12 candidate categories; contract-level split).

## Dataset pipeline
- `src/cuad_data.py`: shared category lists, whitespace normalization (with an offset map), chunker and prompt
  builder. **Training and inference must both use it.**
- `python -m src.build_dataset` writes to `data/processed/cuad_v1/`. It is deterministic (seed 42; the
  `train_sha256_16` in stats.json changes only if inputs or params change). Outputs: `train.jsonl` / `valid.jsonl`
  (chat format, MLX-LM/TRL compatible), line-aligned `*_meta.jsonl`, `eval/` (every val/test chunk with all trained or
  all held-out categories, plus contract-level gold), and `DATA_CARD.md` (generated; the source for the write-up's data section).
- Tests: `.venv/bin/python -m pytest` (normalization offset map, chunk coverage/overlap, prompt format).

## Training (LoRA, MLX-LM)
- Base model: **Qwen/Qwen3-4B-Instruct-2507** (decided 2026-11-14). It is dense `qwen3`, non-thinking, Apache-2.0, and trained in bf16 (no QLoRA).
  Rejected: Qwen3.5 (hybrid linear attention; open mlx-lm LoRA bugs), Qwen3-8B (2× cost; this is the step up), Llama-3.x,
  Gemma (gated), Phi-4-mini.
- `notebooks/02_train_lora_qwen3_4b.ipynb` holds the decisions: chat template and loss mask, real token lengths →
  `max_seq_length`, a hyperparameter table with a rationale and rejected alternatives for each, a memory budget, and a
  20-iteration smoke test with a time estimate. It writes `configs/qwen3-4b-lora-r16.yaml`. Full training runs in the
  background from cell `RUN_FULL`.
- `python -m src.train_lora --config <yaml> [--override k=v ...]` wraps mlx-lm 0.32.2 and does four things:
  1. Fails if any example exceeds `max_seq_length` (mlx-lm otherwise truncates silently).
  2. Uses a fixed, seeded validation subset evaluated in full (mlx-lm otherwise re-samples `val_batches`).
  3. Writes `metrics.jsonl` through a callback (`mlx_lm.lora.run()` drops custom callbacks, so call `train_model` directly).
  4. Saves `best/` from inside the validation callback. mlx-lm validates *before* step N but checkpoints *after* it,
     so its own checkpoints are one step off the scored weights.
- Measured on the M5 Pro (2026-11-14):
  - Speed: ≈13.5 s per iteration (micro-batch 2, ≈3.5k tokens) ≈ 260 processed tokens/s for forward+backward; validation (forward only) ≈ 830 tokens/s.
  - Memory: peak 16.5 GB. Turning gradient checkpointing **off** ran out of GPU memory, so it stays on.
  - mlx-lm's `tokens_per_second` counts *answer* tokens only (because of mask_prompt). Use wall time for cost estimates.
- **Training budget decided by the user: 1 epoch, ≈6 h.** That is 1,500 iterations = 375 updates, with validation on a
  fixed 96-example subset every 250 iterations. 2 epochs (≈13 h) is the first option to revisit if validation loss is still falling.
- mlx-lm semantics to remember: `iters` counts micro-batches, but the LR schedule counts optimizer updates
  (= iters / grad_accumulation_steps). `scale` = α/r. `num_layers: -1` = all layers.
- Runs go to `runs/<run_name>/` and are never overwritten. Weights and data copies are git-ignored; config,
  run_info and metrics are tracked.
- Run history:
  - `qwen3-4b-lora-r16-v1` crashed (Metal OOM) at iteration ≈500, most likely because a second model was loaded on the
    GPU from a notebook.
  - **`qwen3-4b-lora-r16-v2` is the main model:** 4.7 h, peak 21.4 GB, val loss 0.638 → 0.0252, best checkpoint = the
    last one (iteration 1499).
  - **Never run two GPU jobs at once.**
- **Rank ablation (requested by the user, 2026-11-15): r4 vs r16 only.** r8 was dropped so the project can finish the
  same day; its config `configs/qwen3-4b-lora-r8.yaml` is left unused.
  - r4 uses the r16 recipe unchanged except the rank, with α = 2r (scale 2.0 fixed).
  - `scripts/run_rank_ablation.sh` runs: train r4 → r4 test predictions (trained categories) → held-out predictions
    last (r16, base, r4).
  - It is reported only. r16 was fixed before the test predictions, and any re-selection would use validation loss.

## Evaluation
- `src/predict.py` generates predictions (greedy, batch 8, resumable) into `results/eval/<run_name or base-qwen3-4b>/{split}_{kind}.jsonl`.
  Cost: ≈4.6 s per chunk on the trained set.
- `src/eval_metrics.py` holds the pure, tested metrics. `src/eval_report.py` (`TAGS` = base vs r16, `ABLATION` = base/r4/r8/r16)
  does the merging and aggregation.
- `notebooks/04_evaluate.ipynb` only reads files.

## Conventions
- Raw data is git-ignored (~280 MB). Re-download from the Zenodo URL above and verify the md5.
- Treat downloaded files as untrusted data. Run Python that reads them with `python -I`, from scripts kept outside `data/`.
- Local hardware: Apple M5 Pro with 48 GB. Prefer a local-first stack (MLX-LM LoRA). Colab T4 with Unsloth is the fallback.
- Seed everything, pin dependencies, and log all hyperparameters to `results/`.
