# Supplier-Contract Clause Extraction with a LoRA-Tuned Qwen3-4B

**Avathon AI/ML Hiring Challenge · Track B (SLM Fine-Tuning) · Scenario S2 (Gen AI for Enterprise Documents)**

- Write-up: [`write-up/WRITEUP.md`](write-up/WRITEUP.md) (PDF: `write-up/WRITEUP.pdf`)
- Walkthrough video: _link to be added_

## Problem

Procurement and legal teams review every supplier contract for the same risk terms: renewal traps, termination rights,
assignment and change of control, liability caps, minimum commitments, exclusivity and audit rights. This repo fine-tunes
a 4B model to do that first pass.

**Task (category-conditioned extraction).** The input is a contract excerpt plus the requested categories, each with its
CUAD definition. The output is JSON `{category: [verbatim quotes]}`, with `[]` when a category is absent. The model learns
the skill "find the clause that matches this definition", not a fixed schema. Three categories (Insurance,
Post-Termination Services, Non-Compete) are never requested in training and test whether the skill generalises.

## Data

**CUAD v1**, from The Atticus Project. It contains 510 SEC EDGAR contracts, 41 categories and 13,823 lawyer-annotated
spans, released under CC BY 4.0. It is available on Zenodo:
[10.5281/zenodo.4595826](https://zenodo.org/records/4595826).

| | |
|---|---|
| Categories | 16 trained (4 header + 12 clause), 3 held out for testing only |
| Chunking | ≤ 6,000 chars, ≥ 800-char overlap, cut at paragraph or sentence boundaries |
| Split | by contract (never by chunk), seeded and stratified by contract type: 350 / 80 / 80 |
| Samples | 2,997 train · 649 validation · 822 test chunks (each test chunk is scored on trained and on held-out categories) |

The full pipeline and statistics are in [`data/processed/cuad_v1/DATA_CARD.md`](data/processed/cuad_v1/DATA_CARD.md),
which the build script generates.

## Model and training

| | |
|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0) |
| Method | LoRA, r = 16, α = 32, dropout 0.05, all 7 linear projections in all 36 layers (33M trainable params) |
| Training | bf16 weights, 1 epoch = 1,500 iterations, AdamW 1e-4 with a cosine schedule |
| Batch and sequence | effective batch 8, max sequence 5,632 tokens, loss on the answer only |
| Stack | MLX-LM 0.32.2 on an Apple M5 Pro (48 GB) |
| Ablation | r = 4 with the same recipe (α = 2r) |

All hyperparameters are in `configs/*.yaml`. Each run writes `config.yaml`, `run_info.json`, `metrics.jsonl` and
`train.log` to `runs/<run_name>/`. These logs are tracked in git; the adapter weights are not.

## Repository layout

```
src/cuad_data.py         shared categories, normalisation, chunker, prompt builder (used by training AND inference)
src/build_dataset.py     CUAD -> train/valid/eval JSONL + DATA_CARD.md (deterministic, seed 42)
src/train_lora.py        MLX-LM LoRA wrapper: length check, fixed val subset, metrics log, correct best checkpoint
src/predict.py           batched greedy inference, resumable
src/eval_metrics.py      span P/R/F1 (Jaccard >= 0.5, as in the CUAD paper), JSON validity, verbatim check
src/eval_report.py       chunk- and contract-level scoring, base vs fine-tuned, rank ablation
notebooks/01_cuad_eda    data exploration and the design decisions it drove
notebooks/02_train_lora  chat template, sequence length, hyperparameter rationale, memory and time budget
notebooks/03_monitor     live training curves
notebooks/04_evaluate    results, qualitative examples, error analysis, rank ablation (reads files only)
configs/  runs/  results/eval/  tests/  write-up/
```

## Reproduce end-to-end

The pipeline needs Apple Silicon with at least 32 GB of memory. On CUDA the same data and configs work with TRL/PEFT, but
that path is untested.

```bash
# 1. environment (Python 3.12, pinned versions)
uv venv .venv -p 3.12 && uv pip install -r requirements.txt

# 2. data: download CUAD_v1.zip from https://zenodo.org/records/4595826
#    (md5 c38f490a984420b8a62600db401fafd5) and unzip it into data/raw/cuad_v1/
.venv/bin/python -m src.build_dataset          # -> data/processed/cuad_v1/  (stats.json train_sha256_16 = 2cedaea7f8f74e2d)
.venv/bin/python -m pytest                     # 36 tests

# 3. training (~4.7 h for r16 on an M5 Pro; run one GPU job at a time)
.venv/bin/python -m src.train_lora --config configs/qwen3-4b-lora-r16.yaml
.venv/bin/python -m src.train_lora --config configs/qwen3-4b-lora-r4.yaml     # rank ablation

# 4. test predictions (~4.6 s per chunk, 822 chunks per file)
.venv/bin/python -m src.predict --tag base-qwen3-4b --files eval/test_trained.jsonl eval/test_heldout.jsonl
.venv/bin/python -m src.predict --tag qwen3-4b-lora-r16-v2 --adapter runs/qwen3-4b-lora-r16-v2/best \
    --files eval/test_trained.jsonl eval/test_heldout.jsonl

# 5. scores
.venv/bin/python -m src.eval_report --split test   # -> results/eval/summary_test.{json,md}
# then run notebooks/04_evaluate.ipynb (kernel: .venv)

# 6. catastrophic-forgetting check (~15 min): MMLU 285 + GSM8K 100 + 8 open prompts, adapter off vs on
.venv/bin/python -I src/forgetting_check.py --download
.venv/bin/python -I src/forgetting_check.py --tag base
.venv/bin/python -I src/forgetting_check.py --tag r16 --adapter runs/qwen3-4b-lora-r16-v2/best
.venv/bin/python -I src/forgetting_check.py --report base r16      # -> results/forgetting/summary.md
```

Steps 3 and 4 can also be run as one queue: `scripts/run_rank_ablation.sh` runs r4 training and then all remaining
predictions.

## Results

Test set: 80 unseen contracts. The metric is span F1 at Jaccard ≥ 0.5. The headline is contract level: chunk outputs are
merged per contract before scoring.

| | Base Qwen3-4B | LoRA r16 (main) | LoRA r4 (ablation) |
|---|---|---|---|
| Clause F1, contract level | 0.423 | **0.738** | 0.721 |
| Header F1, contract level | 0.529 | 0.832 | **0.837** |
| Clause F1, chunk level | 0.442 | **0.743** | 0.726 |
| **Held-out categories F1** (never trained) | 0.274 | **0.484** | 0.470 |
| False alarms (quotes where there is no clause) | 5.9% | **0.8%** | **0.8%** |
| Strict JSON / verbatim quotes | 99.5% / 89.6% | 99.9% / 99.3% | 99.8% / 98.9% |
| Trainable params / best val loss / train time | – | 33.0M / 0.0252 / 4.7 h | 8.3M / 0.0278 / 5.0 h |

- **Fine-tuning gain:** about +0.31 F1 on trained categories, with 7× fewer false alarms.
- **Learned vs memorized:** on the 3 held-out categories, which were never requested in training, F1 rises
  0.27 → 0.48. The gain comes from precision (false alarms on 40% → 5% of contracts), so the model learned to follow
  definitions and to abstain.
- **Rank:** r4 keeps 95% of r16's gain with a quarter of the parameters, and validation loss ranks the two the same
  way. r16 was fixed before the test runs; the ablation is reported only, not used to pick a model.
- **Forgetting** (`results/forgetting/summary.md`, adapter off vs on, same prompts):
  - Knowledge is intact: on the MMLU questions where the adapter answers with a letter, it scores 71% vs 67% for the
    base model.
  - The *output format* is hijacked: 46% of multiple-choice prompts come back as JSON, so raw MMLU falls 66% → 40%.
  - GSM8K falls 92% → 85%, mostly from unfinished reasoning. Open-ended prompts are unaffected.
  - Mitigation: route non-extraction traffic to the base model (the adapter is detachable), and mix in general
    instruction data on the next run.

Full tables, base-vs-fine-tuned examples and the error analysis are in `notebooks/04_evaluate.ipynb` and the write-up.

## Citation

Hendrycks, D., Burns, C., Chen, A., Ball, S. *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review.*
NeurIPS 2021 Datasets and Benchmarks Track. arXiv:2103.06268. The dataset is released under CC BY 4.0.
