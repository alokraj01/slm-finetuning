# Supplier-Contract Clause Extraction with a LoRA-Tuned Qwen3-4B

**Alok Raj · Track B (SLM Fine-Tuning) · Scenario S2 (Gen AI for Enterprise Documents)**

## 1. Problem & Domain

**Problem.** Procurement and legal teams review every supplier contract for the same risk terms: renewal traps, termination
rights, assignment and change of control, liability caps, minimum commitments, exclusivity and audit rights. Today this is
manual reading of long, inconsistently drafted documents.

**Task.** The input is a contract excerpt plus a list of requested categories, each with its definition. The output is
JSON `{category: [verbatim quotes]}`, with `[]` when a category is absent. Every answer is a quote that can be checked
against the source, so a reviewer can audit it in seconds.

**Why Track B (fine-tuning), not prompting, few-shot or RAG?**
- **What needs weight adaptation.** The model must learn the CUAD annotation conventions: where a span starts and ends,
  "Cap On Liability" vs "Uncapped Liability", and when to answer `[]`. These conventions are encoded in 13.8k
  lawyer-labelled spans, not in any prompt we could write.
- **Few-shot** is too expensive. One example costs about 1.7k tokens (one chunk), and covering 16 categories would
  multiply the prompt length and latency several times over.
- **RAG** solves a different problem: finding the right document. Here the excerpt is already given, and retrieval
  does not teach span selection.
- **Deployment fit.** Contracts are confidential, so a 4B model running on-prem is the realistic target.
- **Evidence.** The base model, given the same prompt, is the zero-shot baseline in Section 4.

## 2. Data Pipeline

| Step | What and why |
|---|---|
| **Source** | **CUAD v1** (The Atticus Project, Zenodo 4595826, CC BY 4.0): 510 real commercial contracts from SEC EDGAR, 41 categories, 13,823 expert-annotated spans. Real contracts with expert labels fit S2 directly. **Rejected:** LEDGAR (provision classification, where a DeBERTa classifier would beat an SLM); ContractNLI (NDAs only, 17 fixed hypotheses); synthetic data (circular evaluation). |
| **Labels** | Taken from the SQuAD-style JSON with exact character offsets. `master_clauses.csv` was not used because it merges fragments with `<omitted>`. |
| **Cleaning** | Whitespace normalization (−3.75% of characters) with an offset map, so every gold span stays an exact substring of its chunk. A check asserts this for every training quote. |
| **Chunking** | Chunks of at most 6,000 characters with at least 800 characters of overlap, cut at paragraph, then sentence, then line breaks. Only 2 of 8,730 trained spans never fit whole in a single chunk. If a chunk boundary cuts a span, that category is **not requested** for that chunk (33 cases), so the model is never taught that half a clause means "absent". |
| **Categories** | **16 trained:** 4 header fields (Document Name, Parties, Agreement Date, Effective Date) and 12 risk clauses. **3 held out** and never requested in training (Insurance, Post-Termination Services, Non-Compete); these test generalization. The other 22 are excluded: they are rare (<10% of contracts), defined relative to another category, IP/licensing-specific, or out of scope. |
| **Formatting** | Chat format (system, user, assistant). Each example requests all positive categories plus random negatives (k ~ U[4,16]) in **shuffled order**, so the model must read the definitions rather than memorize a fixed schema. 50% of chunks with no positive category are kept, so the model learns to abstain. |
| **Split** | Split **by contract** (never by chunk), seed 42, stratified by contract type: **350 / 80 / 80** contracts for train / validation / test. The test set stays untouched until the final evaluation, and the notebook asserts that no test contract leaks into train or validation. |
| **Counts** | **2,997 train examples** (1,028 with every category empty) and 649 validation examples. The evaluation sets cover 834 validation and 822 test chunks. The build is deterministic (`train_sha256_16 = 2cedaea7f8f74e2d`). |

## 3. Approach & Algorithm Decisions

| Decision | Choice | Rejected alternatives and why |
|---|---|---|
| **Base model** | **Qwen3-4B-Instruct-2507**: dense, Apache-2.0 licence, no thinking mode (no `<think>` tokens), and fits unquantized in bf16 (8 GB) on a 48 GB Apple M5 Pro | **Qwen3-8B:** about 2× compute (≈12 h per epoch on this hardware); the planned step up if 4B underfits. **Qwen3.5:** hybrid linear attention, with open mlx-lm LoRA bugs at the time. **Llama-3.x / Gemma:** more restrictive licences (Gemma is gated). **Phi-4-mini:** similar size, with no advantage for this task. |
| **PEFT method** | **LoRA on a bf16 base** | **Full fine-tuning:** weights plus AdamW states ≈ 64 GB, which doesn't fit, and it overfits more easily on 3k examples. **QLoRA:** saves memory we don't need, and 4-bit rounding adds noise to the base-vs-fine-tuned comparison. **DoRA:** slower per step, with gains reported mainly at low rank; kept as an ablation. **IA³:** not supported in mlx-lm, and rescaling-only adapters have too little capacity to learn a new output behaviour. |
| **Rank / α / modules** | **r = 16, α = 32** (scale 2.0), dropout 0.05, on **all 7 linear projections** (`q,k,v,o,gate,up,down`) in **all 36 layers** → 33.0M trainable parameters (0.82%) | **r = 8:** may underfit 16 category boundaries. **r = 32/64:** more capacity than 3k examples justify, with more memorization risk. **Attention-only (`q,v`):** QLoRA (Dettmers 2023) found that adapting all linear layers closes the gap to full fine-tuning. |
| **Loss** | `mask_prompt`: loss on the answer tokens only (9.1% of all tokens) | **Loss on the full sequence:** dominated by copying contract text. |
| **Optimization** | AdamW (weight decay 0.01), LR 1e-4, 5% warmup, cosine decay to 1e-5, effective batch 8 (2 × 4 accumulation) | **LR 2e-4:** faster, but more forgetting risk. **Constant LR:** a noisier end of training. |
| **Budget** | **1 epoch** (375 updates). Measured ≈13.5 s/iteration → ≈6 h | **2 epochs (≈13 h):** the first thing to revisit if validation loss is still falling. |
| **Sequence length** | 5,632 tokens, derived from measured lengths (longest example 5,389); the trainer **refuses to truncate** | **Default 2,048:** would silently truncate most examples. |
| **Stack** | MLX-LM 0.32.2, run locally | **Colab T4 + Unsloth:** session limits, and 16 GB of VRAM would force QLoRA. |
| **Metric** | Span P/R/F1 with Jaccard overlap (as in the CUAD paper) at **chunk** level (diagnostic) and **contract** level (headline: chunk outputs merged and deduplicated). Header fields are scored at contract level only. Also JSON validity and verbatim rate. | **Exact match:** penalizes harmless boundary differences. **ROUGE/BLEU:** don't penalize a correct quote under the wrong category. |

**Reproducibility fixes in `src/train_lora.py`**, each a silent failure in stock mlx-lm:
1. Fail fast instead of truncating examples.
2. Score validation on a fixed, seeded subset instead of re-sampling it each time.
3. Log metrics through a callback (`run()` drops custom callbacks).
4. Save `best/` inside the validation callback, because stock checkpoints are one step off the scored weights.

Every hyperparameter, library version, the git commit and the data hashes are written to `runs/<run>/run_info.json`.

## 4. Results & Error Analysis

> **Status:** the final training run (v2) is in progress, so the numbers below are placeholders. An earlier run (v1)
> crashed with an out-of-memory error at iteration ≈500, probably because a second model was loaded on the shared GPU.
> By then, validation loss had dropped from **0.638 → 0.035** at iteration 249.

| Metric (test set, 80 contracts) | Base (zero-shot) | Fine-tuned |
|---|---|---|
| Clause F1, contract level (headline) | TBD | TBD |
| Header F1, contract level | TBD | TBD |
| Clause F1, chunk level | TBD | TBD |
| **Held-out categories F1** (never trained) | TBD | TBD |
| JSON validity / verbatim rate | TBD | TBD |
| False positives on chunks with no gold span | TBD | TBD |

**Learned or memorized?**
1. **Contract-disjoint split:** test contracts were never seen in training.
2. **Held-out categories:** if F1 on Insurance, Post-Termination Services and Non-Compete holds or improves over the
   base model, the model learned to *follow definitions*. If it drops, it overfitted to the 16 trained names.
3. **Shuffled category subsets per example:** there is no fixed output schema to memorize.

**Qualitative comparison (base vs fine-tuned, same prompt):** TBD.

**Known failure modes**
- **Verbatim but wrong.** An early adapter returned `"on the date first above written"` as the Agreement Date. It is a
  real substring, so the hallucination check passes. That check is necessary but not sufficient.
- **CUAD label noise** (labels kept as published, not cleaned):
  - one-off span-edge slips (1 of 521 Document Name labels);
  - offsets point at the first occurrence of a repeated string (90% of 2,218 such cases), which is why header fields
    are scored at contract level;
  - role aliases ("Company") are labelled as Parties in only 37% of contracts, which caps the achievable Parties F1.
- **Rare categories** (≈90–100 positive examples: Notice Period To Terminate Renewal, Uncapped Liability) are the
  most likely to underfit.
- **Chunk-local view.** The model cannot resolve cross-references such as "as defined in Section 2" when the definition
  sits in another chunk.

## 5. Production & Limitations

- **Hallucination mitigation:**
  - every quote must be a verbatim substring of the input, otherwise it is dropped and flagged;
  - output is validated against the JSON schema, with a retry on failure;
  - training includes 34% all-empty examples, so the model learns to abstain;
  - flagged or empty-but-expected results go to human review.
- **Catastrophic forgetting (planned):** compare the base model with the adapter on a small general benchmark subset.
  The adapter is detachable, so general traffic can always fall back to the base model with the adapter off.
- **Deployment:** an 8 GB base model plus a 132 MB adapter.
  - On GPU: vLLM multi-LoRA, so one base model serves several task adapters.
  - On limited hardware: a 4-bit base via mlx-lm or llama.cpp.
  - Chunks are independent, so a contract is processed as one parallel batch.
- **Limitation before real deployment:**
  - CUAD covers US public-company (EDGAR) contracts in English only; private supplier contracts, scanned PDFs (OCR)
    and other jurisdictions are untested;
  - only 16 categories are trained;
  - label noise caps some scores.

  A pilot on in-house contracts with lawyer review is required.

## Reproduce

```bash
uv venv .venv -p 3.12 && uv pip install -r requirements.txt   # pinned versions
# download CUAD_v1.zip from Zenodo 4595826 (md5 c38f490a984420b8a62600db401fafd5) into data/raw/cuad_v1/
python -m src.build_dataset                                    # deterministic dataset build (seed 42)
python -m pytest                                               # 27 pipeline tests
python -m src.train_lora --config configs/qwen3-4b-lora-r16.yaml
```

*Data: CUAD v1, Hendrycks et al., NeurIPS 2021 Datasets & Benchmarks (arXiv:2103.06268), CC BY 4.0.*
