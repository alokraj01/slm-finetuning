# Supplier-Contract Clause Extraction with a LoRA-Tuned Qwen3-4B

**Alok Raj · Track B (SLM Fine-Tuning) · Scenario S2 (Gen AI for Enterprise Documents)**

## 1. Problem & Domain

Procurement and legal teams check every supplier contract for the same risk terms: renewal traps, termination rights,
assignment and change of control, liability caps, minimum commitments, exclusivity and audit rights. Today this is
manual reading of long, inconsistently drafted documents. **Task:** the input is a contract excerpt plus the requested
categories with their definitions. The output is JSON `{category: [verbatim quotes]}`, with `[]` when a category is
absent, so every answer can be checked against the source.

**Why fine-tuning, not prompting, few-shot or RAG?**

- **What needs weight adaptation.** The model must learn CUAD's annotation conventions: where a clause starts and ends,
  Cap vs Uncapped Liability, when to answer `[]`. These conventions live in 13.8k lawyer-labelled spans, not in any
  prompt. Given the same prompt and definitions, the zero-shot base model reaches a clause F1 of only 0.42.
- **Few-shot** costs about 1.7k tokens per example, multiplied across 16 categories. **RAG** finds the right
  *document*, but the excerpt is already given here. Contracts are confidential, so a 4B on-prem model is realistic.

## 2. Data Pipeline & Algorithm Decisions

| Step | What and why |
|---|---|
| **Source** | **CUAD v1**: 510 SEC-EDGAR contracts, 41 categories, 13,823 expert spans (CC BY 4.0). *Rejected:* LEDGAR (a classification task, where DeBERTa would beat an SLM), ContractNLI (NDAs only), synthetic data (circular evaluation). |
| **Cleaning, chunking** | Labels come from the JSON character offsets. Whitespace is normalized with an offset map, and a check asserts that every gold quote is an exact substring of its chunk. Chunks are ≤ 6,000 characters with ≥ 800 overlap, cut at paragraph or sentence breaks. A category cut by a chunk boundary is not requested for that chunk, so half a clause is never labelled "absent". |
| **Categories** | **16 trained** (4 header fields and 12 risk clauses). **3 held out**, never requested in training (Insurance, Post-Termination Services, Non-Compete), to test generalization. The other 22 are rare, defined relative to another category, or IP-specific. |
| **Format, split** | Each example requests the positive categories plus random negatives in **shuffled order**, so there is no fixed schema to memorize. 34% of examples are all-empty, so the model learns to abstain. Split **by contract** (seeded, stratified by type): 350 / 80 / 80, giving **2,997 training examples**, 649 validation examples and 822 test chunks. |

| Decision | Choice | Rejected alternatives and why |
|---|---|---|
| **Base model** | **Qwen3-4B-Instruct-2507**: dense, Apache-2.0, no thinking tokens, fits in bf16 on a 48 GB M5 Pro | **Qwen3-8B:** 2× compute; the planned step up. **Qwen3.5:** hybrid attention, with open mlx-lm LoRA bugs. **Llama / Gemma:** more restrictive or gated licences. |
| **PEFT** | **LoRA on a bf16 base** | **Full fine-tuning:** ≈ 64 GB with optimizer states, and overfits 3k examples. **QLoRA:** adds 4-bit noise, and the memory saving is not needed. **IA³:** too little capacity. |
| **r / α / modules** | **r = 16, α = 32**, all 7 linear projections in all 36 layers (33M parameters, 0.8%) | **Attention-only:** QLoRA (Dettmers 2023) shows all-linear LoRA closes the gap to full fine-tuning. **r ≥ 32:** more capacity than 3k examples justify. **r = 4:** run as an ablation. |
| **Training** | Loss on the answer only, AdamW 1e-4 with cosine decay, effective batch 8, **1 epoch** (4.7 h), max sequence 5,632 (measured, never truncated) | **Full-sequence loss:** dominated by copying the contract. **2 epochs:** validation loss was already flat at the end (0.0256 → 0.0252). |
| **Metric** | Span F1, Jaccard ≥ 0.5 (as in the CUAD paper). **Contract level** (merged chunks) is the headline. | **Exact match:** penalizes harmless boundary differences. **ROUGE:** rewards a right quote under the wrong category. |

`src/train_lora.py` fixes four silent mlx-lm failures (truncation, re-sampled validation, dropped callbacks, a
one-step-off best checkpoint) and logs every hyperparameter per run.

## 3. Results & Error Analysis

| Test set: 80 unseen contracts, same prompt | Base (zero-shot) | LoRA r16 | LoRA r4 |
|---|---|---|---|
| **Clause F1, contract level** (headline) | 0.423 | **0.738** | 0.721 |
| Header F1, contract level | 0.529 | 0.832 | **0.837** |
| Clause F1, chunk level | 0.442 | **0.743** | 0.726 |
| **Held-out categories F1** (never trained) | 0.274 | **0.484** | 0.470 |
| Clause precision / recall | 0.33 / 0.59 | 0.76 / 0.72 | 0.75 / 0.69 |
| False alarms (quotes where there is no clause) | 5.9% | **0.8%** | **0.8%** |
| Verbatim quotes · strict JSON | 89.6% · 99.5% | 99.3% · 99.9% | 98.9% · 99.8% |
| Trainable params · best validation loss | – | 33.0M · 0.0252 | 8.3M · 0.0278 |

The largest gains are on judgement-heavy clauses: Uncapped Liability 0.04 → 0.60, Cap On Liability 0.37 → 0.75.
**Rank ablation:** r4 keeps 95% of r16's clause gain with ¼ of the parameters; validation loss agrees, so r16 stays.

**Qualitative example (Minimum Commitment, same prompt).** Base: *"Shipper guarantees that during each Contract Year,
Shipper will meet its Quarterly Volume Commitment"*. Fine-tuned adds *"…or, in the event it fails to do so, shall remit
to MPL the Quarterly Deficiency Payment pursuant to Section 3.5."*, exactly the gold span; the base model drops the
financial consequence a reviewer needs.

**Learned or memorized?** Test contracts are unseen and category subsets are shuffled, so there is no fixed schema to
memorize. The direct test is the 3 held-out categories (same template, CUAD definitions): F1 rises from 0.27 to 0.48.
The gain is precision (0.18 → 0.44; false alarms on 40% → 5% of contracts), not recall (0.59 → 0.54): fine-tuning
taught *when to abstain*; finding unseen clause types still rests on the base model (trained categories: 0.74).

**Where it fails:**

- **Misses, not false alarms.** Recall is lowest where drafting varies most: Exclusivity 0.46, Minimum Commitment 0.51,
  Change Of Control 0.55.
- **Span convention inflates the base model's gap.** `dated as of January 29, 2007` vs gold `January 29, 2007` is
  fine for a lawyer, but a Jaccard miss.
- **CUAD label noise.** Role aliases ("Company") are labelled as Parties in only 37% of contracts.
- **Verbatim but wrong** ("on the date first above written" as Agreement Date passes the verbatim check);
  **chunk-local view** ("as defined in Section 2" is unresolvable from another chunk).

## 4. Production & Limitations

- **Hallucination control (built).** Every quote must be a verbatim substring of the input, otherwise it is dropped
  and flagged (0.7% of fine-tuned quotes vs 10.4% for the base model, F1 unchanged). JSON is schema-checked, `[]`
  lets the model abstain, and flagged outputs go to human review.
- **Catastrophic forgetting (measured; adapter on vs off, same prompts).** Knowledge is intact: on the 192 MMLU
  questions where the adapter answers with a letter, it scores 71% vs the base model's 67%. The *output format* is not:
  multiple-choice prompts (a labelled list plus "answer only") resemble the training prompt, and 46% of them come back
  as JSON, so raw MMLU falls 66% → 40%. GSM8K falls 92% → 85% (p = 0.04), mostly unfinished reasoning; open prompts
  are unaffected (0/8 JSON). **Mitigation:** route other traffic to the base model (the adapter is detachable); next
  run, mix ~10% general instruction data into training.
- **Deployment.** An 8 GB base model plus a 132 MB adapter; vLLM multi-LoRA serves many adapters on one base, and a
  contract's chunks run as one parallel batch (≈ 4.6 s per chunk on a laptop).
- **Limitation.** CUAD is US public-company contracts in English; private supplier paper, scanned PDFs and other
  jurisdictions are untested, so a lawyer-reviewed in-house pilot must precede reliance on 0.74 F1.

*Code: github.com/alokraj01/slm-finetuning · Data: CUAD v1 (Hendrycks et al., 2021), CC BY 4.0*
