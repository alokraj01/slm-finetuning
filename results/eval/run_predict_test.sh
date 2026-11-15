#!/bin/zsh
# Final test-set generation: fine-tuned first, then base. Resumable (re-run this script to continue).
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
FT=runs/qwen3-4b-lora-r16-v2/best
$PY -m src.predict --tag qwen3-4b-lora-r16-v2 --adapter $FT --files eval/test_trained.jsonl
$PY -m src.predict --tag base-qwen3-4b --files eval/test_trained.jsonl
$PY -m src.predict --tag qwen3-4b-lora-r16-v2 --adapter $FT --files eval/test_heldout.jsonl
$PY -m src.predict --tag base-qwen3-4b --files eval/test_heldout.jsonl
echo ALL_DONE
