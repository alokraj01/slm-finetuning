#!/bin/zsh
# Rank ablation, r4 vs r16 (r8 dropped for time). The r16 recipe is unchanged except rank=4 (alpha = 2r).
# Order (one GPU job at a time; two jobs at once risk Metal OOM, see run r16-v1):
#   0) wait for any running src.predict job to finish
#   1) train r4 (1 epoch)
#   2) r4 test predictions, trained categories
#   3) held-out predictions last: r16, base, r4
# Re-runnable: finished training is skipped and predictions resume where they stopped. A crashed training run stops
# the script, because train_lora refuses to overwrite a run; rename that run dir to retry it.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R4=qwen3-4b-lora-r4-v1
R16=qwen3-4b-lora-r16-v2

while pgrep -f "src.predict" > /dev/null; do sleep 30; done

if ! grep -q '"finished"' runs/$R4/run_info.json 2>/dev/null; then
  mkdir -p runs/$R4
  echo "$(date +%H:%M) train $R4"
  $PY -m src.train_lora --config configs/qwen3-4b-lora-r4.yaml > runs/$R4/train.log 2>&1
fi

echo "$(date +%H:%M) predict $R4 trained"
$PY -m src.predict --tag $R4 --adapter runs/$R4/best --files eval/test_trained.jsonl

echo "$(date +%H:%M) predict held-out: r16, base, r4"
$PY -m src.predict --tag $R16 --adapter runs/$R16/best --files eval/test_heldout.jsonl
$PY -m src.predict --tag base-qwen3-4b --files eval/test_heldout.jsonl
$PY -m src.predict --tag $R4 --adapter runs/$R4/best --files eval/test_heldout.jsonl
echo "$(date +%H:%M) ALL_DONE"
