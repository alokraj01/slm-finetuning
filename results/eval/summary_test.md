# Evaluation summary: test

## Span F1 (Jaccard >= 0.5), base vs fine-tuned

| level | group | base F1 | FT F1 | Δ F1 | base P / R | FT P / R |
|---|---|---|---|---|---|---|
| contract | header | 0.529 | 0.832 | +0.303 | 0.465 / 0.613 | 0.804 / 0.862 |
| contract | clause | 0.423 | 0.738 | +0.316 | 0.331 / 0.585 | 0.762 / 0.716 |
| contract | heldout | 0.274 | 0.484 | +0.210 | 0.178 / 0.594 | 0.438 / 0.541 |
| chunk | clause | 0.442 | 0.743 | +0.301 | 0.361 / 0.569 | 0.782 / 0.708 |
| chunk | heldout | 0.291 | 0.478 | +0.187 | 0.195 / 0.571 | 0.455 / 0.502 |

## Output health

| model | set | chunks | coverage | strict JSON | JSON (lenient) | verbatim quotes | truncated | gen tokens | s/chunk |
|---|---|---|---|---|---|---|---|---|---|
| base | trained | 822 | 100% | 99.5% | 99.5% | 0.896 | 0.0% | 230 | 4.57 |
| base | heldout | 822 | 100% | 99.8% | 99.8% | 0.846 | 0.0% | 81 | 2.75 |
| ft | trained | 822 | 100% | 99.9% | 99.9% | 0.993 | 0.0% | 159 | 4.67 |
| ft | heldout | 822 | 100% | 99.9% | 99.9% | 0.987 | 0.0% | 40 | 2.39 |
