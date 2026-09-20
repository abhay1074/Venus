# Model card — grader_v2

| field | value |
|---|---|
| tag | grader_v2 |
| backbone | EfficientNet-B3 |
| input | 512 |
| head | ordinal, 4 cumulative sigmoids |
| epochs | 15 |
| batch | 8 |
| lr | 0.0001 |
| schedule | AdamW, cosine decay |
| precision | mixed_float16, float32 head |
| xla | True |
| augmentation | dihedral, zoom 0.9-1.0, brightness/contrast/saturation ±20% |
| loss | weighted BCE on cumulative targets, threshold weights [1,2,1,1], pos_weight [2.28, 3.13, 6.0, 6.0] |
| train_n | 39147 |
| val_n | 3403 |
| best_val_referable_auc | 0.9636 |
| weights | /home/asvin_laptop/venus-cache/models/grader_v2.weights.h5 |
| finished_at | 2026-09-20T13:57:55.294069+00:00 |
| total_minutes | 137.7 |

## Training history

```json
[
 {
  "epoch": 11,
  "loss": 0.2509,
  "minutes": 9.0,
  "referable_auc": 0.9623,
  "qwk": 0.8387,
  "exact": 0.8357,
  "within_one": 0.9442,
  "any_dr_auc": 0.9433,
  "pdr_auc": 0.9891
 },
 {
  "epoch": 12,
  "loss": 0.2363,
  "minutes": 8.9,
  "referable_auc": 0.963,
  "qwk": 0.8468,
  "exact": 0.8472,
  "within_one": 0.9486,
  "any_dr_auc": 0.9435,
  "pdr_auc": 0.9895
 },
 {
  "epoch": 13,
  "loss": 0.2257,
  "minutes": 8.8,
  "referable_auc": 0.9616,
  "qwk": 0.838,
  "exact": 0.8437,
  "within_one": 0.9445,
  "any_dr_auc": 0.9433,
  "pdr_auc": 0.9897
 },
 {
  "epoch": 14,
  "loss": 0.2195,
  "minutes": 8.8,
  "referable_auc": 0.9623,
  "qwk": 0.8435,
  "exact": 0.8472,
  "within_one": 0.9462,
  "any_dr_auc": 0.9441,
  "pdr_auc": 0.9894
 },
 {
  "epoch": 15,
  "loss": 0.2187,
  "minutes": 8.7,
  "referable_auc": 0.9623,
  "qwk": 0.8442,
  "exact": 0.8475,
  "within_one": 0.9474,
  "any_dr_auc": 0.9441,
  "pdr_auc": 0.9894
 }
]
```
(last 5 of 15 entries)
