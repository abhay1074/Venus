# Validation report — Venus AI, model `venus-dr-2.0.0` (grader `grader_v2`)

Generated 2026-09-20 17:09 UTC by `backend/eval/write_docs.py` from the JSON artefacts the code wrote when it measured; nothing here is typed by hand. The Validation screen in the app renders the same files.

## Data and splits

Manifests built 2026-09-20: perceptual-hash de-duplication (256-bit DCT hash, Hamming ≤ 10) dropped 290 near-duplicates (0 across datasets). All splits are by patient where a patient id exists.

| manifest | n | datasets | grades 0–4 | referable |
|---|---|---|---|---|
| train | 39,147 | eyepacs 27,939, ddr 7,967, aptos 3,241 | 0: 27222, 1: 2453, 2: 7735, 3: 673, 4: 1064 | 24.2% |
| val | 3,403 | eyepacs 2,428, ddr 693, aptos 282 | 0: 2392, 1: 199, 2: 674, 3: 48, 4: 90 | 23.9% |
| calibration | 2,000 | eyepacs 2,000 | 0: 1546, 1: 118, 2: 293, 3: 24, 4: 19 | 16.8% |
| external_test_ddr | 3,720 | ddr 3,720 | 0: 1843, 1: 189, 2: 1342, 3: 71, 4: 275 | 45.4% |
| heldout_eyepacs_frozen | 2,758 | eyepacs 2,758 | 0: 745, 1: 454, 2: 648, 3: 468, 4: 443 | 56.5% |
| ungradable | 1,142 | ddr 1,142 | 5: 1142 | 0.0% |
| external_test_messidor2 | 1,744 | messidor2 1,744 | 0: 1017, 1: 270, 2: 347, 3: 75, 4: 35 | 26.2% |

**Calibration set:** grader_v2 predictions on the patient-disjoint calibration manifest (EyePACS, held out before training) (n = 2000, 1000 patients). SHA-256 `6d9863c704ba4e71d6a522bba135d57640b25724a3c474a15a6635a31c896000` — the server refuses to start if the manifest on disk differs.
**External test:** DDR test split: a different acquisition source from every training image, ungradables removed, never trained on (n = 3720, 1688 referable); SHA-256 `e1833d2342e7e67acb74ccc7000f2efe4ad4243a0f699e1d790851ae875475c5`.

## Protocol, in order

1. Platt scaling fitted on the calibration set: a = 0.7773, b = -0.8825; ECE 0.036 → **0.0082**.
2. Referable threshold chosen on the calibration set at 90% sensitivity and locked: **0.1057** (calibration-set specificity 0.859); 85% alternative 0.1872 chosen at the same time. Abstain band ± 0.05.
3. External test scored **once**, 2,000 bootstrap resamples for 95% CIs.

## Results on the external test

| metric | value | 95% CI | target | met |
|---|---|---|---|---|
| Referable-DR AUC | 0.975 | [0.9698, 0.9793] | > 0.95 | yes |
| Sensitivity @ locked threshold | 0.947 | [0.9362, 0.9571] | > 0.9 | yes |
| Specificity @ locked threshold | 0.889 | [0.8749, 0.9019] | > 0.85 | yes |
| PPV / NPV at 18% prevalence | 0.651 / 0.987 | PPV [0.6244, 0.68] | PPV > 0.55 | yes |
| Expected calibration error | 0.042 | — | < 0.05 | yes |
| Sensitivity on grade 2 alone | 0.933 | — | report | — |
| Confusion at the locked threshold | TP 1598 · FN 90 · FP 226 · TN 1806 | | | |
| At the 85% alternative | sens 0.925, spec 0.914 | | | |

Referred fraction by true grade: grade 0 5%, grade 1 73%, grade 2 93%, grade 3 100%, grade 4 100%.
Five-grade output, for contrast (not a claim): quadratic weighted κ 0.913, exact grade 87.3%, within one grade 95.1%.

## Secondary held-out set (within-source)

EyePACS patients frozen from the earlier grader: held-out patients, same source as part of training (within-source) (n = 2758, 1559 referable): AUC 0.954 [0.9469, 0.9608], sensitivity 0.937, specificity 0.754, ECE 0.068.
Five-grade contrast: QWK 0.802, exact 61.4%.

## Additional external test: external_test_messidor2 (scored once at the locked threshold)

Messidor-2 (ADCIS), Krause et al. 2018 adjudicated ICDR grades; never trained on; a third acquisition source — n = 1744 (457 referable, 874 patients), SHA-256 `a93ab0285bd9f0e6b8d646adfb7fc60c918f44a3871b258a635f1252f98b620b`.

AUC **0.963** [0.9535, 0.9707]; at the locked threshold sensitivity **0.980** [0.9664, 0.9915], specificity **0.645** [0.6188, 0.6709], PPV at 18% 0.377, ECE 0.080. Referred by true grade: grade 0 28%, grade 1 62%, grade 2 97%, grade 3 100%, grade 4 100%. Referred by adjudicated DME: DME 0 47%, DME 1 100%.

On this set's own ROC, 90% sensitivity corresponds to specificity 0.900: the discrimination transfers across sources, the calibration shifts conservatively (the locked threshold over-refers here rather than missing cases). A site-specific calibration set before deployment is the remedy the architecture prescribes, and this is the measurement behind it.

Five-grade contrast: QWK 0.714, exact 64.8%, within one grade 91.6%.

## Grader training

EfficientNet-B3 at 512 px, ordinal, 4 cumulative sigmoids; 15 epochs, batch 8, AdamW, cosine decay, mixed_float16, float32 head, XLA True. Augmentation: dihedral, zoom 0.9-1.0, brightness/contrast/saturation ±20%. Loss: weighted BCE on cumulative targets, threshold weights [1,2,1,1], pos_weight [2.28, 3.13, 6.0, 6.0]. Train n = 39,147, val n = 3,403; best validation referable-AUC 0.9636 (epoch 8); 137.7 min on an RTX 5060 laptop GPU.

## Lesion segmentation (DDR test split, scored once)

| lesion | AUPR | Dice at threshold | threshold (DDR valid, max F1) |
|---|---|---|---|
| MA | 0.079 | 0.166 | 0.53 |
| HE | 0.449 | 0.465 | 0.89 |
| EX | 0.477 | 0.486 | 0.92 |
| SE | 0.262 | 0.308 | 0.47 |

## Image quality classifier

EfficientNet-B0 at 256, 3-class softmax, EyeQ train labels on EyePACS images, split by patient 70/10/20: held-out-patient accuracy 92.8%, ungradable-detection AUC **0.992** (target > 0.95), good-vs-rest AUC 0.994. Out-of-source check on DDR's ungradable class (n = 1142): 99.9% labelled reject, 100.0% not labelled good.

## Human review: flag rate, attention agreement, rule grader (validation sample)

498 grade-stratified validation images through the served path (lesions by unet), 487 gradable (retake rate 2.2%; quality labels {'usable': 464, 'good': 23, 'reject': 11}).

- **Human-review flag rate 25.1%** — reasons: disagreement 83, abstain 53, no_lesion 12, attention 10. CNN referable-error rate among flagged images 23.8% vs 14.8% among unflagged (referable accuracy overall 83.0%).
- **Attention agreement** on referable CNN calls: correct calls median 0.542 (IQR 0.349–0.69, n = 241) vs incorrect calls median 0.273 (IQR 0.156–0.421, n = 69). Where the network's attention sits on the detected lesions, it is more often right: the score is a review signal, not a decoration.
- **Rule grader alone** (ICDR table on the lesion counts): referable sensitivity 91.2%, specificity 49.4%; exact grade 47.0%, within one grade 85.0%; agrees with the CNN within one grade on 83.0% of images. It is a consistency check that a clinician can verify by hand, not a second classifier.

## Timing (requirement: < 30 s per image)

50 images, TTA off, AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD (16 threads, no GPU): **median 4.2 s, p95 5.0 s**, max 5.6 s → requirement met at p95. Per stage (median): S0 189 ms, S1 734 ms, S2 194 ms, S3 2774 ms (Grad-CAM 123 ms), report 301 ms.

## District simulation (Stage 4, coupled to the numbers above)

144 full-year runs (cameras × ophthalmologists × operating point). Under the constraints missed ≤ 1183 (20% of 5917 referable cases) and p95 wait ≤ 7.0 days: **2 ophthalmologists with AI (at the 95.0% sensitivity point) vs 7 without, ₹0.69 Cr vs ₹1.72 Cr per year, 1130 vs 866 referable cases missed.**

## Stated plainly

- The raw grader output ranks; Platt scaling on the held-out calibration set is what makes P(referable) reportable as a probability.
- PPV/NPV are recomputed at 18% Indian prevalence because the test sets' prevalence is a sampling artefact.
- A missed target is reported as the achieved number with its CI, not a re-tuned threshold.
- Five-grade metrics are reported for contrast; the referable decision (grade >= 2) is the claim.
- Mild DR (grade 1 vs 0) is not a claim of this system; the referable decision (grade ≥ 2) is what the threshold, the CIs and the simulation describe.
- The external test is a different acquisition source from every training image, but a single one; performance on Indian portable-camera images is unmeasured until such a set exists.
