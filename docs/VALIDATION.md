# Validation report — Venus AI, model `venus-dr-2.0.0` (grader `grader_v2`)

Generated from the operating point written 2026-09-20 15:46 UTC by `backend/eval/write_docs.py`, from the JSON artefacts the code wrote when it measured; nothing here is typed by hand. The Validation screen in the app renders the same files.

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

## What a site calibration set buys (Messidor-2, evaluation of a deployment step)

20 random patient-disjoint halves; site Platt (a, b) and the 90 % sensitivity threshold fitted on the calibration half (or a random subset of it of the stated size), evaluated on the other half; the locked operating point evaluated on the same halves. n = 1,744 images, 874 patients, 457 referable.

| operating point | sensitivity | specificity | ECE |
|---|---|---|---|
| locked (EyePACS calibration set), on the held-out halves | 0.981 | 0.644 | 0.079 |
| site calibration on the other half (n ≈ 872) | 0.896 | 0.900 | 0.021 |
| site sample of 100 labelled images | 0.882 [0.793, 0.965] | 0.893 [0.772, 0.968] | 0.039 |
| site sample of 200 labelled images | 0.892 [0.823, 0.949] | 0.894 [0.806, 0.952] | 0.028 |
| site sample of 400 labelled images | 0.892 [0.843, 0.929] | 0.904 [0.858, 0.946] | 0.023 |
| site sample of 800 labelled images | 0.900 [0.868, 0.935] | 0.898 [0.868, 0.929] | 0.021 |

Brackets are the 5th–95th percentile over the repeats. Discrimination transfers across acquisition sources; calibration does not, and re-fitting Platt scaling and the 90 % sensitivity threshold on a few hundred labelled images from the site restores the intended operating point. This is the number behind "a site-specific calibration set is a prerequisite for deployment". The served operating point is unchanged.

## Grader training

EfficientNet-B3 at 512 px, ordinal, 4 cumulative sigmoids; 15 epochs, batch 8, AdamW, cosine decay, mixed_float16, float32 head, XLA True. Augmentation: dihedral, zoom 0.9-1.0, brightness/contrast/saturation ±20%. Loss: weighted BCE on cumulative targets, threshold weights [1,2,1,1], pos_weight [2.28, 3.13, 6.0, 6.0]. Train n = 39,147, val n = 3,403; best validation referable-AUC 0.9636 (epoch 8); 137.7 min on an RTX 5060 laptop GPU.

## Things tried and not shipped: a 1024 px lesion network for microaneurysms

U-Net, 4 levels, 32 base filters, 1024x1024 frames, trained on 512 px lesion-biased crops (4 per image per epoch), 4 sigmoid channels (160 epochs, 88.3 min), DDR test scored once: MA AUPR 0.099, HE AUPR 0.417, EX AUPR 0.475, SE AUPR 0.203 (the served 512 px network: see the table below). Served for MA only on the same 497 raw validation images:

| | 512 px network only (served) | + 1024 px network for MA |
|---|---|---|
| rule grader alone, exact / within one grade | 47.9% / 84.5% | 48.7% / 82.9% |
| attention agreement, median correct / incorrect referable calls | 0.56 / 0.293 | 0.546 / 0.295 |
| review policy: attention floor · flag rate | 0.2 · 25.1% | 0.15 · 25.6% |
| CNN error rate among flagged vs unflagged | 28.8% vs 13.4% | 25.2% vs 14.6% |
| share of the CNN's referable errors flagged | 41.9% | 37.2% |
| CPU time per image, median | 1.7 s | 2.6 s |

On the same 497 raw validation images the 1024 px network changes nothing downstream: the rule grader's exact grade agreement moves 0.479 -> 0.487 and within-one 0.845 -> 0.829; the attention flag gets less precise (error rate among flagged referable calls at the 0.20 cut 0.60 -> 0.42, so no cut reaches the 60 % rule) and the review policy catches 41.9 % -> 37.2 % of the CNN's referable errors; CPU time per image 1.7 s -> 2.6 s. Its gain is confined to pixel MA AUPR on the DDR test split (0.079 -> 0.099). The serving path stays in the code (config/lesion_thresholds_1024.json re-enables it); the 512 px network reads every class.

## Things tried and not shipped: a second seed, test-time augmentation

Does a second training seed (seed 7, same recipe; ensemble = mean of P(grade >= k)) beat grader_v2 alone, and does it add to test-time augmentation? Measured on the calibration and validation sets only (the external tests were not re-scored):

| set | TTA | grader_v2 | grader_v2_s7 | mean ensemble | ensemble − best single (paired bootstrap 95% CI) |
|---|---|---|---|---|---|
| calibration (n = 2,000) | off | 0.947 | 0.943 | 0.950 | +0.0030 [-0.0004, 0.0067] |
| calibration (n = 2,000) | on | 0.949 | 0.943 | 0.949 | -0.0002 [-0.0032, 0.0031] |
| val (n = 3,403) | off | 0.964 | 0.965 | 0.967 | +0.0018 [-0.0002, 0.0039] |

AUC of the referable decision. Averaging two seeds adds about the same as test-time augmentation (+0.003 on calibration, and TTA on top of the ensemble adds nothing); five-grade exact accuracy moves by about one point. That is below what the district numbers would notice and would double the grader's inference cost on a CPU, so the served grader stays a single network without TTA, and the external test's one scoring stands.

## Lesion segmentation (DDR test split, scored once)

| lesion | read by | AUPR | Dice at threshold | threshold (DDR valid, max F1) |
|---|---|---|---|---|
| MA | 512 px network | 0.079 | 0.166 | 0.53 |
| HE | 512 px network | 0.449 | 0.465 | 0.89 |
| EX | 512 px network | 0.477 | 0.486 | 0.92 |
| SE | 512 px network | 0.262 | 0.308 | 0.47 |

## Image quality classifier

EfficientNet-B0 at 256, 3-class softmax, EyeQ train labels on EyePACS images, split by patient 70/10/20: held-out-patient accuracy 92.8%, ungradable-detection AUC **0.992** (target > 0.95), good-vs-rest AUC 0.994. Out-of-source check on DDR's ungradable class (n = 1142): 99.9% labelled reject, 100.0% not labelled good.

## Human review: flag rate, attention agreement, rule grader (validation sample)

498 grade-stratified validation images through the served path (lesions by unet), 497 gradable (retake rate 0.2%; quality labels {'usable': 476, 'good': 21, 'reject': 1}).

- **Human-review flag rate 23.9%** under the architecture's default rules (±0.05 probability band, attention < 0.15 with lift < 1.5) — reasons: disagreement 80, abstain 29, attention 23, no_lesion 12. CNN referable-error rate among flagged images 26.9% vs 14.3% among unflagged (referable accuracy overall 82.7%).
- **Attention agreement** on referable CNN calls: correct calls median 0.56 (IQR 0.37–0.71, n = 249) vs incorrect calls median 0.293 (IQR 0.181–0.4695, n = 74). Where the network's attention sits on the detected lesions, it is more often right: the score is a review signal, not a decoration.
- **Review policy chosen on the same sample** (`config/review_policy.json`, served): abstain band ±0.35 in logit space around the locked threshold, attention floor 0.2 (the highest cut at which ≥ 60 % of the flagged referable calls are CNN errors). Resulting **flag rate 25.1%** (reasons: disagreement 80, no_lesion 12, abstain 29, attention 30); CNN error rate 28.8% among flagged vs 13.4% unflagged; 41.9% of the CNN's referable errors land in the review queue. This is the rate the district simulation uses.
- **Rule grader alone** (ICDR table on the lesion counts): referable sensitivity 92.2%, specificity 50.0%; exact grade 47.9%, within one grade 84.5%; agrees with the CNN within one grade on 83.9% of images. It is a consistency check that a clinician can verify by hand, not a second classifier.

## Timing (requirement: < 30 s per image)

50 images, TTA off, AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD (16 threads, no GPU): **median 1.5 s, p95 1.7 s**, max 2.8 s → requirement met at p95. Per stage (median): S0 176 ms, S1 634 ms, S2 150 ms, S3 261 ms (Grad-CAM 103 ms), report 290 ms.

On the RTX 5060 (WSL): median 3.4 s, p95 4.1 s — inference is not the cost on either machine; Stage 1 landmarks and the overlay encoding are. (Measured before the PNG-encoding change that took the CPU median from 4.2 s to 1.7 s.)

## District simulation (Stage 4, coupled to the numbers above)

144 full-year runs (cameras × ophthalmologists × operating point). Under the constraints missed ≤ 1192 (20% of 5960 referable cases) and p95 wait ≤ 7.0 days: **2 ophthalmologists with AI (at the 95.0% sensitivity point) vs 7 without, ₹0.68 Cr vs ₹1.71 Cr per year, 1108 vs 932 referable cases missed.**

## Stated plainly

- The raw grader output ranks; Platt scaling on the held-out calibration set is what makes P(referable) reportable as a probability.
- PPV/NPV are recomputed at 18% Indian prevalence because the test sets' prevalence is a sampling artefact.
- A missed target is reported as the achieved number with its CI, not a re-tuned threshold.
- Five-grade metrics are reported for contrast; the referable decision (grade >= 2) is the claim.
- Mild DR (grade 1 vs 0) is not a claim of this system; the referable decision (grade ≥ 2) is what the threshold, the CIs and the simulation describe.
- The external test is a different acquisition source from every training image, but a single one; performance on Indian portable-camera images is unmeasured until such a set exists.
