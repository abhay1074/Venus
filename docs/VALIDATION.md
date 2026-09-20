# Validation report — Venus AI, model `venus-dr-1.0.0` (grader `legacy_v1`)

Generated 2026-09-20 11:59 UTC by `backend/eval/write_docs.py` from the JSON artefacts the code wrote when it measured; nothing here is typed by hand. The Validation screen in the app renders the same files.

## Data and splits

Manifests built 2026-09-20: perceptual-hash de-duplication (256-bit DCT hash, Hamming ≤ 10) dropped 290 near-duplicates (0 across datasets). All splits are by patient where a patient id exists.

| manifest | n | datasets | grades 0–4 | referable |
|---|---|---|---|---|
| train | 39,147 | eyepacs 27,939, ddr 7,967, aptos 3,241 | 0: 27222, 1: 2453, 2: 7735, 3: 673, 4: 1064 | 24.2% |
| val | 3,403 | eyepacs 2,428, ddr 693, aptos 282 | 0: 2392, 1: 199, 2: 674, 3: 48, 4: 90 | 23.9% |
| calibration | 2,000 | eyepacs 2,000 | 0: 1546, 1: 118, 2: 293, 3: 24, 4: 19 | 16.8% |
| external_test_ddr | 3,720 | ddr 3,720 | 0: 1843, 1: 189, 2: 1342, 3: 71, 4: 275 | 45.4% |
| heldout_eyepacs_frozen | 2,758 | eyepacs 2,758 | 0: 745, 1: 454, 2: 648, 3: 468, 4: 443 | 56.5% |
| ungradable | 1,142 | ddr 1,142 | 5: 1142 | 100.0% |

**Calibration set:** v1 grader (EfficientNet-B4/380) raw scores on the frozen EyePACS manifest, 1,500 images stratified 300 per grade; split by patient (n = 755, 689 patients). SHA-256 `68e60b36c651d6e24f343226aaddcf45c9e7fb2bef9317b40c6a0f54c0e5f613` — the server refuses to start if the manifest on disk differs.
**External test:** EyePACS test half (within the same frozen set; cross-source relative to the v1 training data) (n = 745, 437 referable).

## Protocol, in order

1. Platt scaling fitted on the calibration set: a = 0.3692, b = 2.2583; ECE 0.3048 → **0.0551**.
2. Referable threshold chosen on the calibration set at 90% sensitivity and locked: **0.3341** (calibration-set specificity 0.575); 85% alternative 0.4649 chosen at the same time. Abstain band ± 0.05.
3. External test scored **again with --force (stated)**, 2,000 bootstrap resamples for 95% CIs.

## Results on the external test

| metric | value | 95% CI | target | met |
|---|---|---|---|---|
| Referable-DR AUC | 0.883 | [0.8569, 0.9045] | > 0.95 | no |
| Sensitivity @ locked threshold | 0.911 | [0.8816, 0.9348] | > 0.9 | yes |
| Specificity @ locked threshold | 0.607 | [0.5544, 0.6575] | > 0.85 | **no** |
| PPV / NPV at 18% prevalence | 0.337 / 0.969 | PPV [0.3086, 0.3689] | PPV > 0.55 | no |
| Expected calibration error | 0.041 | — | < 0.05 | yes |
| Sensitivity on grade 2 alone | 0.782 | — | report | — |
| Confusion at the locked threshold | TP 398 · FN 39 · FP 121 · TN 187 | | | |
| At the 85% alternative | sens 0.858, spec 0.744 | | | |

Referred fraction by true grade: grade 0 39%, grade 1 40%, grade 2 78%, grade 3 95%, grade 4 99%.

## Timing (requirement: < 30 s per image)

50 images, TTA off, AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD (16 threads, no GPU): **median 3.3 s, p95 4.2 s**, max 15.8 s → requirement met at p95. Per stage (median): S0 88 ms, S1 420 ms, S2 132 ms, S3 2319 ms (Grad-CAM 72 ms), report 317 ms.

## District simulation (Stage 4, coupled to the numbers above)

144 full-year runs (cameras × ophthalmologists × operating point). Under the constraints missed ≤ 1192 (20% of 5960 referable cases) and p95 wait ≤ 7.0 days: **5 ophthalmologists with AI (at the 95.0% sensitivity point) vs 7 without, ₹1.31 Cr vs ₹1.72 Cr per year, 1064 vs 932 referable cases missed.**

## Stated plainly

- The raw grader output ranks; Platt scaling on the held-out calibration set is what makes P(referable) reportable as a probability.
- PPV/NPV are recomputed at 18% Indian prevalence because the test sets' prevalence is a sampling artefact.
- A missed target is reported as the achieved number with its CI, not a re-tuned threshold.
- Five-grade metrics are reported for contrast; the referable decision (grade >= 2) is the claim.
- Mild DR (grade 1 vs 0) is not a claim of this system; the referable decision (grade ≥ 2) is what the threshold, the CIs and the simulation describe.
- The external test is a different acquisition source from every training image, but a single one; performance on Indian portable-camera images is unmeasured until such a set exists.
