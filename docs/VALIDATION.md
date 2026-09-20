# Validation report — Venus AI DR grader, model `venus-dr-1.0.0`

Pre-registered protocol (architecture §6.4 and §11), executed by `backend/eval/calibrate.py`
on 2026-09-19. Everything below is read from `backend/config/operating_point.json`; the
Validation screen in the app renders the same file.

## Data

**Frozen external set.** 1,500 EyePACS (Kaggle 2015) images, stratified 300 per ICDR grade,
never used in training the grader (APTOS + ODIR-5K), a different acquisition source, single-
grader ICDR labels. The grader's raw referable scores on these images are the frozen
manifest `backend/data/manifests/eyepacs_frozen_predictions.csv`.

**Split, by patient.** The image name encodes the patient (`22647_left`). Patients were
shuffled with seed 42 and halved: calibration n = 755 images, test n = 745. The split is
`backend/data/manifests/calibration_split.csv`; its SHA-256
`d29385ea352b8a789d1401c47041dbc8f4124cc5043af09ee7bbb13804ca88d3` is written into the
operating point, and `config.operating_point()` refuses to serve if the file on disk does
not hash to it.

## Steps, in order

1. **Calibration.** Platt scaling on logit(raw score), fitted on the calibration half:
   a = 0.3692, b = 2.2583. Expected calibration error on the calibration half
   0.305 (raw) → **0.055** (calibrated); on the test half **0.041**. The raw score ranks
   well but is not a probability (46.9 % of raw scores below 0.01; thresholds for 80–95 %
   sensitivity all between 0.000 and 0.005); calibration is what makes P(referable)
   reportable as a number.
2. **Threshold, locked.** Chosen on the calibration half at 90 % sensitivity:
   **0.3341** on the calibrated probability (calibration-half specificity 0.575). An 85 %
   alternative, 0.4649, was chosen at the same time because a district officer may prefer it.
   Abstain band ± 0.05 around the threshold routes borderline cases to review.
3. **Test half scored once.** `config/external_test.lock` records the model version; a
   second run refuses without `--force`, and a forced re-score is recorded in the output.
4. **Bootstrap.** 2,000 resamples for 95 % CIs on every metric.

## Results on the test half (n = 745, 437 referable)

| metric | value | 95 % CI | target | met |
|---|---|---|---|---|
| Referable-DR AUC | 0.883 | [0.857, 0.905] | > 0.95 | no |
| Sensitivity @ 0.3341 | 0.911 | [0.882, 0.935] | > 0.90 | yes |
| Specificity @ 0.3341 | 0.607 | [0.554, 0.658] | > 0.85 | **no** |
| PPV at 18 % prevalence | 0.337 | [0.309, 0.369] | > 0.55 | no |
| NPV at 18 % prevalence | 0.969 | — | — | — |
| ECE | 0.041 | — | < 0.05 | yes |
| Sensitivity on grade 2 alone (hardest referable class) | 0.782 | — | report | — |
| Confusion at the locked threshold | TP 398 · FN 39 · FP 121 · TN 187 | | | |
| At the 85 % alternative (0.4649) | sens 0.858, spec 0.744, PPV 0.423 | | | |

Referred fraction by true grade (test half): grade 0 39 %, 1 40 %, 2 78 %, 3 95 %, 4 99 %.

## What we say about the missed targets

The report shows the achieved number with its CI and the gap, not a re-tuned threshold.
Specificity of 0.61 at 90 % sensitivity means roughly 4 in 10 non-referable patients are
sent for a human read; the district simulation (Stage 4) turns that into doctor-hours and
cost, and the sweep shows what a higher operating point (85 % or 95 % sensitivity) buys.
Published 90 % / 98 % (Gulshan et al. 2016, JAMA) used about 128,000 adjudicated images;
with public data and a 48k-image grader the realistic range is 90 % sensitivity at
60–88 % specificity. The planned grader (EfficientNet-B3 at 512 on APTOS + EyePACS + DDR,
Messidor-2 as the frozen test) is the next step, and this protocol re-runs unchanged on it.

## Limitations stated plainly

- One external source (EyePACS, US, single-grader labels). No measured performance on any
  Indian population or portable camera; the adjudicated Messidor-2 set is the planned test.
- The frozen set is 60 % referable by construction; PPV/NPV are recomputed at 18 %.
- The grader's five-grade output is not validated for grading (QWK 0.46 on the full set);
  the grade is shown as evidence and the referable decision is the claim.
- Mild DR (grade 1 vs 0) is not detectable by this grader (AUC 0.66 within-source) and is
  not a claim of the system.
- The classical lesion detectors are not validated against pixel labels (IDRiD AUPR is the
  planned measurement); they are evidence and a consistency check, not a detector claim.
- The modality gate was scored once on 600 held-out fundus images (false rejection 0.67 %,
  Wilson CI [0.26 %, 1.70 %]); about 1 in 150 genuine fundus images is turned away.

## Timing (problem-statement requirement: < 30 s per image)

`python -m backend.eval.timing --images 50` on this laptop CPU (AMD, 16 threads, no GPU),
TTA off: **median 3.3 s, p95 4.2 s**, max 15.8 s (first-request graph trace, now done at
start-up). Per stage, median: Stage 0 ≈ 0.1 s, Stage 1 ≈ 0.45 s, Stage 2 ≈ 0.15 s,
Stage 3 ≈ 2.3 s (Grad-CAM 0.07 s; overlays and PNG encoding of six panels the rest), report ≈ 0.3 s.
