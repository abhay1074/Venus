# Model card — venus-dr-2.0.0

**Venus AI is a triage aid for referable diabetic retinopathy. A clinician reviews every case. It is not a diagnosis and it is not a cleared medical device.**

Generated from the operating point written 2026-09-20 15:46 UTC by `backend/eval/write_docs.py`. Every number below is read from the JSON the evaluation code wrote; `docs/VALIDATION.md` has the full protocol and the caveats.

## What it is

A EfficientNet-B3 classifier with an ordinal head (four cumulative sigmoids, P(grade >= k) for ICDR 1-4), calibrated by Platt scaling and thresholded at a referable decision (ICDR >= 2), plus a lesion U-Net and an ICDR rule grader whose disagreement with the classifier sends the case to a human. Model version `venus-dr-2.0.0`, grader `grader_v2`, calibration fingerprint `6d9863c704ba4e71...`.

## Intended use

- **Triage of referable diabetic retinopathy** (ICDR >= 2) from a colour fundus photograph, in a district screening programme where a clinician reads every flagged case.
- Ordering a queue: the priority tier and the appointment it books are the product, not a diagnosis.
- Estimating programme staffing from *measured* accuracy (the Stage 4 simulation).
- Research, teaching and demonstration.

## Out-of-scope use

- **Any use without a clinician reading the case.** The system is not a reader.
- Diagnosis, treatment selection, or a claim that an eye is healthy. A non-referable result is *not* a statement that there is no disease.
- Grading mild DR (ICDR 1) as an outcome: the threshold, the confidence intervals and the simulation all describe the referable decision. Five-grade numbers are reported for contrast only.
- Detecting anything other than DR — other retinal disease is neither trained for nor measured.
- Neovascularization as a localised finding: P(grade >= 4) is a classifier probability, never a segmentation.
- Commercial use of the released checkpoints (see `NOTICE`: the training data permits non-commercial academic use only).
- Regulatory submission. This is not a cleared or CE-marked device and has had no clinical trial.

## Measured performance

Referable DR (ICDR >= 2). Each row names the set it was measured on; 95 % CIs are 2,000-resample bootstrap.

| metric | value | 95% CI | measured on |
|---|---|---|---|
| AUC | 0.975 | [0.9698, 0.9793] | DDR test split: a different acquisition source from every training image, ungradables removed, never trained on — n = 3,720, 1688 referable |
| Sensitivity at the locked threshold | 0.947 | [0.9362, 0.9571] | same |
| Specificity at the locked threshold | 0.889 | [0.8749, 0.9019] | same |
| PPV at 18 % prevalence | 0.651 | [0.6244, 0.68] | same |
| NPV at 18 % prevalence | 0.987 | — | same |
| Expected calibration error | 0.042 | — | same |
| AUC (within-source held-out patients) | 0.954 | [0.9469, 0.9608] | EyePACS patients frozen from the earlier grader: held-out patients, same source as part of training (within-source) — n = 2,758 |
| Sensitivity / specificity there | 0.937 / 0.754 | — | same |
| AUC (messidor2, scored once at the same threshold) | 0.963 | [0.9535, 0.9707] | n = 1,744, 874 patients |
| Sensitivity / specificity (messidor2) | 0.980 / 0.645 | — | same |
| End-to-end time per image | 1.5 s median, 1.7 s p95 | — | 50 images, 16 CPU threads, no GPU |

The problem statement's targets are sensitivity > 0.9, specificity > 0.85, AUC > 0.95, ECE <= 0.05; on the primary external test this build meets 4 of 4.

## Component parts, measured separately

Lesion segmentation, DDR test split scored once (pixel AUPR / Dice at the threshold chosen on DDR valid):

| lesion | AUPR | Dice |
|---|---|---|
| MA — microaneurysms | 0.079 | 0.166 |
| HE — hemorrhages | 0.449 | 0.465 |
| EX — hard exudates | 0.477 | 0.486 |
| SE — soft exudates | 0.262 | 0.308 |

Image-quality classifier: ungradable-detection AUC 0.992 on held-out patients. It decides good vs usable (whether to enhance); only the hand-crafted hard limits refuse an image.

Human review: the served policy flags 25.1% of gradable images; 28.8% of flagged calls are classifier errors against 13.4% of unflagged ones, catching 41.9% of them. Attention agreement on referable calls: median 0.56 when the classifier is right vs 0.293 when it is wrong (497 validation images).

## Populations and settings NOT measured

This is the most important section of this card.

- **No Indian data of any kind.** Training, calibration and test sets are aptos, ddr, eyepacs, messidor2 — US, Chinese and French acquisition. The PPV above is recomputed at an assumed 18 % Indian prevalence; that assumption is not validated here.
- **No portable or smartphone-camera images.** Every image is from a tabletop fundus camera. The intended deployment is portable cameras at primary health centres, and that shift is unmeasured.
- **No measured subgroup performance by age, sex, ethnicity or comorbidity.** The datasets do not carry these labels, so no fairness claim is made in either direction.
- **No cataract, small-pupil or media-opacity cohort**, which is what a real screening queue in a district contains, and what the quality gate would have to hold up against.
- **No prospective use.** Every number is retrospective on stored images. Nobody has been screened by this system.
- **No inter-reader comparison.** The system has not been compared against the ophthalmologists it would triage for.

## Known weaknesses

- **Microaneurysm segmentation is weak: AUPR 0.079, Dice 0.166.** They are 1-3 px at the 512 px working resolution. A microaneurysm-only finding is ICDR grade 1, below the referable threshold this system is accepted for, so the weakness sits under the decision rather than inside it — but any claim about lesion-level evidence must be read with this number in view.
- **Calibration does not transfer across sources.** On messidor2 the locked threshold over-refers: specificity 0.645 at sensitivity 0.980, against 0.889 on the primary test, while discrimination holds (AUC 0.963). Ranking transfers; the operating point does not.
  A site calibration set fixes it, and the cost is measured: re-fitting on the site's own labelled images restores sensitivity 0.896 / specificity 0.900 (ECE 0.021), and 100 labelled images are already enough (0.882 / 0.893).
- **A non-referable result is not a negative diagnosis.** At the locked threshold the system misses 5.3% of referable cases on the primary external test.
- **The rule grader does not detect venous beading or IRMA**, so its severe-NPDR criterion is incomplete and it is a consistency check, not a second reader.

## How the operating point was set

- Threshold chosen at 90 % sensitivity on grader_v2 predictions on the patient-disjoint calibration manifest (EyePACS, held out before training) (n = 2,000, 1000 patients), then **locked**: 0.105674.
- Platt scaling fitted on the same set: ECE 0.036 -> 0.008.
- The external test was scored **once** under this model version (`config/external_test.lock`); a second scoring refuses without an explicit override that is then recorded in the output.
- The serving code verifies the calibration manifest's SHA-256 at start-up and refuses to run if it does not match the operating point, so a threshold can never be served against data nobody can vouch for.

## Provenance and licence

- Datasets, their terms and the required citations: `NOTICE`. The checkpoints are released for non-commercial research and education because the training data permits only that.
- Privacy, what is stored and the DPDP gaps: `docs/PRIVACY.md`.
- Full protocol, reliability diagram, subgroup contrast and the experiments that were measured and not shipped: `docs/VALIDATION.md`.

**Venus AI is a triage aid for referable diabetic retinopathy. A clinician reviews every case. It is not a diagnosis and it is not a cleared medical device.**
