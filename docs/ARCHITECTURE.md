# Venus AI — architecture as built

This follows the SIH26038 architecture document section by section and records where the
build differs and why. Section numbers refer to that document.

## §2 System overview

`backend/venus/pipeline.py:screen_image(bytes)` runs Stage 0 → 1 → 2 → 3 → 5 and returns one
result dict; `backend/venus/report.py` writes the PDF and the JSON sidecar. Stage 4 runs
separately, parameterised by the operating point the other stages serve with. Every stage
function takes a dict/array in and returns a dict with an `elapsed_ms`.

Design principles as implemented:

- Referable DR (grade ≥ 2) is the decision; the 0–4 grade is evidence.
- Every stage emits a quality or confidence value; the one that reaches the report as a
  probability (P referable) is calibrated on a split the model never trained on.
- The rule grader is deterministic and its trace names the ICDR criterion that fired.
- No served path depends on a network connection.

## §3 What is new

| claim | where it lives |
|---|---|
| 3.1 the model picks the district's staffing | `stage4_simulate.params_from` reads sens/spec from `operating_point.json`; `sweep()` returns the Pareto front and the four slide numbers; the front end shows them |
| 3.2 two graders that must agree | `stage2_grade.rule_grade` + `fuse`; disagreement ≥ 2 levels, referable-without-lesions and the abstain band flag for review; `/screenings` reports the measured flag rate, which `/simulate` uses once ≥ 10 screenings exist |
| 3.3 attention-agreement score | `stage3_explain.attention_agreement`: share of Grad-CAM mass inside a lesion neighbourhood sized to one CAM cell, with the chance level and lift reported; flags when low and no better than chance on a referable call |
| 3.4 rigour as engineering | `eval/calibrate.py`: patient-disjoint split, Platt scaling, threshold locked at 90 % sensitivity, test half scored once (`config/external_test.lock`), fingerprint verified in `config.operating_point()` on every start |

## §4 Stage 0 — `stage0_gate.py`

- **4.1 Modality check**: EfficientNet-B0 gate, five classes, locked threshold 0.987005 on
  P(fundus). A gate that cannot load rejects rather than passing everything through.
- **4.2 FOV**: Otsu on the red channel → largest component → convex hull (fills the dark
  macula and smooths the jagged edge a vignetted periphery leaves) → crop, pad square, 512.
- **4.3 Quality**: sharpness (Laplacian variance on green inside the FOV), illumination
  uniformity (darkest periphery cell / centre mean on a 5×5 grid), saturated and near-black
  fractions, FOV coverage and circularity. Fixed limits, three labels. *Difference:* the
  learned EyeQ CNN is not in this build; `quality.model` says so.
- **4.4 Enhancement**: illumination normalisation (divide by a normalised-convolution
  Gaussian estimate of the illumination, σ = width/30, rescaled to a mid-grey target) and a
  bilateral filter; only for `usable`; never bleeds outside the FOV. *Difference:* the
  4× Ben Graham high-pass the architecture lists is what the grader's own preprocessing
  uses (Stage 2); applied before the classical lesion detectors it amplified noise into
  false exudates on a dark healthy retina, so Stage 0 keeps colour and local contrast
  faithful and lets each detector apply its own CLAHE.
- **4.5 Output**: label, score, features, `enhanced`, and the operator-facing retake reason
  ("Image too blurry — hold still and refocus").

## §5 Stage 1 — `stage1_segment.py`

All classical (the architecture's no-GPU fallbacks), every output tagged `method`.

| structure | method |
|---|---|
| optic disc | brightest smoothed blob (search up to 10 px from the rim, so a disc at the edge of a clipped field is found) refined by a local bright-region fit |
| fovea | darkest smoothed spot in a 1.8–3.2 DD horizontal annulus, rim excluded, with a centre prior |
| vessels | Frangi vesselness, σ 1–4.5, 88th-percentile threshold, specks removed |
| MA / HE | black-hat of CLAHE green, **median + 8·MAD** threshold, dilated vessels and disc excluded, split by area (< 40 px = MA) and moment-based elongation; HE additionally rejected when > 35 % on the vessel map |
| EX / SE | top-hat of green, median + 4.5·MAD (disc excluded from the statistics), vessel reflexes excluded, elongation ≤ 3, **LAB b\* yellowness ≥ 2.5 over local background** (exudates are yellow lipid, reflexes are white); soft exudates by low boundary edge strength |
| quadrants | hemorrhages per ETDRS quadrant, diagonals through the fovea |
| NV | not segmented; Stage 2 reports P(grade ≥ 4) |

Why robust statistics: the disc and arterial reflexes inflate a mean/σ threshold until the
faint macular exudates that actually decide a grade fall under it. Median/MAD does not care.

## §6 Stage 2 — `stage2_grade.py`

- CNN grader: EfficientNet-B4 at 380×380 with the ordinal head (four sigmoid thresholds).
  Grade = thresholds passed; P(grade ≥ 2) is the referable signal; P(grade ≥ 4) the PDR
  evidence; a five-grade bar is derived from the monotone cumulative. The checkpoint's
  unvalidated disease head is built so the weights load and never read.
  *Difference:* the architecture plans EfficientNet-B3 at 512 trained on 48k images; this
  build serves the B4/380 grader that exists, with the numbers it actually gets.
- Preprocessing is the exact pipeline the grader was validated with (bounding-box crop,
  Ben Graham σ 10, CLAHE green, 0–255 float).
- TTA (4 rotations + flip) is available and off by default on CPU.
- Rule grader: ICDR table as written, with **component thresholds** (`EVIDENCE_FLOOR`) so
  one noisy blob cannot set a grade, and a trace listing what was counted, what fell below
  threshold, and that beading/IRMA are not detected.
- Fusion: CNN primary; flags on ≥ 2-level disagreement, referable-without-lesions, and the
  ±0.05 abstain band; the attention flag joins in `pipeline.py`.
- §6.4 calibration and operating point: `eval/calibrate.py` (see docs/VALIDATION.md).

## §7 Stage 3 — `stage3_explain.py`, `report.py`

Grad-CAM on `top_activation` for all four ordinal heads in one `tf.function` (2.7 s → 80 ms
on CPU), mapped onto the FOV bounding box, clipped to the mask, 40 % JET overlay. Lesion
outlines drawn on the original (MA red, HE dark red, EX yellow, SE white), disc circle and
fovea cross in cyan. Attention agreement as in §3.3. One-page PDF via ReportLab: session,
capture time, quality, three panels, grade evidence, confidence and review flags, tier,
recommendation, model version and calibration fingerprint in the footer. JSON sidecar with
the same fields. Occlusion sensitivity / LIME are not included (cut list item 4).

Measured on this laptop (`backend/config/timing_report.json`): median 3.3 s, p95 4.2 s.

## §8 Stage 4 — `stage4_simulate.py`

A heapq discrete-event model with the architecture's entities and parameters: Poisson
arrivals across PHCs, per-PHC camera/operator pools with daily hours, capture with a retake
loop, AI triage applying the **measured** sensitivity, specificity and flag rate, a priority
queue of K ophthalmologists with daily hours (tele-review 3 min, in-person 20 min),
discharge, and the human-only baseline. Queues persist across days; year-end backlog is a
real output; work started after the horizon is backlog, not utilisation.

Outputs: mean/p95 wait, utilisation, backlog, sent-to-doctor, missed by AI / by reader /
unresulted at year end, programme sensitivity, unnecessary referrals, hours, cost, patients
per doctor-hour. `sweep()` runs cameras (1–3) × doctors (1–8) × ROC points (80–97.5 %
sensitivity, measured on the test half) plus baselines, returns the Pareto front (cost vs
missed) and the slide numbers under two editable constraints: missed ≤ a fraction of
referable cases (default 20 %) and p95 wait ≤ 7 days. Cached at startup (full year, ~80 s).

*Difference:* Simulink/SimEvents is replaced by this Python event loop; the parameter struct
and outputs are the same so the SimEvents model can be built block for block.

## §9 Stage 5 — `stage5_schedule.py`

Tiers P0–P4 exactly as the table, with risk factors raising one step (pregnancy and visual
symptoms) or shortening the routine recall to six months (HbA1c > 9, diabetes > 10 y,
hypertension, insulin) and never lowering. `allocate(queue, slots, now, facilities)` is a
pure function: queue key (overdue first, tier, risk, waiting time), earliest slot at the
nearest facility inside the deadline, bump of the lowest-priority holder whose own deadline
still allows re-slotting (logged), overdue items re-opened from now. SQLite tables
`patients, screenings, facilities, slots, appointments`; consent required and timestamped;
patient identifiers only in `patients`. SMS is a simulated log. No-show re-queues at the
same tier with the original queue time.

## §10–11 Data and validation

See docs/VALIDATION.md. Only the EyePACS frozen set (as the grader's raw predictions) is used
in this build; APTOS/DDR/IDRiD/Messidor-2/EyeQ are the training plan for the next grader.

## §12 Deployment

FastAPI + React, all local. Offline operation is native: SQLite, local reports, no external
calls. The React front end is the same code path the tests exercise (`/screen`).

## §13 Tests

`backend/tests/test_stages.py`: one class per stage plus end to end; properties, not shapes
(mask containment, ordinal decode, the ICDR table, fusion flags, fingerprint refusal,
attention lift, confusion-matrix-derived misses, Pareto monotonicity, tier table, ageing,
bumping, < 30 s).
