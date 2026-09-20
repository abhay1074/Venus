# Venus AI — explainable diabetic-retinopathy screening for district programmes

Smart India Hackathon 2026 · Problem statement **SIH26038** (MathWorks) · Theme MedTech

One fundus photograph enters, one annotated report leaves, and the grader's measured
accuracy drives a simulation of a 100,000-patient district programme that books the
patient's appointment in the order their risk demands.

```
camera / phone ──► Stage 0 ──► Stage 1 ──┐
                   gate +      segment   ├─► Stage 2 ──► Stage 3 ──► report (PDF + JSON)
                   quality               │   CNN grader   Grad-CAM
                     │                   │   rule grader  lesion overlay
                     ▼                   │   fusion       attention agreement
                  retake / refuse        │      │
                                         │      ├──► human review queue
                                         │      ▼
                                   Stage 4 district simulation ◄── measured sens / spec / flag rate
                                         │
                                         ▼
                                   Stage 5 priority tier ──► appointment ──► doctor worklist
```

Every stage is one Python module with a typed dict in and out, so each can be tested
and timed alone (`backend/venus/stage*.py`). The layout mirrors the MATLAB `+drscreen`
package in the architecture document one to one; see [docs/MATLAB_MAPPING.md](docs/MATLAB_MAPPING.md).

## What is new, in one sentence

Existing DR AI gives a grade and a heatmap. Venus AI gives a grade, a **second grade a
clinician can check against the ICDR table**, a **number saying whether the network looked
at the lesions**, tells the district **how many ophthalmologists that accuracy buys**, and
**books the appointment** in the order the risk demands.

## Measured, not claimed

| metric (referable DR, ICDR ≥ 2) | value | where |
|---|---|---|
| AUC | **0.883** [0.857, 0.905] | EyePACS test half, n = 745, cross-source, scored once |
| Sensitivity at the locked threshold | **0.911** [0.882, 0.935] | same |
| Specificity at the locked threshold | **0.607** [0.554, 0.658] — target 0.85 missed, reported with CI | same |
| Expected calibration error | 0.305 raw → **0.055** calibrated (test half 0.041) | Platt scaling on the calibration half |
| PPV / NPV at 18 % Indian prevalence | 0.337 / 0.969 | derived |
| End-to-end time, laptop CPU | **median 3.3 s, p95 4.2 s** (50 images) | `backend/config/timing_report.json` |
| District: ophthalmologists for ≥ 80 % programme sensitivity | **5 with AI vs 7 without**, ₹1.31 Cr vs ₹1.72 Cr / year | full-year sweep, 144 runs |

The threshold was chosen on a patient-disjoint calibration half at 90 % sensitivity and
locked; the test half was scored exactly once; `config/operating_point.json` carries the
calibration split's SHA-256 and the server refuses to start if the split on disk does not
match. A missed target is reported with its confidence interval, not a re-tuned threshold.
Full protocol: [docs/VALIDATION.md](docs/VALIDATION.md).

## Quick start (Windows, CPU)

```powershell
# 1. trained checkpoints (not committed): copy into backend\weights\
#      eye_best.weights.h5             EfficientNet-B4 ordinal DR grader, TF 2.21 / Keras 3
#      eye_modality_gate.weights.h5    EfficientNet-B0 fundus-vs-other gate
# 2. one-time setup: venv, deps, npm, calibration, tests
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
# 3. run
powershell -ExecutionPolicy Bypass -File scripts\serve.ps1
```

API at `http://127.0.0.1:8000` (OpenAPI at `/docs`), front end at `http://127.0.0.1:5173`.
Deep links: `/?sample=dr_exudates.png&run=1` screens a shipped image on load;
`/?page=district` opens a screen directly. `python scripts/preflight.py` checks everything
the demo depends on. GPU serving from WSL: `wsl bash scripts/serve-gpu.sh`.

The whole system runs offline: models, SQLite records and PDF reports are local. A PHC
laptop needs no connectivity to screen, tier and book.

## The five screens

| screen | what it shows |
|---|---|
| **Screen** | drop / camera / demo image → Stage 0 verdict with the operator-facing retake reason, or the full result: verdict, calibrated P(referable) against the locked threshold and abstain band, five-grade bar, rule-grader trace, lesion counts per type and per ETDRS quadrant, attention agreement, overlays (original, enhanced, lesions, vessels, Grad-CAM ×2), per-stage timing, PDF, book |
| **Review queue** | flagged cases first, both grades side by side, why each was flagged, measured flag rate (the workload the simulation absorbs) |
| **District** | one-year discrete-event simulation, AI vs human-only, at editable staffing; the cached Pareto sweep (cameras × doctors × operating point) with the four slide numbers under editable constraints |
| **Appointments** | two-minute intake with consent, tier computed from the result + risk factors, earliest feasible slot at the nearest facility (with ageing and bumping), simulated SMS, doctor worklist with outcomes and no-show re-queue |
| **Validation** | the pre-registered protocol, metrics with CIs, reliability diagram, ROC points fed to the sweep, subgroup by grade, benchmark table against published figures, and what is stated plainly |

## Repository

```
backend/
  venus/            the pipeline: stage0_gate  stage1_segment  stage2_grade  stage3_explain
                    stage4_simulate  stage5_schedule  report  pipeline (screen_image)  config
  main.py           FastAPI: /screen /screenings /report /simulate /sweep /intake /appointments /worklist
  eval/             calibrate.py (locks the operating point)  timing.py (50-image CPU run)
  config/           operating_point.json  external_test.lock  (sweep_cache, timing_report generated)
  data/manifests/   frozen EyePACS predictions + the patient-disjoint calibration split (fingerprinted)
  tests/            one TestCase per stage + end-to-end timing; 29 tests
  weights/          checkpoints (copied in, not committed)
frontend/           React + Vite + Tailwind + Recharts, five screens
samples/            demo images with provenance and licences (samples/README.md)
scripts/            setup.ps1  serve.ps1  serve-gpu.sh  preflight.py
docs/               ARCHITECTURE.md  VALIDATION.md  MATLAB_MAPPING.md  DEMO.md  provenance/
```

## What this build is honest about

- **Lesion segmentation is classical** (Frangi vessels, morphological top-hat / black-hat
  with robust thresholds, colour and shape tests). No pixel-level lesion network is shipped,
  because the only public pixel-labelled DR set has 81 images and a network trained on it
  could not be validated here. The rule grade it feeds is a consistency check on the CNN;
  disagreement routes to a human rather than deciding anything alone.
- **Neovascularization is a classifier probability**, P(grade ≥ 4) from the grader, and the
  report says "classifier, not localised".
- **The quality label is handcrafted** (sharpness, illumination uniformity, exposure, FOV
  coverage and circularity against fixed limits); the learned EyeQ classifier is not in this
  build and the result says so.
- **Specificity is 0.61, not 0.85.** With public data and a 48k-image grader this is the
  realistic range; the district simulation shows what that specificity costs in doctor time
  and what a better operating point would buy.
- **Mild DR is not a claim.** The referable decision (grade ≥ 2) is what the threshold, the
  CIs and the simulation describe.
- The problem statement asks for a MATLAB core. This is the Python reference implementation
  of the same pipeline; the MATLAB mapping is documented block for block.

Screening aid, not a diagnosis. Every image is read by an eye-care professional.
