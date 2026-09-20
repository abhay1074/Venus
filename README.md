# Venus AI — explainable diabetic-retinopathy screening for district programmes

Smart India Hackathon 2026 · Problem statement **SIH26038** (MathWorks) · Theme MedTech

One fundus photograph enters, one annotated report leaves, and the grader's *measured* accuracy
drives a simulation of a 100,000-patient district programme that books the patient's appointment
in the order their risk demands. Built as a Python/FastAPI + React reference implementation with a
line-for-line MATLAB port (`matlab/`), trained models, a pre-registered validation protocol, and
an offline deployment path.

```
camera / phone ──► Stage 0 ──► Stage 1 ──┐
                   gate +      segment   ├─► Stage 2 ──► Stage 3 ──► report (PDF + JSON)
                   quality     (U-Net)   │   CNN grader   Grad-CAM
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

## What is new, in one sentence

Existing DR AI gives a grade and a heatmap. Venus AI gives a grade, a **second grade a clinician
can check against the ICDR table**, a **number saying whether the network looked at the lesions**,
tells the district **how many ophthalmologists that accuracy buys**, and **books the appointment**
in the order the risk demands.

## Measured, not claimed

All numbers below are read from the JSON the code wrote when it measured (`backend/config/`,
`models/cards/`); `docs/VALIDATION.md` is generated from the same files and has the CIs, the
protocol and the caveats. See it for the current values — the table here is refreshed with it.

| what | how it was measured |
|---|---|
| Referable-DR AUC, sensitivity, specificity, PPV/NPV at 18 % prevalence, ECE | **DDR test split, 3,720 images**, a different acquisition source from every training image, frozen with a SHA-256 before training, scored **once** with the threshold locked on a 2,000-image patient-disjoint EyePACS calibration set |
| Lesion segmentation AUPR / Dice per lesion type | DDR lesion-segmentation test split (225 images), thresholds chosen on the valid split |
| Image-quality classifier: ungradable-detection AUC | EyeQ labels on EyePACS, held-out patients, plus DDR's ungradable class as an out-of-source check |
| Human-review flag rate, attention-agreement for correct vs incorrect calls, rule-grader stand-alone accuracy | 600 stratified validation images through the served path |
| End-to-end time | 50-image run on a laptop CPU (no GPU) |
| District staffing (K doctors with AI vs K′ without, cost, missed cases) | 144 full-year discrete-event runs coupled to the numbers above |

## Quick start (Windows, CPU)

```powershell
# checkpoints (not committed): backend\weights\grader_v2.weights.h5, eye_modality_gate.weights.h5,
#   lesion_unet.weights.h5, quality_cnn.weights.h5  (wsl bash scripts/pull-models.sh after training)
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1     # venv, deps, npm, tests
powershell -ExecutionPolicy Bypass -File scripts\serve.ps1     # API :8000, web :5173
python scripts\preflight.py                                     # everything the demo depends on
```

Deep links: `/?sample=dr_exudates.png&run=1` screens a shipped image on load; `/?page=district`
opens a screen directly. `docker compose up` builds and runs both services. For a PHC laptop,
`scripts\build-offline-bundle.ps1` produces a folder with a wheel cache and `run.bat`: the API
serves the built front end itself, one process, no network.

## Training and validation (WSL, RTX 5060)

```bash
wsl bash scripts/wsl-gpu.sh backend.data.cache_stage0        # Stage 0 once over 52k images -> ~/venus-cache
wsl bash scripts/wsl-gpu.sh backend.data.rehash              # 256-bit hashes + nearest-neighbour distances
wsl bash scripts/wsl-gpu.sh backend.data.build_manifests     # dedup, patient splits, frozen test, fingerprints
wsl bash scripts/train-all.sh                                # grader -> U-Net -> quality -> calibrate -> flag rate -> export -> docs
```

Raw datasets are read from `VENUS_DATA_ROOT` (APTOS 2019, EyePACS 2015, DDR; ~66 GB, public,
not committed). The manifests with their fingerprints are committed.

## The five screens

| screen | what it shows |
|---|---|
| **Screen** | drop / camera / demo image → Stage 0 verdict with the operator-facing retake reason, or the full result: verdict, calibrated P(referable) against the locked threshold and abstain band, five-grade bar, rule-grader trace, lesion counts per type and per ETDRS quadrant (U-Net or classical, stated), attention agreement, overlays (original, enhanced, lesions, vessels, Grad-CAM ×2), per-stage timing, PDF, book |
| **Review queue** | flagged cases first, both grades side by side, why each was flagged, measured flag rate |
| **District** | one-year discrete-event simulation, AI vs human-only, at editable staffing; the cached Pareto sweep (cameras × doctors × operating point) with the four slide numbers under editable constraints |
| **Appointments** | two-minute intake with consent, tier from the result + risk factors, earliest feasible slot at the nearest facility (ageing, bumping), simulated SMS, doctor worklist with outcomes and no-show re-queue |
| **Validation** | the pre-registered protocol, metrics with CIs, reliability diagram, ROC points fed to the sweep, subgroup by grade, secondary held-out set, benchmark table, what is stated plainly |

## Repository

```
backend/venus/      stage0_gate stage1_segment stage2_grade stage3_explain stage4_simulate stage5_schedule
                    nets (shared network definitions) report pipeline (screen_image) config
backend/main.py     FastAPI: /screen /screenings /report /simulate /sweep /intake /appointments /worklist
backend/data/       sources cache_stage0 rehash build_manifests  + manifests/ (committed, fingerprinted)
backend/training/   train_grader train_lesion_unet train_quality
backend/eval/       calibrate (locks the operating point) score_grader flag_rate timing export_models write_docs
backend/tests/      one TestCase per stage + end to end (29 tests; weight-dependent ones skip without weights)
frontend/           React + Vite + Tailwind + Recharts
matlab/             +drscreen (all stages), simulink (MATLAB DES + SimEvents builder + parsim sweep), app, tests
samples/            demo images with provenance and licences
scripts/            setup.ps1 serve.ps1 serve-gpu.sh wsl-gpu.sh train-all.sh pull-models.sh finalise-models.sh
                    build-offline-bundle.ps1 preflight.py
docs/               ARCHITECTURE.md (as built, section by section) VALIDATION.md (generated) MATLAB_MAPPING.md DEMO.md
cloud.md            the work diary; CLAUDE.md the notes for whoever continues
```

## What this build is honest about

- **Specificity is what it is.** The threshold is locked at 90 % sensitivity on the calibration
  set; whatever specificity the external test returns is reported with its CI and fed to the
  district simulation, which shows what it costs in doctor-hours. Nothing is re-tuned on the test.
- **One external source.** The DDR test split is a different acquisition source from all training
  data, but it is one source (Chinese hospitals). Messidor-2 (adjudicated labels) needs a
  registration that had not been granted; the architecture's stated fallback is what is used.
  Performance on Indian portable-camera images is unmeasured until such a set exists.
- **Neovascularization is a classifier probability** (P(grade ≥ 4)), never a segmentation.
- **Mild DR is not a claim.** The referable decision (grade ≥ 2) is what the threshold, the CIs
  and the simulation describe; five-grade metrics are shown for contrast.
- **The MATLAB port was written without MATLAB.** Every file parses and the pure logic passes its
  cases under GNU Octave (`matlab/octave_smoke.m`); the toolbox-dependent parts (network import,
  Grad-CAM, Report Generator, SimEvents block parameters) are verified on first run in MATLAB, as
  `matlab/README.md` describes.

Screening aid, not a diagnosis. Every image is read by an eye-care professional.
