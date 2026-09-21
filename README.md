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

Every number below is read from the JSON the code wrote when it measured; `docs/VALIDATION.md`
is generated from the same files with the CIs, the protocol and the caveats.

| metric (referable DR, ICDR ≥ 2) | value | where |
|---|---|---|
| AUC | **0.975** [0.970, 0.979] | **DDR test split, 3,720 images** — a different acquisition source from every training image, SHA-256-frozen before training, scored **once** with the threshold locked on a 2,000-image patient-disjoint EyePACS calibration set |
| Sensitivity / specificity at the locked threshold | **0.947** [0.936, 0.957] / **0.889** [0.875, 0.902] | same — the problem statement asks > 0.90 / > 0.85 |
| PPV / NPV at 18 % Indian prevalence · ECE | 0.651 / 0.987 · 0.042 | same |
| Messidor-2 (adjudicated labels, third source), scored once at the same threshold | AUC **0.963**, sens **0.980**, spec 0.645 (its own ROC: 0.90 sens at 0.90 spec) | 1,744 images, 874 patients |
| Held-out EyePACS patients (within-source) | AUC 0.954, sens 0.937, spec 0.754 | 2,758 images |
| **What a site calibration set buys** (Messidor-2, patient-disjoint halves, 20 repeats) | locked point: sens 0.981 / spec 0.644 / ECE 0.079 → re-fitted on **400 labelled site images: sens 0.89 [0.84, 0.93] / spec 0.90 [0.86, 0.95] / ECE 0.023** | `backend/eval/site_calibration.py`; served point unchanged |
| Lesion U-Net, pixel AUPR / Dice (DDR test, 225 images, once) | HE 0.45 / 0.47 · EX 0.48 / 0.49 · SE 0.26 / 0.31 · MA 0.08 / 0.17 | thresholds chosen on DDR valid; a 1024 px network lifts MA to 0.10 / 0.23 but changes nothing downstream at +0.9 s per image — measured, not shipped |
| Quality CNN, ungradable-detection AUC | **0.992** (held-out patients); 99.9 % of DDR's ungradable class caught | EyeQ labels |
| Attention agreement, referable calls (median) | **0.54** when the CNN is right vs **0.27** when it is wrong | 487 validation images |
| Human-review flag rate (validation-chosen policy) · retake rate | 25.2 % · 0.2 % | 497 raw validation images through the served path; 28.8 % of flagged calls are CNN errors vs 13.4 % of unflagged |
| End-to-end time, laptop CPU, all three networks | **median 1.5 s, p95 1.7 s** | 50 images |
| District: ophthalmologists for ≥ 80 % programme sensitivity, p95 wait ≤ 7 days | **2 with AI vs 7 without**, ₹0.69 Cr vs ₹1.72 Cr / year | 144 full-year runs |

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
wsl bash scripts/train-night.sh / train-day2.sh              # second round: 1024 px patch U-Net (MA), grader seed 7 (ensemble check)
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
| **Validation** | the pre-registered protocol, metrics with CIs, reliability diagram, ROC points fed to the sweep, subgroup by grade, secondary held-out set, benchmark table, the served review policy and the flag/attention study behind it, the lesion U-Net test numbers, what was measured and not shipped, what is stated plainly |

## Repository

```
backend/venus/      stage0_gate stage1_segment stage2_grade stage3_explain stage4_simulate stage5_schedule
                    nets (shared network definitions) report pipeline (screen_image) config
backend/main.py     FastAPI: /screen /screenings /report /simulate /sweep /intake /appointments /worklist /validation-extras
backend/data/       sources cache_stage0 rehash build_manifests  + manifests/ (committed, fingerprinted)
backend/training/   train_grader train_lesion_unet train_quality
backend/eval/       calibrate (locks the operating point) score_grader score_external flag_rate review_policy
                    compare_graders site_calibration timing export_models write_docs figures (docs/figures from the same JSON)
backend/tests/      one TestCase per stage + end to end (29 tests; weight-dependent ones skip without weights)
frontend/           React + Vite + Tailwind + Recharts
matlab/             +drscreen (all stages), simulink (MATLAB DES + SimEvents builder + parsim sweep), app, tests
samples/            demo images with provenance and licences
scripts/            setup.ps1 serve.ps1 serve-gpu.sh wsl-gpu.sh train-all.sh pull-models.sh finalise-models.sh
                    build-offline-bundle.ps1 preflight.py
docs/               ARCHITECTURE.md (as built, section by section) VALIDATION.md (generated) figures/ (generated)
                    MATLAB_MAPPING.md DEMO.md
cloud.md            the work diary; CLAUDE.md the notes for whoever continues
```

## What this build is honest about

- **Specificity is what it is.** The threshold is locked at 90 % sensitivity on the calibration
  set; whatever specificity the external test returns is reported with its CI and fed to the
  district simulation, which shows what it costs in doctor-hours. Nothing is re-tuned on the test.
- **Two external sources, neither Indian.** DDR (Chinese hospitals) and Messidor-2 (French,
  adjudicated). On Messidor-2 the locked threshold over-refers (specificity 0.65 at sensitivity
  0.98) although its own ROC reaches 0.90/0.90: discrimination transfers, calibration shifts. A
  site-specific calibration set is a prerequisite for deployment, and this is the measurement
  behind that statement — and `backend/eval/site_calibration.py` measures what the prerequisite
  costs: re-fitting on ~400 labelled images from the site restores 0.89 / 0.90 with ECE 0.02.
  Performance on Indian portable-camera images is unmeasured until such a set exists.
- **Neovascularization is a classifier probability** (P(grade ≥ 4)), never a segmentation.
- **Mild DR is not a claim.** The referable decision (grade ≥ 2) is what the threshold, the CIs
  and the simulation describe; five-grade metrics are shown for contrast.
- **The MATLAB port was written without MATLAB.** Every file parses and the pure logic passes its
  cases under GNU Octave (`matlab/octave_smoke.m`): the calibration fingerprint verifies, the
  district DES reproduces Python's sweep numbers within sampling error (doctor hours 2,947 vs
  2,903; programme sensitivity 0.814 vs 0.814), the graders, fusion, review policy and tiers match
  case by case. The toolbox-dependent parts (network import, Grad-CAM, Report Generator, SimEvents
  block parameters, the `DoctorPoolDES` block) are verified on first run in MATLAB, as
  `matlab/README.md` describes.

Screening aid, not a diagnosis. Every image is read by an eye-care professional.
