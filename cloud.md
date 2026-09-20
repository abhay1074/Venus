# cloud.md — work diary

One line per step, newest at the bottom. Whoever (or whatever) picks this up next: read the
last few lines, then `CLAUDE.md` for the state and the conventions.

## 2026-09-19

- Cloned the old MediScan repo for reference; found the trained eye checkpoints only on this laptop (`Downloads\MediScan-main\...\backend\weights`). TF 2.21/Keras 3 needed; made `.venv` with tensorflow-cpu 2.21.
- Decision: rebuild from scratch as **Venus AI**, five stages exactly as the SIH26038 architecture PDF; Python/FastAPI + React reference implementation, MATLAB port alongside.
- Wrote Stage 0 (gate/FOV/quality/enhance), Stage 1 (classical OD/fovea/vessels/lesions), Stage 2 (v1 grader + ICDR rule grader + fusion), Stage 3 (Grad-CAM, overlays, attention agreement, PDF), Stage 4 (district DES + sweep), Stage 5 (tiers, allocator, SQLite), FastAPI, React front end (5 screens), 29 tests.
- Locked the first operating point from the v1 grader's frozen EyePACS scores (patient-disjoint halves, Platt, 90 % sens threshold): sens 0.911 / spec 0.607 / AUC 0.883.
- Tuned the classical lesion detectors on the 8 shipped samples (robust MAD thresholds, yellowness test, moment elongation); compiled Grad-CAM with tf.function (2.7 s → 80 ms).
- Timing on the laptop CPU: median 3.3 s, p95 4.2 s per image. Pushed everything to github.com/abhay1074/Venus.

## 2026-09-20

- User: 9 days left, make it complete (not a demo); datasets are on disk in the MediScan folder; MATLAB will be installed; GPU (RTX 5060, WSL) free.
- Verified WSL GPU works only with the pip-CUDA `LD_LIBRARY_PATH` + PTX cache → `scripts/wsl-gpu.sh`. Found XLA + mixed precision gives 130 img/s for B3@512 (vs 14 fp32).
- Wrote the data plan: `backend/data/sources.py` (APTOS 3,662 / EyePACS 35,126 / DDR 13,673 + 757 lesion masks), `cache_stage0.py` (512-px cache, 52,461 images, 0 errors, ~50 min), `rehash.py` (256-bit pHash; nearest-neighbour distances bimodal → threshold 10), `build_manifests.py` (290 near-dups dropped; train 39,147 / val 3,403 / calibration 2,000 EyePACS patients / **external DDR test 3,720** / held-out EyePACS 2,758 / ungradable 1,142; fingerprints).
- Wrote trainers: `train_grader.py` (EfficientNet-B3@512, ordinal head, XLA+AMP), `train_lesion_unet.py` (DDR masks), `train_quality.py` (EyeQ labels, 12,543 EyePACS images). Shared nets in `backend/venus/nets.py`. Serving now selects grader v2 when `backend/weights/grader_v2.weights.h5` exists; U-Net and quality CNN are optional with classical/handcrafted fallbacks.
- Launched `scripts/train-all.sh` on the GPU (grader 15 epochs → scoring → U-Net 60 epochs → quality 12 epochs). **Epoch 1 val referable-AUC 0.943, QWK 0.79** (v1: 0.887 / 0.46); ~12 min/epoch.
- Wrote the MATLAB implementation: `matlab/+drscreen` (all stages), `matlab/simulink` (MATLAB DES reference, SimEvents builder, parsim sweep, Pareto plot), `matlab/app/DRScreenApp.m`, `matlab/tests` (4 test classes), `matlab/README.md`. Not yet executed anywhere — MATLAB is not installed; installing Octave (winget) for a syntax pass.
- Added CI (`.github/workflows/ci.yml`), Dockerfiles + compose, `scripts/pull-models.sh`, `backend/eval/write_docs.py` (VALIDATION.md + model cards generated from JSON), `export_models.py` (SavedModel/ONNX for MATLAB). Front-end Validation page updated to the new operating-point schema. Pushed.
- Now: waiting for training (grader done ≈ 3 h after 11:40 UTC WSL time), Octave installing in the background; next is pulling the weights, calibrating v2 (`backend.eval.calibrate --tag grader_v2 --model-version venus-dr-2.0.0`), re-timing, re-running the sweep, regenerating docs, pushing.
- Installed GNU Octave 11.3 (winget) as a stand-in parser/runtime: all 30 non-test MATLAB files parse; `matlab/octave_smoke.m` executes gradeRule / fuse / calibrate / tier / simulateDistrict with the MATLAB tests' cases — 0 failures after fixing two MATLAB-invalid idioms Octave caught (indexing into a function-call result; two-argument `round`). Test classes need `matlab.unittest`, so those still wait for MATLAB.
- Dry-ran `train_lesion_unet.py` and `train_quality.py` on the Windows CPU with tiny subsets (via `\wsl$`): both run end to end; fixed a class-count guard in the quality trainer. Grader v2 epoch 3 val AUC 0.957; UI now shows grader tag, lesion method (U-Net/classical) and quality-CNN probabilities; `scripts/finalise-models.sh` written (pull weights → calibrate v2 → export → docs).
