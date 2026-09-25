# Venus AI — notes for anyone (or any agent) continuing this work

Read `cloud.md` first: it is the running diary, one line per step, newest at the bottom.
Append to it as you work — that is how the next person picks up where you left off.

## What this is

SIH 2026 problem statement SIH26038 (MathWorks): explainable diabetic-retinopathy screening
for district programmes. Five stages, one result struct, one locked and fingerprinted
operating point. `README.md` is the overview; `docs/ARCHITECTURE.md` maps the code to the
architecture document section by section; `docs/VALIDATION.md` is generated, never edited.

## Layout

- `backend/venus/` the pipeline (stage0_gate … stage5_schedule, pipeline, nets, report, config)
- `backend/main.py` FastAPI; `frontend/` React (Vite + Tailwind + Recharts), five screens
- `backend/data/` dataset readers, Stage 0 cache job, hashing, manifest builder
- `backend/training/` grader v2, lesion U-Net, quality CNN; `backend/eval/` calibrate (locks the
  operating point, scores the external test once), score_grader, timing, export_models, write_docs
- `matlab/` the MATLAB implementation (+drscreen, simulink, app, tests) — mirrors the Python
- `scripts/` setup/serve (Windows), wsl-gpu.sh (GPU launcher), train-all.sh (day-1 chain), train-night.sh +
  launch-night.ps1 (day-2 experiments: 1024 px patch U-Net, grader seed 7), pull-models.sh, preflight.py

## Machines and paths (this laptop)

- Windows venv `.venv` (tensorflow-cpu 2.21) serves the API: `scripts\serve.ps1`.
- WSL Ubuntu has the GPU (RTX 5060) and `~/mediscan-env` (TF 2.21). Run anything on the GPU with
  `wsl bash scripts/wsl-gpu.sh <module> [args]`; caches, models and predictions live in
  `~/venus-cache` (Linux filesystem, on purpose). Raw datasets:
  `C:\Users\anilm\Downloads\MediScan-main\MediScan-main\backend\data\raw\eye` (`VENUS_DATA_ROOT`).
- Trained checkpoints are gitignored. `wsl bash scripts/pull-models.sh` copies them from
  `~/venus-cache/models` into `backend/weights` (v1 checkpoints came from the old MediScan repo).
- Lesions are read by the 512 px U-Net. A 1024 px network exists (`backend/weights/lesion_unet_1024.weights.h5`,
  `~/venus-cache/models/`) and the serving path for it is in `stage1_segment` — it is disabled on purpose
  (`config/experiments/lesion_unet_1024.json` says why); a `config/lesion_thresholds_1024.json` re-enables it.

## Rules that are not negotiable

- The external test is scored once per model version (`config/external_test.lock`); the
  threshold is chosen on the calibration set only. Never tune on the test manifests.
- `config/operating_point.json` carries the calibration manifest's SHA-256 and the grader tag;
  the server refuses to start if either does not match. Re-run `backend.eval.calibrate` after
  any model change, then `backend.eval.timing`, then `backend.eval.write_docs`.
- A missed target is reported with its CI, not re-tuned. Numbers in docs come from JSON.
- Site operating points (`calibrate --site`, `scripts/site-calibrate.sh`) live in
  `config/operating_point_site_<name>.json` with their own lock entries; the served file changes only
  with `--activate`. The served `model_version` is read from the operating point, not the constant.
- Experiments train under their own `--tag`; only the default tags write served config
  (`config/lesion_thresholds.json` comes from tag `lesion_unet`). Promote by copying weights +
  thresholds deliberately, then re-run flag_rate → review_policy → write_docs.
- Human-review rules live in `config/review_policy.json` (chosen on validation by
  `backend.eval.review_policy`); stage2/stage3/stage4 read it through `config.review_policy()`.
- Tests: `python -m pytest backend/tests -q` (104; weight-dependent ones skip without weights, and
  `test_docs_consistency.py` fails if a README number disagrees with its JSON — fix the README, not the JSON);
  MATLAB `cd matlab; runTests` (23 incl. CrossCheckTest against `tests/reference/*`, regenerate those
  with `backend.eval.matlab_reference` and the small .mat dumps if the models or Stage 1 change).
- Stage 1 landmarks: the optic disc is found on the un-enhanced frame after flat-fielding, grey/white
  pixels excluded; measured against clinician fovea marks by `backend/eval/landmark_check.py` (MESSIDOR
  annotations from uhu.es, design/held-out halves). Change the detector -> re-run it, then
  flag_rate -> review_policy -> sweep -> timing -> write_docs -> matlab_reference.
- The repo lives in OneDrive: `backend/weights` and `models/export` are pinned (attrib +P) so they are
  never turned into cloud-only placeholders; re-pin after copying new checkpoints.
- Commit messages end with the Co-Authored-By line used in the history; push to
  github.com/abhay1074/Venus `main`.

## Deadline

SIH submission 30 September 2026; planned upload 28 September. PPT exists (user's side);
its numbers must be refreshed from `docs/VALIDATION.md` before submission.
