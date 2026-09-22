# Venus AI — MATLAB implementation

The pipeline the problem statement asks for, in MATLAB, with the same stage boundaries, struct
fields, configuration files and numbers as the Python reference under `backend/`. Verified in
**MATLAB R2026a on Windows** (see "What is verified" below); the models are the ones trained
here (Python/TensorFlow on the GPU), imported with `importNetworkFromTensorFlow` except the
lesion U-Net, which is built natively from its Keras checkpoint.

```
matlab/
  +drscreen/       gate.m normaliseFov.m qualityFeatures.m qualityLabel.m enhance.m   (Stage 0)
                   segment.m                                                          (Stage 1)
                   gradeCNN.m gradeRule.m fuse.m calibrate.m operatingPoint.m reviewPolicy.m  (Stage 2)
                   explain.m report.m                                                 (Stage 3)
                   tier.m tierTable.m allocate.m                                      (Stage 5)
                   screenImage.m loadModels.m  + helpers
  simulink/        districtParams.m simulateDistrict.m (MATLAB DES reference)
                   buildDistrictModel.m (writes district_screening.slx, SimEvents)
                   runSweep.m (parsim / parfor sweep, Pareto front, slide numbers) paretoPlot.m
  app/             DRScreenApp.m (uifigure app: Capture, Result, Review queue, District)
  tests/           GradeTest ScheduleTest SimulateTest GateSegmentTest (matlab.unittest)
  runTests.m       runs them all
```

## Set-up (once MATLAB is installed)

Toolboxes: Image Processing, Computer Vision, Deep Learning (+ the TensorFlow/ONNX
converter support package), Statistics and Machine Learning, Simulink, SimEvents, MATLAB
Report Generator; Parallel Computing is optional (parsim/parfor in the sweep).

1. Export the trained networks from the Python side (needs the checkpoints in `backend/weights`):
   `wsl bash scripts/wsl-gpu.sh backend.eval.export_models --onnx` → `models/export/`.
2. In MATLAB, from the repository root:
   ```matlab
   addpath('matlab'); addpath('matlab/simulink'); addpath('matlab/app');
   runTests                                  % logic tests need no networks
   models = drscreen.loadModels();           % imports + caches the networks
   r = drscreen.screenImage('samples/dr_exudates.png', models);   % one image -> report
   app = DRScreenApp(models);                % the four-screen app
   buildDistrictModel(districtParams());     % writes simulink/district_screening.slx
   sweep = runSweep(); paretoPlot(sweep);    % the Pareto front and the four slide numbers
   ```

## What is verified, and what is not

**Verified by running it in MATLAB R2026a on Windows** (Simulink, SimEvents, Deep Learning,
Image Processing, Computer Vision, Statistics and Machine Learning, MATLAB Report Generator,
Parallel Computing, and the *Deep Learning Toolbox Converter for TensorFlow Models* support
package):

- `runTests` — **22 of 22 test cases pass**, including `tests/CrossCheckTest.m`, which checks
  this port against the Python reference rather than against itself.
- `drscreen.loadModels`, `drscreen.screenImage` end to end on every shipped sample, with the
  Grad-CAM overlays and the Report Generator PDF.
- `buildDistrictModel` → `district_screening.slx`, simulated for a full year, and
  `runSweep(struct('useSimulink', true))` through `parsim`.
- `app/DRScreenApp.m` driven programmatically: screen an image, switch overlays, run a
  district year, export the tabs (`docs/figures/matlab_app_*.png`).

**Not verified, and honest about it:**

- Only R2026a, only Windows. Every `set_param` in `buildDistrictModel` is wrapped so a block
  parameter renamed in another release prints one line instead of aborting the build.
- The `struct('plainServer', true)` variant of `buildDistrictModel` (plain Entity Server, no
  daily hour budget) is kept for comparison but has not been built since the R2026a fixes.
- `runSweep(struct('useSimulink', true))` was run over a reduced grid (12 configurations), not
  the full 144-run sweep; the default `useSimulink=false` path runs the full sweep.
- The app's interactive callbacks that open OS dialogs or external viewers — `uigetfile` in
  `pickImage`, `openReport`, `openSimulink` — were not exercised; the functions they call were.

`buildDistrictModel.m` builds the SimEvents diagram with `add_block`/`set_param` — **verified
in R2026a** (library `sldelib`; the entity type is a bus object `PatientBus` recreated by the
model's PreLoadFcn from `districtBus.m`; the generator's Generate action comes from
`generateAction.m`; an Entity Output Switch reads `route` / `outcome` as the port index; the
terminators' arrival counters are logged by To Workspace blocks). Every `set_param` is still
wrapped so a future rename prints one line instead of aborting. The doctors are a MATLAB
Discrete-Event System block running `DoctorPoolDES.m` (K slots with the daily hour budget of
`simulateDistrict.m`: a case that would push a doctor past the budget waits for the next working
day) behind an Entity Queue with priority on `aiPositive`; `buildDistrictModel(p, name,
struct('plainServer', true))` builds the plain Entity Server variant for comparison.

Measured: a full simulated year takes 16 s in SimEvents; at 2 ophthalmologists it misses 255
referable cases at the AI against 254 in the MATLAB DES and 18,156 vs 18,347 cases resulted.
`runSweep(struct('useSimulink', true))` runs every configuration through `parsim` (12 runs in
141 s on 8 workers) and keeps the SimEvents counters beside the DES numbers in
`runs(i).simulink`; the default `useSimulink=false` path is the MATLAB DES alone.

Under GNU Octave (`octave_smoke.m`) the port also verifies the calibration fingerprint
(`sha256File` uses Octave's `hash`), reads the served operating point and review policy into
`districtParams`, and cross-checks `simulateDistrict` against Python's cached sweep at the same
parameters (2 ophthalmologists with AI: doctor hours 2,947 vs 2,903, programme sensitivity
0.814 vs 0.814 — different RNG streams, same model).

`drscreen.loadModels` — **verified in R2026a** with the *Deep Learning Toolbox Converter for
TensorFlow Models* support package — imports the grader, gate, quality CNN and the grader's
feature model (`grader_v2_features`, inputs → `top_activation`) from `models/export`, caching each
as `.mat` next to the export (first import ~70 s, cached load ~9 s; the importer's generated
`+<model>` packages land there too). A Keras 3 SavedModel arrives as one opaque call layer that
takes an unformatted batch-first array: `drscreen.predictNet` handles that. The U-Net is not
imported — the importer has no transposed convolution — but built natively by
`drscreen.buildUnet` from the Keras `.h5` checkpoint. Grad-CAM is the closed form for a
GAP + sigmoid head, `ReLU(Σ_c w_kc A_c)` on the feature maps with the dense weights from
`grader_v2_head.json`. `tests/CrossCheckTest.m` proves all of it against Python on identical
inputs: gate / quality / grader outputs within 2e-6, U-Net within 1e-5, Grad-CAM maps within
2e-5, and end-to-end decisions on every shipped sample (accepted, referable, review flag, tier,
CNN grade; P(referable) within 0.05 — the tolerance image primitives such as `imresize` and the
bilateral filter leave). A full screen with the PDF report takes ~10 s on the CPU. The desktop app (`app/DRScreenApp.m`: Capture, Result, Review queue, District tabs) runs on the same functions — `docs/figures/matlab_app_result.png` and `matlab_app_district.png` are its exported screens.
If `models/export/lesion_unet_1024` and a `config/lesion_thresholds_1024.json` exist, the 1024 px
U-Net reads the classes that file lists from a fresh FOV normalisation of the uploaded image, exactly
as the Python path does; the shipped configuration has no such file (the experiment is recorded in
`config/experiments/lesion_unet_1024.json`), so the 512 px network reads every class.

## Correspondence with the Python reference

| MATLAB | Python | shared artefact |
|---|---|---|
| `drscreen.operatingPoint` | `venus.config.operating_point` | `backend/config/operating_point.json` + calibration manifest fingerprint |
| `drscreen.gradeRule` / `fuse` / `tier` / `allocate` | `stage2_grade.rule_grade` / `fuse`, `stage5_schedule.priority_tier` / `allocate` | same thresholds and tier table |
| `simulateDistrict`, `runSweep` | `stage4_simulate.simulate`, `sweep` | same parameter struct; `sweep_result.json` has the shape of `backend/config/sweep_cache.json` |
| `drscreen.segment` (unet path) | `stage1_segment.unet_lesions` | `backend/config/lesion_thresholds.json` |
| `drscreen.report` | `venus.report` | one-page PDF with the same panels and footer |
