# Venus AI — MATLAB implementation

The pipeline the problem statement asks for, in MATLAB, with the same stage boundaries, struct
fields, configuration files and numbers as the Python reference under `backend/`. Written
against R2024b; the models are the ones trained here (Python/TensorFlow on the GPU) and
imported with `importNetworkFromTensorFlow`.

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

## What is verified and what to expect on first run

Everything under `+drscreen`, `simulink/simulateDistrict.m`, `runSweep.m`, `paretoPlot.m`,
`tests/` and `app/` is plain MATLAB written against documented APIs, with unit tests that
mirror the Python ones (same synthetic cases, same expected values). They were written on a
machine without MATLAB, so the first run is the first execution: run `runTests` first and
fix anything it reports — the logic is a line-for-line port of code that passes its 29
Python tests.

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

`drscreen.loadModels` imports the SavedModel folders and caches each as `.mat`. If a layer
is unsupported by `importNetworkFromTensorFlow` in your release, the `.onnx` export of the same
model is tried next (`importNetworkFromONNX`); EfficientNet and the U-Net use only standard
layers (conv, depthwise conv, batch norm, swish, squeeze-excite multiply, transposed conv).
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
