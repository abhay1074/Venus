# Venus AI — MATLAB implementation

The pipeline the problem statement asks for, in MATLAB, with the same stage boundaries, struct
fields, configuration files and numbers as the Python reference under `backend/`. Written
against R2024b; the models are the ones trained here (Python/TensorFlow on the GPU) and
imported with `importNetworkFromTensorFlow`.

```
matlab/
  +drscreen/       gate.m normaliseFov.m qualityFeatures.m qualityLabel.m enhance.m   (Stage 0)
                   segment.m                                                          (Stage 1)
                   gradeCNN.m gradeRule.m fuse.m calibrate.m operatingPoint.m         (Stage 2)
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

`buildDistrictModel.m` builds the SimEvents diagram with `add_block`/`set_param`. Block
parameter names differ between SimEvents releases; every `set_param` is wrapped so a renamed
parameter prints one line (`set_param(<block>, '<name>') failed`) instead of aborting.
Fix the reported names against the block dialog (Entity Generator → Event actions →
Generate; Entity Output Switch → Switching criterion "From attribute"). The block model does
not include the doctors' daily hour budget; `simulateDistrict.m` does, and `runSweep` uses it
by default (`useSimulink=true` switches to `parsim` over `Simulink.SimulationInput`).

`drscreen.loadModels` imports the SavedModel folders and caches each as `.mat`. If a layer
is unsupported by `importNetworkFromTensorFlow` in your release, the `.onnx` export of the same
model is tried next (`importNetworkFromONNX`); EfficientNet and the U-Net use only standard
layers (conv, depthwise conv, batch norm, swish, squeeze-excite multiply, transposed conv).

## Correspondence with the Python reference

| MATLAB | Python | shared artefact |
|---|---|---|
| `drscreen.operatingPoint` | `venus.config.operating_point` | `backend/config/operating_point.json` + calibration manifest fingerprint |
| `drscreen.gradeRule` / `fuse` / `tier` / `allocate` | `stage2_grade.rule_grade` / `fuse`, `stage5_schedule.priority_tier` / `allocate` | same thresholds and tier table |
| `simulateDistrict`, `runSweep` | `stage4_simulate.simulate`, `sweep` | same parameter struct; `sweep_result.json` has the shape of `backend/config/sweep_cache.json` |
| `drscreen.segment` (unet path) | `stage1_segment.unet_lesions` | `backend/config/lesion_thresholds.json` |
| `drscreen.report` | `venus.report` | one-page PDF with the same panels and footer |
