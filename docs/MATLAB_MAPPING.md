# MATLAB mapping

The problem statement requires a MATLAB-based pipeline. Venus AI is the Python reference
implementation of the architecture's `+drscreen` package; each module below maps to one
MATLAB file and names the toolbox functions that implement the same step, so the MATLAB
build is a translation, not a redesign. Struct field names are kept identical so the App
Designer screens and the Simulink model read the same fields.

| Python module | MATLAB file | key toolbox functions |
|---|---|---|
| `venus/stage0_gate.py` — `modality_check`, `fov_mask`, `normalise_fov`, `quality_features`, `quality_label`, `enhance` | `+drscreen/gate.m`, `quality.m`, `enhance.m` | `imagePretrainedNetwork("efficientnetb0")` + `trainnet` (gate); `graythresh`, `imbinarize`, `bwareafilt`, `bwconvhull`, `regionprops("Circularity")`; `imgaussfilt` (Laplacian variance via `imfilter(fspecial("laplacian"))`), `imgaussfilt` (Ben Graham), `adapthisteq` (green channel), `imbilatfilt` |
| `venus/stage1_segment.py` — `optic_disc`, `fovea`, `vessels`, `red_lesions`, `bright_lesions`, `colour_normalise`, `unet_lesions` (512 px network + 1024 px network for the classes in `config/lesion_thresholds_1024.json`), `quadrant_counts` | `+drscreen/segment.m`, `colourNormalise.m` | `imgaussfilt`, `bwconncomp`, `regionprops("Centroid","Area","MajorAxisLength","MinorAxisLength")`; `fibermetric` (Frangi vesselness); `imtophat` / `imbothat` with `strel("disk",r)`; `mad` for the robust threshold; `rgb2lab` / `lab2rgb` for yellowness and for the LAB colour match to the DDR reference; `predict` on the imported U-Nets, the 1024 px one on a fresh `normaliseFov(raw, 1024)`; `bwareaopen` for minimum areas; `imresize(..., 'box') > 0` to bring a 1024 px mask back to 512 without losing one-pixel blobs |
| `venus/stage2_grade.py` — `cnn_grade`, `calibrate`, `rule_grade`, `fuse`; `config.review_policy` | `+drscreen/gradeCNN.m`, `gradeRule.m`, `fuse.m`, `reviewPolicy.m` | `imagePretrainedNetwork("efficientnetb4")` / `dlnetwork`, `predict`; Platt scaling as two scalars from `config/operating_point.json` (`jsondecode`); the rule grader is plain MATLAB `if/elseif` on the ICDR table; the abstain band (logit half-width) and attention floor are read from `config/review_policy.json` by both implementations |
| `venus/stage3_explain.py` — `gradcam`, overlays, `attention_agreement` | `+drscreen/explain.m` | `gradCAM(net, X, "dr_ordinal_thresholds", ...)` per head, `imresize`, `labeloverlay`, `imdilate` for the lesion neighbourhood; `occlusionSensitivity`, `imageLIME` on demand |
| `venus/report.py` | `+drscreen/report.m` | `mlreportgen.dom` (Document, Image, Table, Paragraph), `jsonencode` sidecar |
| `venus/pipeline.py` — `screen_image` | `+drscreen/screenImage.m` | calls the above in order; `tic/toc` per stage |
| `venus/stage4_simulate.py` — `Params`, `simulate`, `compare`, `sweep` | `simulink/district_screening.slx`, `runSweep.m`, `paretoPlot.m` | SimEvents Entity Generator (Poisson, PHC attribute), Entity Queue, Entity Server (cameras per PHC); `DoctorPoolDES.m`, a `matlab.DiscreteEventSystem` (priority queue + K doctor slots with the daily hour budget), Entity Output Switch (AI triage reading `sens`, `spec`, `flag_rate` from the operating-point JSON), Entity Terminator with counters; `Simulink.SimulationInput` + `parsim` for the sweep; `paretoPlot.m` = the cost-vs-missed scatter |
| `venus/stage5_schedule.py` — `priority_tier`, `allocate`, SQLite | `+drscreen/tier.m`, `allocate.m`, Database Toolbox | `sqlite("venus.db")`, `sqlwrite`/`fetch`; `allocate(queue, slots, now)` is the same pure function, unit-tested with `matlab.unittest.TestCase` on the same synthetic starvation and bump cases |
| `eval/calibrate.py` | `eval/calibrate.m`, `scoreExternal.m`, `bootstrapCI.m` | `fitglm(..., "Distribution","binomial")` on the logit (Platt), `perfcurve`, `bootstrp`, `confusionmat`; the fingerprint via `Simulink.getFileChecksum` or `java.security.MessageDigest` |
| `eval/timing.py` | `eval/timing.m` | `timeit` / `tic-toc` over 50 images |
| `backend/main.py` + `frontend/` | `app/DRScreen.mlapp` (App Designer, four screens) and optionally MATLAB Production Server `screenImage(bytes) -> json` | `uifigure`, `uiimage`, `uitable`, `uiaxes`; `productionServerCompiler` |
| `backend/tests/test_stages.py` | `tests/` one `TestCase` per stage | `matlab.unittest` |

**Weights.** The Keras checkpoints (EfficientNet-B4 grader, EfficientNet-B0 gate) import
into MATLAB with `importNetworkFromTensorFlow` after `model.export("saved_model")` in
TF 2.21; the ordinal head and the Platt parameters are unchanged, so the locked operating
point transfers as is.

**What stays identical across both implementations**: the stage boundaries, the result
struct (`stage0 … stage5`, `timing_ms`, `model_version`, `calibration_fingerprint`), the
operating-point file and its fingerprint check, the ICDR rule table with component
thresholds, the attention-agreement definition, the simulation parameter struct and outputs,
and the tier table and allocator semantics.
