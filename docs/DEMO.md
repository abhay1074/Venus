# Demo script — seven minutes

Before you start: `scripts\serve.ps1`, then `python scripts\preflight.py` must print
`all checks passed`. The first request after start-up is already warm. Numbers quoted below
are from `docs/VALIDATION.md`; if that file has been regenerated since, use its values.

## 1. The gate (45 s) — Screen

Click **Not a fundus (dermoscopy)** → Screen. Refused before any DR model runs.
Click **Blurred capture**. Stage 0 says *"Image too blurry — hold still and refocus"* — the
sentence the PHC operator sees. Nothing diagnostic ran; tier P0, retake on the same visit.

> "Portable cameras produce bad images. The first thing the pipeline decides is whether this
> is a retina and whether it is gradable, and it tells the operator why to retake. The learned
> quality classifier decides whether to enhance; we measured that letting it refuse images
> would have thrown away 78 % of proliferative cases, so refusal stays with hard limits."

## 2. The result (2 min) — Screen

Click **Proliferative DR (NEI)** → Screen (~4 s).

Point at, in order:
1. **Referable DR · P = 1.00**, ICDR grade 4, **P(NV) = 1.00** — proliferative evidence as a
   classifier probability, stated as "not localised".
2. **CNN grade 4, rule grade 4** — two independent graders that agree. Open the rule trace:
   the U-Net found 13 microaneurysms, 7 hemorrhages, 3 exudates; the ICDR line that fired is
   printed. A clinician can check it against the textbook.
3. Toggle **Lesion overlay** → **Grad-CAM · referable**.
4. **P1 · within 7 days · district hospital** — the tier.
5. **PDF report** — the one page a clinician checks in a minute; model version and
   calibration fingerprint in the footer.

Click **NPDR, washed-out (NEI)**: CNN grade 3, P = 0.94; the U-Net counts 24 MA and 11 HE;
attention agreement 0.51 (lift 2.1) — the network looked at the lesions, and that is a
number. Click **DR with macular exudates**: 12 exudates found, grades agree, attention lift 4.4.
Click **Normal right eye**: not referable, P = 0.00, no lesions, P4 routine.

> "Explainability that changes a decision, not one that decorates it: when the graders or the
> heatmap disagree, a person sees the case, with the evidence attached."

## 3. The review queue (30 s)

Flagged first, both grades side by side, why each was flagged. The flag rate measured on
validation images is the workload the simulation absorbs.

## 4. The evidence (1 min) — Validation

> "The external test is 3,720 images from a different acquisition source than every training
> image, frozen with a SHA-256 before training and scored once. The threshold was chosen on a
> separate 2,000-image calibration set at 90 % sensitivity and locked; the server refuses to
> start if the calibration file changes."

**AUC 0.975 [0.970, 0.979] · sensitivity 0.947 · specificity 0.889 · PPV 0.65 at 18 %
prevalence · ECE 0.042.** All four problem-statement targets met, with confidence intervals.
Beside it, the within-source held-out set (AUC 0.954) and the five-grade metrics for contrast.

## 5. The district (2 min) — District

**Run one year, AI vs no-AI** at 3 ophthalmologists (~1 s for 2 × 100,000 patients), then the
**Pareto front**: 144 full-year runs over cameras × doctors × operating point, with the four
slide numbers under editable constraints (missed ≤ 20 % of referable cases, p95 wait ≤ 7 days).

> "The simulation is parameterised by the numbers on the previous screen — not textbook
> numbers. Change the model and the staffing answer changes with it."

## 6. The appointment (45 s) — Appointments

Fill name, phone, village, HbA1c 9.5, tick blurred vision, paste the session, consent →
**Compute tier and book**. Tier reasons are listed; the earliest slot at the nearest clinic
inside the deadline is booked; the SMS text is shown. In the worklist record **no-show**:
the patient re-queues at the same tier with the waiting time preserved.

## Questions judges ask

**"How do we know these numbers are real?"** Every number is in a JSON file the code wrote
when it measured, with the fingerprint of the data it measured on; the docs are generated
from those files. The external test was scored once; a second scoring refuses.

**"Why is the U-Net's MA score low?"** Microaneurysms are 2–4 px at 512; the DDR paper's own
MA AUPR is ~0.11, ours 0.08. HE and EX (0.45, 0.48) match or beat it. The rule grader is a
consistency check, and disagreement goes to a human — that is its job.

**"MATLAB?"** `matlab/` is a line-for-line port: same structs, same config files, same tests.
It parses and its logic passes its cases under Octave; the toolbox parts are verified in MATLAB.

**"Does it work offline?"** Yes: models, SQLite records and PDF reports are local; the API
serves the built front end itself; `scripts/build-offline-bundle.ps1` makes the PHC folder.
