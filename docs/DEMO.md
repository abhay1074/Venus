# Demo script — seven minutes

Before you start: `scripts\serve.ps1`, then `python scripts\preflight.py` must print
`all checks passed`. The first request after start-up is already warm.

## 1. The gate (45 s) — Screen

Click **Not a fundus (dermoscopy)** → Screen. Refused before any DR model runs.
Click **Blurred capture**. Stage 0 says *"Image too blurry — hold still and refocus"* — the
sentence the PHC operator sees. Nothing diagnostic ran; tier P0, retake on the same visit.

> "Portable cameras produce bad images. The first thing the pipeline decides is whether this
> is a retina and whether it is gradable, and it tells the operator why to retake."

## 2. The result (2 min) — Screen

Click **DR with macular exudates** → Screen (~4 s).

Point at, in order:
1. **Referable DR · P = 0.93** against the locked threshold 0.33, with the abstain band drawn.
2. **CNN grade 2, rule grade 2** — two independent graders that agree. Open the rule trace:
   "6 hard exudates; 2 microaneurysms → moderate NPDR". A clinician can check that line
   against the ICDR table.
3. Toggle **Lesion overlay** → **Grad-CAM · referable**. The heatmap sits on the exudate
   cluster. **Attention agreement 0.12, chance 0.05, lift 2.4** — the network looked at the
   lesions, and that is a number, not an impression.
4. **P2 · within 30 days · nearest ophthalmology clinic** — the tier.
5. **PDF report** — the one page a clinician checks in a minute; model version and
   calibration fingerprint in the footer.

Click **Dark, vignetted capture**: quality *usable* → enhancement ran → graded. Then
**NPDR, washed-out**: CNN says referable, the classical detections are elsewhere, attention
agreement 0.01 → **human review (P3)** with both grades attached.

> "Explainability that changes a decision, not one that decorates it: when the graders or the
> heatmap disagree, a person sees the case, with the evidence."

## 3. The review queue (30 s)

Flagged first, both grades side by side, why each was flagged. The measured flag rate is
the workload the simulation absorbs.

## 4. The district (2 min) — District

**Run one year, AI vs no-AI** at 3 ophthalmologists (~1 s for 2 × 100,000 patients).

> "With three doctors and no AI, the district reads 39 % of its referable cases within the
> year and the p95 wait is four months. With the AI at its measured 91 % / 61 %, 70 % and
> three weeks. Those misses are not typed in; they come out of the confusion matrix."

Scroll to the **Pareto front**: 144 full-year runs over cameras × doctors × operating
point. The four slide numbers under the constraints (≤ 20 % missed, p95 wait ≤ 7 days):
**5 ophthalmologists with AI vs 7 without, ₹1.31 Cr vs ₹1.72 Cr a year, 1,064 vs 932
missed** — and both constraints are editable because they are the district officer's call.

## 5. The appointment (1 min) — Appointments

Fill name, phone, village, HbA1c 9.5, tick blurred vision, paste the session, consent →
**Compute tier and book**. Tier reasons are listed; the earliest slot at the nearest clinic
inside the deadline is booked; the SMS text is shown. In the worklist record **no-show**:
the patient re-queues at the same tier with the waiting time preserved.

## 6. The evidence (45 s) — Validation

Patient-disjoint split, Platt scaling (ECE 0.31 → 0.04), threshold locked at 90 % sensitivity
on the calibration half, test half scored once, bootstrap CIs. **Specificity 0.61 — target
0.85 missed, reported with its CI, not re-tuned.** Benchmark table beside Gulshan 2016.

> "Every number on this page has a fingerprint. If the calibration file changes, the server
> refuses to start."

## Questions judges ask

**"Why is specificity 0.61?"** Public data, 48k training images, single-grader labels. The
published 98 % used 128k adjudicated images. The Pareto front shows what each point of
specificity is worth in doctor-hours; the next grader re-runs the identical protocol.

**"Why classical segmentation?"** The only public pixel-labelled DR set has 81 images. A
network trained on it could not be validated, so the rule grade is a consistency check with
component thresholds, and disagreement goes to a human — that is its job.

**"MATLAB?"** This is the Python reference implementation; `docs/MATLAB_MAPPING.md` maps every
module to the toolbox functions and the SimEvents blocks, with identical structs and files.

**"Does it work offline?"** Yes: models, SQLite records and PDF reports are local; nothing
in the served path calls out.
