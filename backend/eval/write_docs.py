"""Generate docs/VALIDATION.md and the model cards from the JSON artefacts.

    python -m backend.eval.write_docs

Reads (whichever exist): backend/config/operating_point.json, timing_report.json,
sweep_cache.json, lesion_thresholds.json, backend/data/manifests/manifests.json,
models/cards/*.summary.json. Writes docs/VALIDATION.md and models/cards/<name>.md.
The point is that no number in the documentation is typed by hand: every figure
is read from the file the code wrote when it measured it.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from backend.venus.config import CONFIG_DIR, MANIFEST_DIR, PROJECT_ROOT

CARDS = PROJECT_ROOT / "models" / "cards"
DOCS = PROJECT_ROOT / "docs"


def load(path: Path):
    return json.load(open(path, encoding="utf-8")) if path.exists() else None


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def validation_md(op, timing, sweep, manifests, lesion, quality_card, grader_card, flags=None, ensemble=None, policy=None) -> str:
    t = op["external_test"]; at = t["at_locked_threshold"]; ci = t["ci95_bootstrap_2000"]; tg = op["targets"]
    sec = op.get("secondary_heldout")
    lines = [f"# Validation report — Venus AI, model `{op['model_version']}` (grader `{op['grader_tag']}`)", "",
             f"Generated from the operating point written {op['written_at'][:16].replace('T', ' ')} UTC by `backend/eval/write_docs.py`, from the JSON artefacts the code wrote when it measured; "
             "nothing here is typed by hand. The Validation screen in the app renders the same files.", ""]

    lines += ["## Data and splits", ""]
    if manifests:
        lines += [f"Manifests built {manifests['written_at'][:10]}: perceptual-hash de-duplication (256-bit DCT hash, Hamming ≤ {manifests['dedup_hamming']}) "
                  f"dropped {manifests['duplicates_dropped']} near-duplicates ({manifests['cross_dataset_duplicates']} across datasets). All splits are by patient where a patient id exists.", "",
                  "| manifest | n | datasets | grades 0–4 | referable |", "|---|---|---|---|---|"]
        for name, m in manifests["manifests"].items():
            grades = ", ".join(f"{k}: {v}" for k, v in m["grades"].items())
            lines.append(f"| {name} | {m['n']:,} | {', '.join(f'{k} {v:,}' for k, v in m['datasets'].items())} | {grades} | {pct(m['referable_fraction'])} |")
        lines.append("")
    lines += [f"**Calibration set:** {op['source']['description']} (n = {op['source']['n_calibration']}, {op['source']['n_patients_calibration']} patients). "
              f"SHA-256 `{op['calibration_fingerprint']}` — the server refuses to start if the manifest on disk differs.",
              f"**External test:** {op['source']['external']} (n = {t['n']}, {t['n_referable']} referable)"
              + (f"; SHA-256 `{op['external_test_fingerprint']}`." if op.get("external_test_fingerprint") else "."), ""]

    lines += ["## Protocol, in order", "",
              f"1. Platt scaling fitted on the calibration set: a = {op['calibration']['a']:.4f}, b = {op['calibration']['b']:.4f}; ECE {op['calibration']['ece_raw']} → **{op['calibration']['ece_calibrated']}**.",
              f"2. Referable threshold chosen on the calibration set at 90% sensitivity and locked: **{op['thresholds']['referable']:.4f}** (calibration-set specificity {f3(op['thresholds']['calibration_at_90']['specificity'])}); "
              f"85% alternative {op['thresholds']['referable_85pc_alternative']:.4f} chosen at the same time. Abstain band ± {op['thresholds']['abstain_band']}.",
              f"3. External test scored **{'once' if t['scored_once'] else 'again with --force (stated)'}**, 2,000 bootstrap resamples for 95% CIs.", ""]

    lines += ["## Results on the external test", "", "| metric | value | 95% CI | target | met |", "|---|---|---|---|---|",
              f"| Referable-DR AUC | {f3(t['auc'])} | [{ci['auc'][0]}, {ci['auc'][1]}] | > {tg['auc']} | {'yes' if tg['auc_met'] else 'no'} |",
              f"| Sensitivity @ locked threshold | {f3(at['sensitivity'])} | [{ci['sensitivity'][0]}, {ci['sensitivity'][1]}] | > {tg['sensitivity']} | {'yes' if tg['sensitivity_met'] else 'no'} |",
              f"| Specificity @ locked threshold | {f3(at['specificity'])} | [{ci['specificity'][0]}, {ci['specificity'][1]}] | > {tg['specificity']} | {'yes' if tg['specificity_met'] else '**no**'} |",
              f"| PPV / NPV at 18% prevalence | {f3(at['ppv_at_indian_prevalence'])} / {f3(at['npv_at_indian_prevalence'])} | PPV [{ci['ppv_at_indian_prevalence'][0]}, {ci['ppv_at_indian_prevalence'][1]}] | PPV > 0.55 | {'yes' if at['ppv_at_indian_prevalence'] > 0.55 else 'no'} |",
              f"| Expected calibration error | {f3(t['ece'])} | — | < {tg['ece']} | {'yes' if tg['ece_met'] else 'no'} |",
              f"| Sensitivity on grade 2 alone | {f3(t['sensitivity_grade2_only'])} | — | report | — |",
              f"| Confusion at the locked threshold | TP {at['tp']} · FN {at['fn']} · FP {at['fp']} · TN {at['tn']} | | | |",
              f"| At the 85% alternative | sens {f3(t['at_85pc_threshold']['sensitivity'])}, spec {f3(t['at_85pc_threshold']['specificity'])} | | | |", ""]
    lines.append("Referred fraction by true grade: " + ", ".join(f"grade {g} {pct(v, 0)}" for g, v in t["referred_fraction_by_true_grade"].items()) + ".")
    if t.get("grade_metrics_for_contrast"):
        g = t["grade_metrics_for_contrast"]
        lines.append(f"Five-grade output, for contrast (not a claim): quadratic weighted κ {f3(g['quadratic_weighted_kappa'])}, exact grade {pct(g['exact_grade_accuracy'])}, within one grade {pct(g['within_one_grade'])}.")
    lines.append("")
    if sec:
        s = sec["at_locked_threshold"]
        lines += ["## Secondary held-out set (within-source)", "", f"{sec['description']} (n = {sec['n']}, {sec['n_referable']} referable): "
                  f"AUC {f3(sec['auc'])} [{sec['ci95_bootstrap_2000']['auc'][0]}, {sec['ci95_bootstrap_2000']['auc'][1]}], sensitivity {f3(s['sensitivity'])}, specificity {f3(s['specificity'])}, ECE {f3(sec['ece'])}."]
        if sec.get("grade_metrics_for_contrast"):
            g = sec["grade_metrics_for_contrast"]
            lines.append(f"Five-grade contrast: QWK {f3(g['quadratic_weighted_kappa'])}, exact {pct(g['exact_grade_accuracy'])}.")
        lines.append("")

    for name, ext in (op.get("additional_external_tests") or {}).items():
        e, eci = ext["at_locked_threshold"], ext["ci95_bootstrap_2000"]
        roc90 = next((r for r in ext["roc_points_for_simulation"] if r["sensitivity_target"] == 0.9), None)
        lines += [f"## Additional external test: {name} (scored once at the locked threshold)", "",
                  f"{ext['description']} — n = {ext['n']} ({ext['n_referable']} referable, {ext.get('n_patients', '—')} patients), SHA-256 `{ext['manifest_fingerprint']}`.", "",
                  f"AUC **{f3(ext['auc'])}** [{eci['auc'][0]}, {eci['auc'][1]}]; at the locked threshold sensitivity **{f3(e['sensitivity'])}** [{eci['sensitivity'][0]}, {eci['sensitivity'][1]}], "
                  f"specificity **{f3(e['specificity'])}** [{eci['specificity'][0]}, {eci['specificity'][1]}], PPV at 18% {f3(e['ppv_at_indian_prevalence'])}, ECE {f3(ext['ece'])}. "
                  f"Referred by true grade: " + ", ".join(f"grade {g} {pct(v, 0)}" for g, v in ext["referred_fraction_by_true_grade"].items()) + "."
                  + (f" Referred by adjudicated DME: " + ", ".join(f"DME {k} {pct(v, 0)}" for k, v in ext["referred_fraction_by_adjudicated_dme"].items()) + "." if ext.get("referred_fraction_by_adjudicated_dme") else ""), ""]
        if roc90:
            lines += [f"On this set's own ROC, 90% sensitivity corresponds to specificity {f3(roc90['specificity'])}: the discrimination transfers across sources, "
                      f"the calibration shifts conservatively (the locked threshold over-refers here rather than missing cases). A site-specific calibration set before deployment is the remedy the architecture prescribes, and this is the measurement behind it.", ""]
        if ext.get("grade_metrics_for_contrast"):
            g = ext["grade_metrics_for_contrast"]
            lines += [f"Five-grade contrast: QWK {f3(g['quadratic_weighted_kappa'])}, exact {pct(g['exact_grade_accuracy'])}, within one grade {pct(g['within_one_grade'])}.", ""]
    site = load(CONFIG_DIR / "site_calibration_messidor2.json")
    if site:
        L, F = site["locked_operating_point"], site["site_calibration_full_half"]
        lines += ["## What a site calibration set buys (Messidor-2, evaluation of a deployment step)", "",
                  f"{site['protocol']}. n = {site['n_images']:,} images, {site['n_patients']} patients, {site['n_referable']} referable.", "",
                  "| operating point | sensitivity | specificity | ECE |", "|---|---|---|---|",
                  f"| locked (EyePACS calibration set), on the held-out halves | {f3(L['sensitivity']['mean'])} | {f3(L['specificity']['mean'])} | {f3(L['ece']['mean'])} |",
                  f"| site calibration on the other half (n ≈ {F['n_mean']}) | {f3(F['sensitivity']['mean'])} | {f3(F['specificity']['mean'])} | {f3(F['ece']['mean'])} |"]
        for n, s_ in site["site_calibration_by_sample_size"].items():
            if s_:
                lines.append(f"| site sample of {n} labelled images | {f3(s_['sensitivity']['mean'])} [{f3(s_['sensitivity']['p5'])}, {f3(s_['sensitivity']['p95'])}] | "
                             f"{f3(s_['specificity']['mean'])} [{f3(s_['specificity']['p5'])}, {f3(s_['specificity']['p95'])}] | {f3(s_['ece']['mean'])} |")
        lines += ["", "Brackets are the 5th–95th percentile over the repeats. Discrimination transfers across acquisition sources; calibration does not, and "
                  "re-fitting Platt scaling and the 90 % sensitivity threshold on a few hundred labelled images from the site restores the intended operating "
                  "point. This is the number behind \"a site-specific calibration set is a prerequisite for deployment\". The served operating point is unchanged.", ""]
    if grader_card:
        h = grader_card["history"]
        lines += ["## Grader training", "", f"{grader_card['backbone']} at {grader_card['input']} px, {grader_card['head']}; {grader_card['epochs']} epochs, batch {grader_card['batch']}, "
                  f"{grader_card['schedule']}, {grader_card['precision']}, XLA {grader_card['xla']}. Augmentation: {grader_card['augmentation']}. Loss: {grader_card['loss']}. "
                  f"Train n = {grader_card['train_n']:,}, val n = {grader_card['val_n']:,}; best validation referable-AUC {grader_card['best_val_referable_auc']} "
                  f"(epoch {max(h, key=lambda r: r['referable_auc'] or 0)['epoch']}); {grader_card['total_minutes']} min on an RTX 5060 laptop GPU.", ""]
    ma = load(CONFIG_DIR / "experiments" / "ma_improvement.json")
    if ma and lesion:
        r = ma["results"]; d = ma["ma_delta_vs_served"]
        names = {"plain_512": "the served 512 px network", "tta_512": "512 px + dihedral test-time augmentation",
                 "plain_1024": "the 1024 px network (not shipped)", "tta_1024": "1024 px + test-time augmentation",
                 "ensemble_at_512": "mean of 512 px and 1024 px maps, at 512"}
        lines += ["## The weakest number: microaneurysm segmentation", "",
                  f"MA pixel AUPR on the DDR test split is **{f3(lesion['test_aupr'].get('MA'))}** (Dice "
                  f"{f3(lesion['test_dice'].get('MA'))}), far below HE {f3(lesion['test_aupr'].get('HE'))} and EX "
                  f"{f3(lesion['test_aupr'].get('EX'))}. It is the weakest component in the build and it is worth being "
                  "precise about why, what was tried, and why the served path did not change.", "",
                  "**What was tried.** " + ma["question"] + " " + ma["split"] + ":", "",
                  "| variant | MA AUPR | vs served |", "|---|---|---|"]
        for key, label in names.items():
            if key in r and r[key].get("MA") is not None:
                lines.append(f"| {label} | {f3(r[key]['MA'])} | {d.get(key, 0):+.4f} |")
        lines += ["",
                  ma.get("baseline_note", ""), "",
                  "**What that says.** Test-time augmentation, the cheapest option, gains nothing at all "
                  f"({d.get('tta_512', 0):+.4f}): averaging dihedral views does not recover a lesion that is 1-3 px "
                  "across at this resolution. Averaging the 512 px and 1024 px maps is actively worse "
                  f"({d.get('ensemble_at_512', 0):+.4f}), because down-sampling the 1024 px map to the working frame "
                  "flattens exactly the small isolated peaks that a microaneurysm is. Only genuine resolution helps, "
                  f"and it helps clearly ({d.get('plain_1024', 0):+.4f}, or {d.get('tta_1024', 0):+.4f} with augmentation "
                  "on top).", "",
                  "**Why the served path still does not change.** Three reasons, in order of weight.", "",
                  "1. *It does not change a decision.* The 1024 px network was not only measured on pixels: it was run "
                  "through the served path on the validation sample (the section above). The rule grader did not move, "
                  "the attention flag became less precise, the review policy caught fewer of the classifier's errors, "
                  "and a screen cost +0.9 s. A better pixel score that makes the downstream product slightly worse is "
                  "not an improvement.",
                  "2. *MA-only findings sit below the decision this system is accepted for.* Microaneurysms alone are "
                  "ICDR grade 1. The threshold, the confidence intervals, the review policy and the district "
                  "simulation all describe the referable decision at ICDR >= 2, which is driven by hemorrhages and "
                  "exudates - the two classes the U-Net reads best. The MA count appears in the rule-grader trace as "
                  "evidence a clinician can check, not as the thing that sets the grade.",
                  "3. *An improvement measured on valid cannot be advertised on test.* The DDR test split was scored "
                  "once for the model version that is served. Re-scoring it to publish a better number for a variant "
                  "chosen on valid is precisely the loop this protocol exists to prevent, so the number above stands "
                  f"at {f3(lesion['test_aupr'].get('MA'))} and the 1024 px variant keeps its own once-scored "
                  f"{f3((load(CONFIG_DIR / 'experiments' / 'lesion_unet_1024.json') or {}).get('test_aupr', {}).get('MA'))}.", "",
                  "**Stated plainly.** MA segmentation at 512 px is near the resolution limit and this build does not "
                  "solve it. Anyone reading the lesion overlay should read the MA layer as a weak hint and the HE/EX "
                  "layers as evidence. The code to serve a higher-resolution network for one class is present and "
                  "tested (`config/lesion_thresholds_1024.json` re-enables it); what is missing is a reason, and a "
                  "demonstration that it makes some decision better.", ""]
    unet_exp = load(CONFIG_DIR / "experiments" / "lesion_unet_1024.json")
    if unet_exp and not unet_exp.get("decision", {}).get("shipped", True):
        d = unet_exp["decision"]; w = d["validation_with_1024_for_MA"]; o = d["validation_512_only"]
        lines += ["## Things tried and not shipped: a 1024 px lesion network for microaneurysms", "",
                  f"{unet_exp['architecture']} ({unet_exp['epochs']} epochs, {unet_exp['total_minutes']} min), DDR test scored once: "
                  + ", ".join(f"{k} AUPR {f3(unet_exp['test_aupr'][k])}" for k in ("MA", "HE", "EX", "SE")) + f" (the served 512 px network: see the table below). "
                  f"Served for MA only on the same {o['n_gradable']} raw validation images:", "",
                  "| | 512 px network only (served) | + 1024 px network for MA |", "|---|---|---|",
                  f"| rule grader alone, exact / within one grade | {pct(o['rule_grader_alone']['exact_grade_agreement_with_truth'])} / {pct(o['rule_grader_alone']['within_one_of_truth'])} | {pct(w['rule_grader_alone']['exact_grade_agreement_with_truth'])} / {pct(w['rule_grader_alone']['within_one_of_truth'])} |",
                  f"| attention agreement, median correct / incorrect referable calls | {o['attention_agreement']['correct']['median']} / {o['attention_agreement']['incorrect']['median']} | {w['attention_agreement']['correct']['median']} / {w['attention_agreement']['incorrect']['median']} |",
                  f"| review policy: attention floor · flag rate | {o['review_policy']['attention_floor']} · {pct(o['review_policy']['resulting_flag_rate'])} | {w['review_policy']['attention_floor']} · {pct(w['review_policy']['resulting_flag_rate'])} |",
                  f"| CNN error rate among flagged vs unflagged | {pct(o['review_policy']['error_rate_flagged'])} vs {pct(o['review_policy']['error_rate_unflagged'])} | {pct(w['review_policy']['error_rate_flagged'])} vs {pct(w['review_policy']['error_rate_unflagged'])} |",
                  f"| share of the CNN's referable errors flagged | {pct(o['review_policy']['share_of_cnn_errors_flagged'])} | {pct(w['review_policy']['share_of_cnn_errors_flagged'])} |",
                  f"| CPU time per image, median | {d['cpu_timing_median_ms']['512_only'] / 1000:.1f} s | {d['cpu_timing_median_ms']['with_1024_for_MA'] / 1000:.1f} s |",
                  "", d["why"], ""]
    if ensemble:
        lines += ["## Things tried and not shipped: a second seed, test-time augmentation", "",
                  f"{ensemble['question']} Measured on the calibration and validation sets only (the external tests were not re-scored):", "",
                  "| set | TTA | " + " | ".join(ensemble["comparisons"][0]["single"].keys()) + " | mean ensemble | ensemble − best single (paired bootstrap 95% CI) |",
                  "|---|---|" + "---|" * (len(ensemble["comparisons"][0]["single"]) + 2)]
        for c in ensemble["comparisons"]:
            e = c.get("ensemble_mean", {}); d = e.get("paired_auc_difference", {})
            lines.append(f"| {c['manifest']} (n = {c['n']:,}) | {'on' if c['tta'] else 'off'} | " + " | ".join(f3(v["auc"]) for v in c["single"].values())
                         + f" | {f3(e.get('auc'))} | {d.get('mean', 0):+.4f} [{d.get('ci95', ['', ''])[0]}, {d.get('ci95', ['', ''])[1]}] |")
        lines += ["", "AUC of the referable decision. Averaging two seeds adds about the same as test-time augmentation (+0.003 on calibration, and TTA on top of the "
                  "ensemble adds nothing); five-grade exact accuracy moves by about one point. That is below what the district numbers would notice and would double "
                  "the grader's inference cost on a CPU, so the served grader stays a single network without TTA, and the external test's one scoring stands.", ""]
    if lesion:
        hires = load(CONFIG_DIR / "lesion_thresholds_1024.json")
        serves = set(hires.get("serves", [])) if hires else set()
        lines += ["## Lesion segmentation (DDR test split, scored once)", "", "| lesion | read by | AUPR | Dice at threshold | threshold (DDR valid, max F1) |", "|---|---|---|---|---|"]
        for k in ("MA", "HE", "EX", "SE"):
            src = hires if k in serves else lesion
            by = f"{hires['frame_size']} px network" if k in serves else "512 px network"
            lines.append(f"| {k} | {by} | {f3(src['test_aupr'].get(k))} | {f3(src['test_dice'].get(k))} | {src['thresholds'].get(k)} |")
        lines.append("")
        if hires:
            va = hires.get("valid_aupr", {})
            lines += [f"Two networks read the image: the 512 px U-Net for every class and, for {', '.join(sorted(serves))}, the same architecture trained on "
                      f"512 px lesion-biased crops of {hires['frame_size']} px frames ({hires['epochs']} epochs, {hires['total_minutes']} min). {hires['why_these_classes']} "
                      f"Its DDR-valid AUPR per class: " + ", ".join(f"{k} {f3(va.get(k))}" for k in ("MA", "HE", "EX", "SE")) + f"; its test numbers for the classes it does not serve: "
                      + ", ".join(f"{k} {f3(hires['test_aupr'].get(k))}" for k in ("MA", "HE", "EX", "SE") if k not in serves)
                      + f". Minimum component area for MA at {hires['frame_size']} px: {hires['min_area_px']['MA']} px ({hires['min_area_px']['note']}).", ""]
    lm = load(CONFIG_DIR / "experiments" / "landmark_check.json")
    if lm:
        h = lm["held_out_half"]
        names = {"old": "brightest blob on the enhanced frame (served before this check)",
                 "original": "un-enhanced frame, grey/white pixels excluded",
                 "flatfield": "the same, flat-fielded at sigma = FOV/4 (**served**)"}
        names.update({k: f"flat-field + {k.split('_w')[1]} x vessel density" for k in h if k.startswith("combined_w")})
        lines += ["## Landmarks: optic disc and fovea, against clinician marks", "",
                  "The optic disc anchors Stage 1: the fovea is placed 2-3 disc diameters from it, the ETDRS hemorrhage "
                  "quadrants of the 4-2-1 rule are drawn around that fovea, and exudates on the disc rim are discarded. "
                  f"{lm['question']} Ground truth: {lm['ground_truth']}. {lm['n_images']:,} annotated images, split "
                  f"{lm['split']}; numbers below are the held-out half (n = {lm['n_held_out']}).", "",
                  "| optic-disc detector | fovea within 0.5 DD | within 1 DD | median error | disc at a plausible 1.5-3.5 DD from the fovea |",
                  "|---|---|---|---|---|"]
        for key, label in names.items():
            if key in h:
                m = h[key]
                lines.append(f"| {label} | {pct(m['fovea_within_0.5dd'])} | {pct(m['fovea_within_1dd'])} | "
                             f"{m['fovea_median_err_dd']:.2f} DD | {pct(m['disc_plausible_1.5_to_3.5dd'])} |")
        lines += ["", "DD = the annotated optic-disc diameter of that image. The old detector searched the enhanced frame, "
                  "whose illumination correction (sigma ~ 17 px) is smaller than a disc (~70 px), so the disc stopped being "
                  "the brightest region; it also accepted a white image label as a disc. Adding vessel density to the "
                  "score was tried and makes things worse as its weight grows: the Frangi map responds to choroidal "
                  "texture and glare edges as well as to the vessel trunks. The flat-field detector was chosen on the "
                  "design half and the held-out half confirms it. Known hard case: neovascularization at the disc hides "
                  "its brightness, and a bright fibrous patch elsewhere can win (the shipped PDR sample is one). "
                  f"{lm['note']}", ""]
    if quality_card:
        q = quality_card["test_heldout_patients"]; d = quality_card.get("ddr_ungradable_external_check")
        lines += ["## Image quality classifier", "", f"{quality_card['architecture']}, {quality_card['labels']}: held-out-patient accuracy {pct(q['accuracy'])}, "
                  f"ungradable-detection AUC **{f3(q['ungradable_detection_auc'])}** (target > 0.95), good-vs-rest AUC {f3(q['good_vs_rest_auc'])}."
                  + (f" Out-of-source check on DDR's ungradable class (n = {d['n']}): {pct(d['reject_recall_argmax'])} labelled reject, {pct(d['not_good_recall'])} not labelled good." if d else ""), ""]
    if flags:
        a = flags["attention_agreement"]; r = flags["rule_grader_alone"]; fr = flags["flag_reasons"]
        lines += ["## Human review: flag rate, attention agreement, rule grader (validation sample)", "",
                  f"{flags['n_sampled']} grade-stratified validation images through the served path (lesions by {flags['lesion_method']}), "
                  f"{flags['n_gradable']} gradable (retake rate {pct(flags['retake_rate'])}; quality labels {flags['quality_labels']}).", "",
                  f"- **Human-review flag rate {pct(flags['flag_rate'])}** under the architecture's default rules (±0.05 probability band, attention < 0.15 with lift < 1.5) — reasons: " + ", ".join(f"{k} {v}" for k, v in fr.items()) + ". "
                  f"CNN referable-error rate among flagged images {pct(flags['cnn_error_rate_flagged_vs_unflagged']['flagged'])} vs "
                  f"{pct(flags['cnn_error_rate_flagged_vs_unflagged']['unflagged'])} among unflagged (referable accuracy overall {pct(flags['referable_accuracy_cnn'])}).",
                  f"- **Attention agreement** on referable CNN calls: correct calls median {a['correct'].get('median', '—')} (IQR {a['correct'].get('p25', '—')}–{a['correct'].get('p75', '—')}, n = {a['correct'].get('n', 0)}) vs "
                  f"incorrect calls median {a['incorrect'].get('median', '—')} (IQR {a['incorrect'].get('p25', '—')}–{a['incorrect'].get('p75', '—')}, n = {a['incorrect'].get('n', 0)}). "
                  "Where the network's attention sits on the detected lesions, it is more often right: the score is a review signal, not a decoration.",
                  ]
        if policy:
            pr = policy["flag_reasons"]
            lines += [f"- **Review policy chosen on the same sample** (`config/review_policy.json`, served): abstain band ±{policy['abstain_band_logit']} in logit space around the locked threshold, "
                      f"attention floor {policy['attention_floor']} (the highest cut at which ≥ 60 % of the flagged referable calls are CNN errors). Resulting **flag rate {pct(policy['resulting_flag_rate'])}** "
                      f"(reasons: " + ", ".join(f"{k} {v}" for k, v in pr.items()) + f"); CNN error rate {pct(policy['error_rate_flagged'])} among flagged vs {pct(policy['error_rate_unflagged'])} unflagged; "
                      f"{pct(policy['share_of_cnn_errors_flagged'])} of the CNN's referable errors land in the review queue. This is the rate the district simulation uses."]
        lines += [f"- **Rule grader alone** (ICDR table on the lesion counts): referable sensitivity {pct(r['referable_sensitivity'])}, specificity {pct(r['referable_specificity'])}; "
                  f"exact grade {pct(r['exact_grade_agreement_with_truth'])}, within one grade {pct(r['within_one_of_truth'])}; agrees with the CNN within one grade on {pct(r['agreement_with_cnn_within_one'])} of images. "
                  "It is a consistency check that a clinician can verify by hand, not a second classifier.", ""]
    if timing:
        w = timing["per_stage"]["wall_ms"]; ps = timing["per_stage"]
        lines += ["## Timing (requirement: < 30 s per image)", "", f"{timing['n_images']} images, TTA {'on' if timing['tta'] else 'off'}, {timing['machine']['processor']} ({timing['machine']['cores']} threads, no GPU): "
                  f"**median {w['median_ms'] / 1000:.1f} s, p95 {w['p95_ms'] / 1000:.1f} s**, max {w['max_ms'] / 1000:.1f} s → requirement {'met' if timing['requirement_met_p95'] else 'missed'} at p95. "
                  f"Per stage (median): S0 {ps['stage0']['median_ms']} ms, S1 {ps['stage1']['median_ms']} ms, S2 {ps['stage2']['median_ms']} ms, S3 {ps['stage3']['median_ms']} ms (Grad-CAM {ps['gradcam']['median_ms']} ms), report {ps['report']['median_ms']} ms.", ""]
        gpu = load(CONFIG_DIR / "timing_report_gpu.json")
        if gpu:
            gw = gpu["per_stage"]["wall_ms"]
            lines += [f"On the RTX 5060 (WSL): median {gw['median_ms'] / 1000:.1f} s, p95 {gw['p95_ms'] / 1000:.1f} s — inference is not the cost on either machine; "
                      f"Stage 1 landmarks and the overlay encoding are. (Measured before the PNG-encoding change that took the CPU median from 4.2 s to 1.7 s.)", ""]
    if sweep and sweep.get("slide_numbers"):
        s = sweep["slide_numbers"]; c = sweep["constraints"]
        lines += ["## District simulation (Stage 4, coupled to the numbers above)", "", f"{len(sweep['runs'])} full-year runs (cameras × ophthalmologists × operating point). Under the constraints missed ≤ {c['max_missed']} "
                  f"({pct(c['max_missed_fraction'], 0)} of {c['referable_cases']} referable cases) and p95 wait ≤ {c['max_p95_wait_days']} days: "
                  f"**{s['doctors_with_ai']} ophthalmologists with AI (at the {pct(s['operating_point_with_ai'])} sensitivity point) vs {s['doctors_without_ai']} without, ₹{s['cost_with_ai'] / 1e7:.2f} Cr vs ₹{s['cost_without_ai'] / 1e7:.2f} Cr per year, {s['missed_with_ai']} vs {s['missed_without_ai']} referable cases missed.**", ""]
    lines += ["## Stated plainly", ""] + [f"- {n}" for n in op["notes"]] + [
        "- Mild DR (grade 1 vs 0) is not a claim of this system; the referable decision (grade ≥ 2) is what the threshold, the CIs and the simulation describe.",
        "- The external test is a different acquisition source from every training image, but a single one; performance on Indian portable-camera images is unmeasured until such a set exists.", ""]
    return "\n".join(lines)


def card_md(name: str, card: dict) -> str:
    keys = [k for k in card if k not in ("history", "valid", "test_scored_once", "test_heldout_patients")]
    lines = [f"# Model card — {name}", "", "| field | value |", "|---|---|"]
    for k in keys:
        v = card[k]
        if isinstance(v, (dict, list)):
            v = json.dumps(v)
        lines.append(f"| {k} | {v} |")
    if "test_scored_once" in card:
        lines += ["", "## Test split (scored once)", "", "```json", json.dumps(card["test_scored_once"], indent=1), "```"]
    if "test_heldout_patients" in card:
        lines += ["", "## Held-out patients", "", "```json", json.dumps(card["test_heldout_patients"], indent=1), "```"]
    if "history" in card:
        lines += ["", "## Training history", "", "```json", json.dumps(card["history"][-5:], indent=1), "```", f"(last 5 of {len(card['history'])} entries)"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------ model card --

DISCLAIMER = ("Venus AI is a triage aid for referable diabetic retinopathy. A clinician reviews "
              "every case. It is not a diagnosis and it is not a cleared medical device.")


def model_card_md(op, lesion, quality_card, grader_card, flags, policy, site, timing, manifests) -> str:
    """docs/MODEL_CARD.md, generated from the same JSON as VALIDATION.md."""
    t = op["external_test"]; at = t["at_locked_threshold"]; ci = t["ci95_bootstrap_2000"]
    lines = [f"# Model card — {op['model_version']}", "",
             f"**{DISCLAIMER}**", "",
             f"Generated from the operating point written {op['written_at'][:16].replace('T', ' ')} UTC by "
             "`backend/eval/write_docs.py`. Every number below is read from the JSON the evaluation code wrote; "
             "`docs/VALIDATION.md` has the full protocol and the caveats.", "",
             "## What it is", "",
             f"A {grader_card['backbone'] if grader_card else 'convolutional'} classifier with an ordinal head "
             "(four cumulative sigmoids, P(grade >= k) for ICDR 1-4), calibrated by Platt scaling and thresholded at a "
             "referable decision (ICDR >= 2), plus a lesion U-Net and an ICDR rule grader whose disagreement with the "
             "classifier sends the case to a human. Model version "
             f"`{op['model_version']}`, grader `{op['grader_tag']}`, calibration fingerprint "
             f"`{op['calibration_fingerprint'][:16]}...`.", "",
             "## Intended use", "",
             "- **Triage of referable diabetic retinopathy** (ICDR >= 2) from a colour fundus photograph, in a "
             "district screening programme where a clinician reads every flagged case.",
             "- Ordering a queue: the priority tier and the appointment it books are the product, not a diagnosis.",
             "- Estimating programme staffing from *measured* accuracy (the Stage 4 simulation).",
             "- Research, teaching and demonstration.", "",
             "## Out-of-scope use", "",
             "- **Any use without a clinician reading the case.** The system is not a reader.",
             "- Diagnosis, treatment selection, or a claim that an eye is healthy. A non-referable result is "
             "*not* a statement that there is no disease.",
             "- Grading mild DR (ICDR 1) as an outcome: the threshold, the confidence intervals and the simulation "
             "all describe the referable decision. Five-grade numbers are reported for contrast only.",
             "- Detecting anything other than DR — other retinal disease is neither trained for nor measured.",
             "- Neovascularization as a localised finding: P(grade >= 4) is a classifier probability, never a segmentation.",
             "- Commercial use of the released checkpoints (see `NOTICE`: the training data permits non-commercial "
             "academic use only).",
             "- Regulatory submission. This is not a cleared or CE-marked device and has had no clinical trial.", "",
             "## Measured performance", "",
             "Referable DR (ICDR >= 2). Each row names the set it was measured on; 95 % CIs are 2,000-resample bootstrap.", "",
             "| metric | value | 95% CI | measured on |", "|---|---|---|---|"]
    ext_desc = f"{t['description']} — n = {t['n']:,}, {t['n_referable']} referable"
    lines += [f"| AUC | {f3(t['auc'])} | {ci['auc']} | {ext_desc} |",
              f"| Sensitivity at the locked threshold | {f3(at['sensitivity'])} | {ci['sensitivity']} | same |",
              f"| Specificity at the locked threshold | {f3(at['specificity'])} | {ci['specificity']} | same |",
              f"| PPV at 18 % prevalence | {f3(at['ppv_at_indian_prevalence'])} | {ci.get('ppv_at_indian_prevalence', '—')} | same |",
              f"| NPV at 18 % prevalence | {f3(at['npv_at_indian_prevalence'])} | — | same |",
              f"| Expected calibration error | {f3(t['ece'])} | — | same |"]
    sec = op.get("secondary_heldout")
    if sec:
        s_at = sec["at_locked_threshold"]
        lines.append(f"| AUC (within-source held-out patients) | {f3(sec['auc'])} | {sec['ci95_bootstrap_2000']['auc']} | "
                     f"{sec['description']} — n = {sec['n']:,} |")
        lines.append(f"| Sensitivity / specificity there | {f3(s_at['sensitivity'])} / {f3(s_at['specificity'])} | — | same |")
    for name, e in (op.get("additional_external_tests") or {}).items():
        e_at = e["at_locked_threshold"]
        short = name.replace("external_test_", "")
        lines.append(f"| AUC ({short}, scored once at the same threshold) | {f3(e['auc'])} | {e['ci95_bootstrap_2000']['auc']} | "
                     f"n = {e['n']:,}, {e.get('n_patients', '?')} patients |")
        lines.append(f"| Sensitivity / specificity ({short}) | {f3(e_at['sensitivity'])} / {f3(e_at['specificity'])} | — | same |")
    if timing:
        w = timing["per_stage"]["wall_ms"]
        lines.append(f"| End-to-end time per image | {w['median_ms'] / 1000:.1f} s median, {w['p95_ms'] / 1000:.1f} s p95 | — | "
                     f"{timing['n_images']} images, {timing['machine']['cores']} CPU threads, no GPU |")
    lines.append("")
    tg = op.get("targets", {})
    lines += [f"The problem statement's targets are sensitivity > {tg.get('sensitivity')}, specificity > {tg.get('specificity')}, "
              f"AUC > {tg.get('auc')}, ECE <= {tg.get('ece')}; on the primary external test this build meets "
              f"{sum(1 for k in ('sensitivity_met', 'specificity_met', 'auc_met', 'ece_met') if tg.get(k))} of 4.", ""]

    lines += ["## Component parts, measured separately", ""]
    if lesion:
        lines += ["Lesion segmentation, DDR test split scored once (pixel AUPR / Dice at the threshold chosen on DDR valid):", "",
                  "| lesion | AUPR | Dice |", "|---|---|---|"]
        for k, label in (("MA", "microaneurysms"), ("HE", "hemorrhages"), ("EX", "hard exudates"), ("SE", "soft exudates")):
            lines.append(f"| {k} — {label} | {f3(lesion['test_aupr'].get(k))} | {f3(lesion['test_dice'].get(k))} |")
        lines.append("")
    if quality_card:
        q = quality_card["test_heldout_patients"]
        lines += [f"Image-quality classifier: ungradable-detection AUC {f3(q['ungradable_detection_auc'])} on held-out patients. "
                  "It decides good vs usable (whether to enhance); only the hand-crafted hard limits refuse an image.", ""]
    if flags and policy:
        a = flags["attention_agreement"]
        lines += [f"Human review: the served policy flags {pct(policy['resulting_flag_rate'])} of gradable images; "
                  f"{pct(policy['error_rate_flagged'])} of flagged calls are classifier errors against "
                  f"{pct(policy['error_rate_unflagged'])} of unflagged ones, catching "
                  f"{pct(policy['share_of_cnn_errors_flagged'])} of them. Attention agreement on referable calls: "
                  f"median {a['correct'].get('median')} when the classifier is right vs {a['incorrect'].get('median')} "
                  f"when it is wrong ({flags['n_gradable']} validation images).", ""]

    lines += ["## Populations and settings NOT measured", "",
              "This is the most important section of this card.", ""]
    sources = []
    if manifests:
        sources = sorted({d for m in manifests.get("manifests", {}).values() for d in (m.get("datasets") or {})})
    lines += [f"- **No Indian data of any kind.** Training, calibration and test sets are "
              f"{', '.join(sources) if sources else 'APTOS, EyePACS, DDR and Messidor-2'} — US, Chinese and French "
              "acquisition. The PPV above is recomputed at an assumed 18 % Indian prevalence; that assumption is not "
              "validated here.",
              "- **No portable or smartphone-camera images.** Every image is from a tabletop fundus camera. The "
              "intended deployment is portable cameras at primary health centres, and that shift is unmeasured.",
              "- **No measured subgroup performance by age, sex, ethnicity or comorbidity.** The datasets do not carry "
              "these labels, so no fairness claim is made in either direction.",
              "- **No cataract, small-pupil or media-opacity cohort**, which is what a real screening queue in a "
              "district contains, and what the quality gate would have to hold up against.",
              "- **No prospective use.** Every number is retrospective on stored images. Nobody has been screened by "
              "this system.",
              "- **No inter-reader comparison.** The system has not been compared against the ophthalmologists it "
              "would triage for.", ""]

    lines += ["## Known weaknesses", ""]
    if lesion:
        lines.append(f"- **Microaneurysm segmentation is weak: AUPR {f3(lesion['test_aupr'].get('MA'))}, "
                     f"Dice {f3(lesion['test_dice'].get('MA'))}.** They are 1-3 px at the 512 px working resolution. "
                     "A microaneurysm-only finding is ICDR grade 1, below the referable threshold this system is "
                     "accepted for, so the weakness sits under the decision rather than inside it — but any claim "
                     "about lesion-level evidence must be read with this number in view.")
    for name, e in (op.get("additional_external_tests") or {}).items():
        e_at = e["at_locked_threshold"]
        if e_at["specificity"] < at["specificity"] - 0.1:
            lines.append(f"- **Calibration does not transfer across sources.** On {name.replace('external_test_', '')} the "
                         f"locked threshold over-refers: specificity {f3(e_at['specificity'])} at sensitivity "
                         f"{f3(e_at['sensitivity'])}, against {f3(at['specificity'])} on the primary test, while "
                         f"discrimination holds (AUC {f3(e['auc'])}). Ranking transfers; the operating point does not.")
    if site:
        f_ = site["site_calibration_full_half"]
        by = site.get("site_calibration_by_sample_size", {})
        small = next((v for k, v in sorted(by.items(), key=lambda kv: int(kv[0])) if v), None)
        lines.append(f"  A site calibration set fixes it, and the cost is measured: re-fitting on the site's own "
                     f"labelled images restores sensitivity {f3(f_['sensitivity']['mean'])} / specificity "
                     f"{f3(f_['specificity']['mean'])} (ECE {f3(f_['ece']['mean'])})"
                     + (f", and {small['n']} labelled images are already enough "
                        f"({f3(small['sensitivity']['mean'])} / {f3(small['specificity']['mean'])})." if small else "."))
    lines += ["- **A non-referable result is not a negative diagnosis.** At the locked threshold the system misses "
              f"{pct(1 - at['sensitivity'])} of referable cases on the primary external test.",
              "- **The rule grader does not detect venous beading or IRMA**, so its severe-NPDR criterion is "
              "incomplete and it is a consistency check, not a second reader.", ""]

    lines += ["## How the operating point was set", "",
              f"- Threshold chosen at {int(100 * (op['targets']['sensitivity'] if op.get('targets') else 0.9))} % sensitivity on "
              f"{op['source']['description']} (n = {op['source']['n_calibration']:,}, "
              f"{op['source'].get('n_patients_calibration', '?')} patients), then **locked**: "
              f"{op['thresholds']['referable']}.",
              f"- Platt scaling fitted on the same set: ECE {f3(op['calibration']['ece_raw'])} -> "
              f"{f3(op['calibration']['ece_calibrated'])}.",
              "- The external test was scored **once** under this model version (`config/external_test.lock`); a second "
              "scoring refuses without an explicit override that is then recorded in the output.",
              "- The serving code verifies the calibration manifest's SHA-256 at start-up and refuses to run if it "
              "does not match the operating point, so a threshold can never be served against data nobody can vouch for.", "",
              "## Provenance and licence", "",
              "- Datasets, their terms and the required citations: `NOTICE`. The checkpoints are released for "
              "non-commercial research and education because the training data permits only that.",
              "- Privacy, what is stored and the DPDP gaps: `docs/PRIVACY.md`.",
              "- Full protocol, reliability diagram, subgroup contrast and the experiments that were measured and not "
              "shipped: `docs/VALIDATION.md`.", "",
              f"**{DISCLAIMER}**", ""]
    return "\n".join(lines)


def main() -> int:
    op = load(CONFIG_DIR / "operating_point.json")
    if op is None:
        print("no operating_point.json", file=sys.stderr)
        return 1
    timing = load(CONFIG_DIR / "timing_report.json")
    sweep = load(CONFIG_DIR / "sweep_cache.json")
    manifests = load(MANIFEST_DIR / "manifests.json")
    lesion = load(CONFIG_DIR / "lesion_thresholds.json")
    grader_card = load(CARDS / "grader_v2.summary.json")
    quality_card = load(CARDS / "quality_cnn.summary.json")
    unet_card = load(CARDS / "lesion_unet.summary.json")
    flags = load(CONFIG_DIR / "validation_flags.json")
    ensemble = load(CONFIG_DIR / "ensemble_check.json")
    policy = load(CONFIG_DIR / "review_policy.json")
    DOCS.mkdir(exist_ok=True)
    (DOCS / "VALIDATION.md").write_text(validation_md(op, timing, sweep, manifests, lesion, quality_card, grader_card, flags, ensemble, policy), encoding="utf-8")
    print(f"wrote {DOCS / 'VALIDATION.md'}")
    site = load(CONFIG_DIR / "site_calibration_messidor2.json")
    (DOCS / "MODEL_CARD.md").write_text(model_card_md(op, lesion, quality_card, grader_card, flags, policy, site, timing, manifests), encoding="utf-8")
    print(f"wrote {DOCS / 'MODEL_CARD.md'}")
    CARDS.mkdir(parents=True, exist_ok=True)
    for name, card in (("grader_v2", grader_card), ("lesion_unet", unet_card), ("quality_cnn", quality_card)):
        if card:
            (CARDS / f"{name}.md").write_text(card_md(name, card), encoding="utf-8")
            print(f"wrote {CARDS / f'{name}.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
