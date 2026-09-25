"""The README's numbers table must agree with the JSON the code wrote.

docs/VALIDATION.md and docs/MODEL_CARD.md are generated, so they cannot drift.
The README's "Measured, not claimed" table is written by hand for readability,
which is exactly how a number goes stale after a re-measurement. Each check
below formats the value from its JSON the way the README prints it and asserts
that text is in the README row. When one fails, the JSON is right and the
README gets fixed.

Run:  python -m pytest backend/tests/test_docs_consistency.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "backend" / "config"


def load(name):
    path = CONFIG / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def readme_rows() -> list[str]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index("## Measured, not claimed")
    end = text.index("## ", start + 5)
    return [line for line in text[start:end].splitlines() if line.startswith("|")]


def row(rows, starts_with: str) -> str:
    hits = [r for r in rows if r.lower().startswith(("| " + starts_with).lower()) or starts_with.lower() in r.lower()[:120]]
    assert hits, f"no README row for {starts_with!r}"
    return hits[0]


def f3(x):
    return f"{x:.3f}"


def pct1(x):
    return f"{100 * x:.1f} %"


def expect(text: str, *needles: str):
    missing = [n for n in needles if n not in text]
    assert not missing, f"README row is stale: expected {missing} in\n  {text}"


def test_primary_external_test(readme_rows):
    t = load("operating_point.json")["external_test"]; at = t["at_locked_threshold"]; ci = t["ci95_bootstrap_2000"]
    expect(row(readme_rows, "AUC |"), f3(t["auc"]), f"[{f3(ci['auc'][0])}, {f3(ci['auc'][1])}]", f"{t['n']:,} images")
    expect(row(readme_rows, "Sensitivity / specificity at the locked"), f3(at["sensitivity"]), f3(at["specificity"]),
           f"[{f3(ci['sensitivity'][0])}, {f3(ci['sensitivity'][1])}]", f"[{f3(ci['specificity'][0])}, {f3(ci['specificity'][1])}]")
    expect(row(readme_rows, "PPV / NPV"), f3(at["ppv_at_indian_prevalence"]), f3(at["npv_at_indian_prevalence"]), f3(t["ece"]))


def test_messidor_and_secondary(readme_rows):
    op = load("operating_point.json")
    m = op["additional_external_tests"]["external_test_messidor2"]; mat = m["at_locked_threshold"]
    expect(row(readme_rows, "Messidor-2 (adjudicated"), f3(m["auc"]), f3(mat["sensitivity"]), f3(mat["specificity"]),
           f"{m['n']:,} images", f"{m['n_patients']} patients")
    s = op["secondary_heldout"]; sat = s["at_locked_threshold"]
    expect(row(readme_rows, "Held-out EyePACS"), f3(s["auc"]), f3(sat["sensitivity"]), f3(sat["specificity"]), f"{s['n']:,} images")


def test_site_calibration(readme_rows):
    d = load("site_calibration_messidor2.json")
    L = d["locked_operating_point"]; s = d["site_calibration_by_sample_size"]["400"]
    two = lambda v: f"{v:.2f}"  # noqa: E731
    expect(row(readme_rows, "**What a site calibration"),
           f3(L["sensitivity"]["mean"]), f3(L["specificity"]["mean"]), f3(L["ece"]["mean"]),
           f"sens {two(s['sensitivity']['mean'])} [{two(s['sensitivity']['p5'])}, {two(s['sensitivity']['p95'])}]",
           f"spec {two(s['specificity']['mean'])} [{two(s['specificity']['p5'])}, {two(s['specificity']['p95'])}]",
           f"ECE {f3(s['ece']['mean'])}")


def test_lesion_unet(readme_rows):
    t = load("lesion_thresholds.json")
    text = row(readme_rows, "Lesion U-Net")
    for k in ("HE", "EX", "SE", "MA"):
        expect(text, f"{k} {t['test_aupr'][k]:.2f} / {t['test_dice'][k]:.2f}")


def test_landmarks(readme_rows):
    h = load("experiments/landmark_check.json")["held_out_half"]
    expect(row(readme_rows, "Optic disc / fovea"), pct1(h["flatfield"]["fovea_within_1dd"]).replace(" %", " %"),
           pct1(h["old"]["fovea_within_1dd"]))


def test_review_and_attention(readme_rows):
    v = load("validation_flags.json"); p = load("review_policy.json"); a = v["attention_agreement"]
    expect(row(readme_rows, "Attention agreement"), f"{a['correct']['median']:.2f}", f"{a['incorrect']['median']:.2f}",
           f"{v['n_gradable']} validation images")
    expect(row(readme_rows, "Human-review flag rate"), pct1(p["resulting_flag_rate"]), pct1(v["retake_rate"]),
           f"{v['n_gradable']} raw validation images", pct1(p["error_rate_flagged"]), pct1(p["error_rate_unflagged"]))


def test_timing_and_district(readme_rows):
    w = load("timing_report.json")["per_stage"]["wall_ms"]
    expect(row(readme_rows, "End-to-end time"), f"median {w['median_ms'] / 1000:.1f} s", f"p95 {w['p95_ms'] / 1000:.1f} s")
    s = load("sweep_cache.json")["slide_numbers"]
    expect(row(readme_rows, "District:"), f"{s['doctors_with_ai']} with AI vs {s['doctors_without_ai']} without",
           f"₹{s['cost_with_ai'] / 1e7:.2f} Cr vs ₹{s['cost_without_ai'] / 1e7:.2f} Cr")
