"""Slide figures, drawn from the same JSON the docs are generated from.

    python -m backend.eval.figures            -> docs/figures/*.png (+ .svg)

Every number on a slide should trace to config/*.json; these figures do, so a
re-run after any model change refreshes the deck's pictures the way
backend.eval.write_docs refreshes its text. Nothing is computed here.

    roc_operating_points.png   the external-test ROC points fed to the sweep, the locked point marked
    reliability.png            calibration: predicted vs observed referable rate per bin, ECE in the title
    referral_by_grade.png      referred fraction by true ICDR grade (external test)
    attention_agreement.png    attention agreement, correct vs incorrect referable CNN calls (validation)
    pareto.png                 district sweep: cost vs missed referable cases, the Pareto front and the two slide points
    doctors_vs_missed.png      programme sensitivity by ophthalmologist count, AI vs human-only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from backend.venus.config import CONFIG_DIR, PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "docs" / "figures"
# Categorical slots in fixed order (never cycled); text in ink, not series colour.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlesize": 12, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False, "savefig.dpi": 200,
})


def load(name):
    path = CONFIG_DIR / name
    return json.load(open(path, encoding="utf-8")) if path.exists() else None


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)
    print(f"wrote docs/figures/{name}.png")


def roc_points(op):
    t = op["external_test"]; at = t["at_locked_threshold"]
    pts = sorted(t["roc_points_for_simulation"], key=lambda r: 1 - r["specificity"])
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    ax.plot([1 - p["specificity"] for p in pts], [p["sensitivity"] for p in pts], color=BLUE, linewidth=2, marker="o", markersize=7,
            markerfacecolor="white", markeredgewidth=2, label="operating points fed to the district sweep")
    for p in pts:
        ax.annotate(f"{int(p['sensitivity_target'] * 100)} %", (1 - p["specificity"], p["sensitivity"]), textcoords="offset points", xytext=(8, -10), color=INK2, fontsize=9)
    ax.plot(1 - at["specificity"], at["sensitivity"], marker="o", markersize=11, color=ORANGE, linestyle="none",
            label=f"locked threshold: sens {at['sensitivity']:.3f}, spec {at['specificity']:.3f}")
    ax.set_xlabel("1 − specificity (false-positive rate)"); ax.set_ylabel("sensitivity (referable DR, ICDR ≥ 2)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(f"External test ({t['n']:,} images): AUC {t['auc']:.3f} [{t['ci95_bootstrap_2000']['auc'][0]}, {t['ci95_bootstrap_2000']['auc'][1]}]")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "roc_operating_points")


def reliability(op):
    t = op["external_test"]
    bins = [b for b in t["reliability_diagram"] if b["n"] > 0]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    x = [b["confidence"] for b in bins]; y = [b["accuracy"] for b in bins]; n = np.array([b["n"] for b in bins])
    ax.plot(x, y, color=BLUE, linewidth=2)
    ax.scatter(x, y, s=20 + 300 * n / n.max(), color=BLUE, zorder=3, edgecolor="white", linewidth=1.5)
    for b in bins:
        ax.annotate(f"n={b['n']:,}", (b["confidence"], b["accuracy"]), textcoords="offset points", xytext=(8, -4), color=INK2, fontsize=8)
    ax.set_xlabel("predicted P(referable), bin mean"); ax.set_ylabel("observed referable fraction")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Calibration, external test: ECE {t['ece']:.3f}")
    ax.text(0.02, 0.96, "Platt scaling fitted on the calibration set;" + chr(10) + "marker size = images in the bin", transform=ax.transAxes, va="top", fontsize=9, color=INK2)
    save(fig, "reliability")


def referral_by_grade(op):
    t = op["external_test"]; f = t["referred_fraction_by_true_grade"]
    labels = ["0 none", "1 mild", "2 moderate", "3 severe", "4 PDR"]
    vals = [f[str(k)] for k in range(5)]
    colours = [BLUE if k >= 2 else INK2 for k in range(5)]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bars = ax.bar(labels, vals, color=colours, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v * 100:.0f} %", ha="center", color=INK, fontsize=10)
    ax.axhline(0.9, color=ORANGE, linewidth=1.2, linestyle="--"); ax.text(4.35, 0.905, "90 % target", color=ORANGE, fontsize=9, ha="right")
    ax.set_ylim(0, 1.08); ax.set_ylabel("fraction referred at the locked threshold"); ax.set_xlabel("true ICDR grade (external test)")
    ax.set_title("Referred fraction by true grade")
    ax.grid(axis="x", visible=False)
    save(fig, "referral_by_grade")


def attention(flags, policy):
    rows = [r for r in flags["rows"] if r["accepted"] and r["attention"] is not None and sum(r["counts"].values()) > 0]
    op = load("operating_point.json"); thr = op["thresholds"]["referable"]
    ref = [r for r in rows if r["p_referable"] >= thr]
    correct = [r["attention"] for r in ref if r["grade"] >= 2]; wrong = [r["attention"] for r in ref if r["grade"] < 2]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bins = np.linspace(0, 1, 21)
    ax.hist(correct, bins=bins, density=True, histtype="stepfilled", color=BLUE, alpha=0.55, edgecolor=BLUE, linewidth=1.5,
            label=f"CNN correct (n = {len(correct)}), median {np.median(correct):.2f}")
    ax.hist(wrong, bins=bins, density=True, histtype="stepfilled", color=ORANGE, alpha=0.55, edgecolor=ORANGE, linewidth=1.5,
            label=f"CNN wrong (n = {len(wrong)}), median {np.median(wrong):.2f}")
    floor = policy["attention_floor"] if policy else 0.15
    ax.axvline(floor, color=INK, linewidth=1.2, linestyle="--")
    ax.text(floor - 0.01, ax.get_ylim()[1] * 0.95, f"review floor {floor}" + chr(10) + "(flag below)", fontsize=9, color=INK, ha="right", va="top")
    ax.set_xlabel("attention agreement: share of Grad-CAM mass on the detected lesions"); ax.set_ylabel("density of referable CNN calls")
    ax.set_title("Where the network looks predicts whether it is right")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "attention_agreement")


def pareto(sweep):
    runs = sweep["runs"]; front = sweep["pareto_front"]; ba = sweep["best_ai"]; bb = sweep["best_baseline"]
    ai = [r for r in runs if r["ai"]]; base = [r for r in runs if not r["ai"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter([r["cost_inr_total"] / 1e7 for r in base], [r["missed_total"] for r in base], s=28, color=INK2, alpha=0.6, label="human-only reading (1–8 ophthalmologists)")
    ax.scatter([r["cost_inr_total"] / 1e7 for r in ai], [r["missed_total"] for r in ai], s=28, color=BLUE, alpha=0.45, label="AI triage (cameras × doctors × operating point)")
    ax.plot([r["cost_inr_total"] / 1e7 for r in front], [r["missed_total"] for r in front], color=BLUE, linewidth=2, label="Pareto front (AI)")
    ymax = max(r["missed_total"] for r in runs)
    label = lambda r, who: f"{who}: {r['ophthalmologists']} ophthalmologists" + "\n" + f"₹{r['cost_inr_total'] / 1e7:.2f} Cr, {r['missed_total']:,} missed"
    for r, c, txt, at in ((ba, ORANGE, label(ba, "with AI"), (0.62, 0.42 * ymax)),
                          (bb, INK, label(bb, "without AI"), (1.62, 0.50 * ymax))):
        ax.scatter([r["cost_inr_total"] / 1e7], [r["missed_total"]], s=150, color=c, zorder=4, edgecolor="white", linewidth=1.5)
        ax.annotate(txt, (r["cost_inr_total"] / 1e7, r["missed_total"]), xytext=at, textcoords="data", ha="center", color=c, fontsize=9, fontweight="bold",
                    arrowprops={"arrowstyle": "-", "color": c, "linewidth": 1}, bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "none", "alpha": 0.9})
    c = sweep["constraints"]
    ax.axhline(c["max_missed"], color=ORANGE, linewidth=1, linestyle=":")
    ax.text(ax.get_xlim()[1], c["max_missed"] + 60, f"constraint: missed ≤ {c['max_missed']:,} ({int(c['max_missed_fraction'] * 100)} % of {c['referable_cases']:,} referable)",
            ha="right", fontsize=8, color=ORANGE, bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9})
    ax.set_xlabel("annual cost, ₹ crore (operator + ophthalmologist hours + cameras)"); ax.set_ylabel("referable cases missed in the year")
    ax.set_title(f"100,000-patient district, {len(runs)} full-year simulations")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "pareto")


def doctors_vs_missed(sweep):
    runs = sweep["runs"]
    ai95 = {r["ophthalmologists"]: r for r in runs if r["ai"] and r["cameras_per_phc"] == 1 and r["sensitivity_target"] == sweep["best_ai"]["sensitivity_target"]}
    base = {r["ophthalmologists"]: r for r in runs if not r["ai"] and r["cameras_per_phc"] == 1}
    ks = sorted(set(ai95) & set(base))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ks, [ai95[k]["programme_sensitivity"] for k in ks], color=BLUE, linewidth=2, marker="o", markerfacecolor="white", markeredgewidth=2, label=f"AI triage at the {int(sweep['best_ai']['sensitivity_target'] * 100)} % point")
    ax.plot(ks, [base[k]["programme_sensitivity"] for k in ks], color=INK2, linewidth=2, marker="o", markerfacecolor="white", markeredgewidth=2, label="human-only reading")
    ax.axhline(0.80, color=ORANGE, linewidth=1.2, linestyle="--"); ax.text(ks[0], 0.76, "80 % programme sensitivity", ha="left", color=ORANGE, fontsize=9)
    ax.set_xlabel("ophthalmologists in the district"); ax.set_ylabel("programme sensitivity (referable cases resulted within the year)")
    ax.set_ylim(0, 1); ax.set_xticks(ks)
    ax.set_title("Share of referable cases resulted within the year, by staffing")
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "doctors_vs_missed")


def main() -> int:
    op = load("operating_point.json")
    if op is None:
        print("no operating_point.json", file=sys.stderr); return 1
    roc_points(op); reliability(op); referral_by_grade(op)
    flags, policy = load("validation_flags.json"), load("review_policy.json")
    if flags:
        attention(flags, policy)
    sweep = load("sweep_cache.json")
    if sweep:
        pareto(sweep); doctors_vs_missed(sweep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
