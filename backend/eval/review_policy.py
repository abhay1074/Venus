"""Choose the human-review policy parameters on the validation sample.

    python -m backend.eval.review_policy

Reads backend/config/validation_flags.json (backend.eval.flag_rate output) and
writes backend/config/review_policy.json with:

    abstain_band_logit   half-width of the abstain band around the locked
                         threshold, in logit space (the architecture's ±0.05
                         in probability is asymmetric at a threshold near 0.1;
                         0.35 logit ≈ ±0.05 probability at p = 0.5)
    attention_floor      referable CNN calls whose attention agreement is below
                         this are flagged; chosen as the cut where the error
                         rate among flagged calls stays >= 60 % while flagging
                         as many calls as possible (validation)

and the flag rate the combined policy produces on the same sample, so the
district simulation and the docs use a number that matches the served rule.
The external test is never touched here.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import numpy as np

from backend.venus.config import CONFIG_DIR, OPERATING_POINT_PATH


def logit(x):
    x = np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--flags", default=str(CONFIG_DIR / "validation_flags.json"), help="flag_rate output to choose from (an experiment's file, if not the served one)")
    parser.add_argument("--out", default=str(CONFIG_DIR / "review_policy.json"), help="where to write; default is the served policy")
    args = parser.parse_args(argv)
    v = json.load(open(args.flags, encoding="utf-8"))
    op = json.load(open(OPERATING_POINT_PATH, encoding="utf-8"))
    t = op["thresholds"]["referable"]
    rows = [r for r in v["rows"] if r["accepted"]]
    p = np.array([r["p_referable"] for r in rows]); y = np.array([r["grade"] >= 2 for r in rows]); pred = p >= t
    err = y != pred
    dis = np.array([abs(r["cnn_grade"] - r["rule_grade"]) >= 2 for r in rows])
    nol = np.array([r["referable"] and sum(r["counts"].values()) == 0 for r in rows])
    att = np.array([r["attention"] if r["attention"] is not None else np.nan for r in rows])
    counted = np.array([sum(r["counts"].values()) > 0 for r in rows])

    band_logit = 0.35
    band = np.abs(logit(p) - logit(t)) <= band_logit

    referable_calls = pred & ~np.isnan(att) & counted
    candidates = {}
    for cut in np.round(np.arange(0.10, 0.41, 0.05), 2):
        f = referable_calls & (att < cut)
        if f.sum() == 0:
            continue
        candidates[float(cut)] = {"flagged_share_of_referable_calls": round(float(f[referable_calls].mean()), 4),
                                  "error_rate_flagged": round(float(err[f].mean()), 4),
                                  "error_rate_unflagged": round(float(err[referable_calls & ~f].mean()), 4)}
    eligible = [c for c, s in candidates.items() if s["error_rate_flagged"] >= 0.60]
    attention_floor = max(eligible) if eligible else 0.15
    att_flag = referable_calls & (att < attention_floor)

    flag = dis | nol | band | att_flag
    policy = {
        "written_at": datetime.now(timezone.utc).isoformat(), "chosen_on": f"validation sample, n = {len(rows)} gradable ({v['manifest']})",
        "model_version": op["model_version"], "grader": op["grader_tag"],
        "abstain_band_logit": band_logit, "attention_floor": attention_floor,
        "attention_candidates": candidates,
        "resulting_flag_rate": round(float(flag.mean()), 4),
        "flag_reasons": {"disagreement": int(dis.sum()), "no_lesion": int(nol.sum()), "abstain": int(band.sum()), "attention": int(att_flag.sum())},
        "error_rate_flagged": round(float(err[flag].mean()), 4), "error_rate_unflagged": round(float(err[~flag].mean()), 4),
        "share_of_cnn_errors_flagged": round(float(flag[err].mean()), 4),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(policy, handle, indent=2)
    print(json.dumps({k: v for k, v in policy.items() if k != "attention_candidates"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
