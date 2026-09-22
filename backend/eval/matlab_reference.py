"""Python decisions on the shipped samples, for the MATLAB cross-check.

    python -m backend.eval.matlab_reference     -> matlab/tests/reference/python_samples.json

matlab/tests/CrossCheckTest.m screens the same files with drscreen.screenImage
and asserts the decisions agree: accepted, referable, human-review flag, tier,
CNN grade, rule grade within one, and P(referable) within 0.05 - the networks
are numerically identical on identical inputs (<= 2e-6, tests/reference/*.mat);
what differs between the two implementations is image primitives (resize,
bilateral filter, morphology), and this is the tolerance they leave.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from backend.venus import pipeline
from backend.venus.config import PROJECT_ROOT, SAMPLES_DIR

OUT = PROJECT_ROOT / "matlab" / "tests" / "reference" / "python_samples.json"
SAMPLES = ["normal_right_eye.jpg", "npdr_hemorrhages_nei.jpg", "pdr_proliferative_nei.jpg", "dr_exudates.png",
           "demo_fundus.jpeg", "usable_dark_vignetted.jpg", "retake_blurred.jpg", "not_fundus_dermoscopy.jpg"]


def main() -> int:
    rows = {}
    for name in SAMPLES:
        path = SAMPLES_DIR / name
        if not path.exists():
            continue
        r = pipeline.screen_image(path.read_bytes(), persist=False, write_report=False)
        row = {"accepted": r["accepted"], "stop_reason": r.get("stop_reason"), "tier": r["stage5"]["tier"]}
        if r["accepted"]:
            f = r["stage2"]["fusion"]; L = r["stage1"]["lesions"]
            row.update({"quality": r["stage0"]["quality"]["label"], "cnn_grade": r["stage2"]["cnn"]["grade"], "rule_grade": r["stage2"]["rule"]["grade"],
                        "p_referable": f["p_referable"], "referable": f["referable"], "flag": f["flag_for_review"],
                        "attention": r["stage3"]["attention_agreement"]["score"], "lesions": {k: L[k]["count"] for k in ("MA", "HE", "EX", "SE")}})
        rows[name] = row
        print(name, row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump({"written_at": datetime.now(timezone.utc).isoformat(), "samples": rows}, handle, indent=2)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
