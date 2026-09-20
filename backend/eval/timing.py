"""End-to-end timing run on this machine's CPU.

The problem statement asks for a result in under 30 s per image on the kind
of laptop a PHC has. This runs the shipped sample images round-robin until
`--images` screenings have been timed (default 50), reports per-stage medians
and the 95th percentile of the total, and writes config/timing_report.json so
the number in the submission is a measured one.

Run:  python -m backend.eval.timing --images 50
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from backend.venus import pipeline
from backend.venus.config import CONFIG_DIR, MODEL_VERSION, SAMPLES_DIR

GRADABLE = ["normal_right_eye.jpg", "dr_exudates.png", "pdr_proliferative_nei.jpg",
            "npdr_hemorrhages_nei.jpg", "usable_dark_vignetted.jpg", "demo_fundus.jpeg"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=50)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args(argv)

    warm = pipeline.warm_up()
    print(f"warm-up {warm['warm_up_ms']} ms  ({platform.processor() or platform.machine()}, {os.cpu_count()} cores)")
    payloads = [(name, (SAMPLES_DIR / name).read_bytes()) for name in GRADABLE if (SAMPLES_DIR / name).exists()]
    if not payloads:
        print("no sample images found", file=sys.stderr)
        return 1

    rows = []
    for i in range(args.images):
        name, payload = payloads[i % len(payloads)]
        started = time.perf_counter()
        result = pipeline.screen_image(payload, tta=args.tta, write_report=True, persist=False)
        wall = int((time.perf_counter() - started) * 1000)
        t = result["timing_ms"]
        rows.append({"image": name, "wall_ms": wall, **{k: t.get(k) for k in ("stage0", "stage1", "stage2", "stage3", "gradcam", "report", "total")}})
        print(f"{i + 1:3d} {name:28s} {wall:6d} ms  (S0 {t['stage0']}, S1 {t['stage1']}, S2 {t['stage2']}, S3 {t['stage3']}, report {t.get('report')})")

    def stat(key):
        v = np.array([r[key] for r in rows if r[key] is not None], float)
        return {"median_ms": int(np.median(v)), "p95_ms": int(np.percentile(v, 95)), "max_ms": int(v.max())}

    report = {
        "model_version": MODEL_VERSION,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "machine": {"processor": platform.processor() or platform.machine(), "cores": os.cpu_count(),
                    "platform": platform.platform(), "python": platform.python_version()},
        "n_images": len(rows), "tta": args.tta, "warm_up_ms": warm["warm_up_ms"],
        "per_stage": {k: stat(k) for k in ("stage0", "stage1", "stage2", "stage3", "gradcam", "report", "wall_ms")},
        "requirement_ms": 30000,
        "requirement_met_p95": stat("wall_ms")["p95_ms"] < 30000,
        "rows": rows,
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_DIR / "timing_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    w = report["per_stage"]["wall_ms"]
    print(f"\nend-to-end: median {w['median_ms']} ms, p95 {w['p95_ms']} ms, max {w['max_ms']} ms over {len(rows)} images "
          f"-> 30 s requirement {'MET' if report['requirement_met_p95'] else 'MISSED'} at p95")
    print(f"wrote {CONFIG_DIR / 'timing_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
