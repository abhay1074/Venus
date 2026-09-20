"""Check everything the live demo depends on, before the demo.

    python scripts/preflight.py [--api http://127.0.0.1:8000]
"""
import argparse
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--origin", default="http://127.0.0.1:5173")
    args = parser.parse_args()
    ok = True

    def check(label, passed, detail=""):
        nonlocal ok
        ok = ok and passed
        print(f"  {'PASS' if passed else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")

    c = httpx.Client(base_url=args.api, timeout=120)
    try:
        h = c.get("/health").json()
    except Exception as exc:
        print(f"  FAIL  API unreachable at {args.api}: {exc}")
        return 1
    check("API status ok", h["status"] == "ok", h["status"])
    check("modality gate loaded", h["gate"]["loaded"], str(h["gate"]["error"] or ""))
    check("CNN grader loaded", h["grader"]["loaded"], f"{h['grader'].get('tag', '')} {h['grader']['error'] or ''}".strip())
    print(f"  info  quality CNN {'loaded' if h.get('quality_cnn', {}).get('loaded') else 'absent (handcrafted quality)'}; "
          f"lesion U-Net {'loaded' if h.get('lesion_unet', {}).get('loaded') else 'absent (classical detectors)'}")
    check("operating point verified against calibration fingerprint", h["operating_point"]["ok"])
    check("district sweep cached", h["sweep"]["cached"], "running" if h["sweep"]["running"] else "")

    r = c.options("/screen", headers={"Origin": args.origin, "Access-Control-Request-Method": "POST"})
    check("CORS preflight from the front end", r.status_code == 200 and "access-control-allow-origin" in r.headers)

    sample = ROOT / "samples" / "dr_exudates.png"
    t = time.perf_counter()
    r = c.post("/screen", files={"file": (sample.name, sample.read_bytes(), "image/png")}, headers={"Origin": args.origin})
    ms = int((time.perf_counter() - t) * 1000)
    body = r.json()
    check("referable sample screens end to end", r.status_code == 200 and body.get("accepted"), f"{ms} ms")
    check("result is referable with both grades present", body.get("stage2", {}).get("fusion", {}).get("referable") is True)
    check("PDF report served", c.get(f"/report/{body.get('session_id')}.pdf").status_code == 200)

    derm = ROOT / "samples" / "not_fundus_dermoscopy.jpg"
    r = c.post("/screen", files={"file": (derm.name, derm.read_bytes(), "image/jpeg")})
    check("dermoscopy refused by the modality gate", r.status_code == 200 and r.json().get("accepted") is False)

    r = c.post("/simulate", json={"params": {"ophthalmologists": 4}})
    check("district simulation runs", r.status_code == 200 and "delta" in r.json(), f"{r.json().get('elapsed_ms')} ms")

    try:
        w = httpx.get(args.origin, timeout=10)
        check("front end reachable", w.status_code == 200)
    except Exception:
        check("front end reachable", False, "start it with: cd frontend && npm run dev")

    print("\nall checks passed" if ok else "\nsome checks failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
