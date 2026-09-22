"""screen_image(bytes) -> result: the end-to-end entry point.

    Stage 0 rejects or enhances before any model sees the image.
    Stages 1 and 2 run on the same accepted image.
    Fusion compares the CNN grade with the lesion-rule grade.
    Stage 3 produces the evidence and the report.
    Stage 5 turns the result into a priority tier.

A rejected image returns early with the operator-facing reason and a P0
(retake) tier; nothing diagnostic runs on it.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

from backend.venus import report as report_module
from backend.venus import stage0_gate, stage1_segment, stage2_grade, stage3_explain, stage5_schedule
from backend.venus.config import operating_point
from backend.venus.imaging import decode_image, to_png_base64

logger = logging.getLogger(__name__)

RECOMMENDATIONS = {
    "P0": "Image not gradable. Retake on the same visit following the reason shown to the operator.",
    "P1": "Urgent: proliferative-stage evidence or sight-threatening symptoms. Book an in-person ophthalmology "
          "appointment at the district hospital within 7 days.",
    "P2": "Referable diabetic retinopathy. Book an ophthalmology or tele-review appointment within 30 days. "
          "This is a screening result read alongside the evidence on this page, not a diagnosis.",
    "P3": "The two graders or the confidence band disagree. A human reader reviews this image within 3 days with "
          "both grades and the evidence attached, then the case is re-tiered.",
    "P4": "No referable diabetic retinopathy found on this image. Routine recall; sooner if symptoms appear. "
          "A negative screen does not exclude disease: this system misses roughly 1 in 11 referable cases at its operating point.",
}


def warm_up() -> dict:
    """Load both networks so the first real request is not the slow one."""
    started = time.perf_counter()
    gate = stage0_gate.load_gate()
    grader = stage2_grade.load_grader()
    stage0_gate.load_quality()
    stage1_segment.load_unet()
    point = operating_point()
    if gate is not None and grader is not None:
        # One dummy pass through each network and through the compiled Grad-CAM
        # function: TensorFlow traces graphs on first use, and that trace
        # (10-15 s on a laptop CPU) must not land on the first patient.
        import numpy as np
        side = int(grader.input_shape[1])
        blank = np.zeros((side, side, 3), np.float32)
        stage0_gate.modality_check(np.zeros((256, 256, 3), np.uint8))
        grader.predict(blank[None, ...], verbose=0)
        stage3_explain.gradcam(blank, [1])
    return {"gate": stage0_gate.gate_status(), "grader": stage2_grade.grader_status(),
            "quality_cnn": stage0_gate.quality_status(), "lesion_unet": stage1_segment.unet_status(),
            "operating_point": {"threshold": point["thresholds"]["referable"],
                                "fingerprint": point["calibration_fingerprint"]},
            "warm_up_ms": int((time.perf_counter() - started) * 1000)}


def screen_image(payload: bytes, intake: dict | None = None, tta: bool = False,
                 write_report: bool = True, persist: bool = True) -> dict:
    started = time.perf_counter()
    session_id = f"VS-{uuid.uuid4().hex[:10].upper()}"
    captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    point = operating_point()
    image = decode_image(payload)

    s0 = stage0_gate.run(image)
    if not s0["accepted"]:
        tier = stage5_schedule.priority_tier(None, intake)
        total = int((time.perf_counter() - started) * 1000)
        result = {
            "session_id": session_id, "captured_at": captured_at, "model_version": point["model_version"],
            "calibration_fingerprint": point["calibration_fingerprint"],
            "accepted": False, "stop_reason": s0["stop_reason"],
            "stage0": {k: v for k, v in s0.items() if k not in ("image", "mask", "original")},
            "stage5": tier,
            "recommendation": RECOMMENDATIONS["P0"],
            "timing_ms": {"stage0": s0["elapsed_ms"], "total": total},
            "images": {"original": to_png_base64(s0["original"]) if "original" in s0 else None},
        }
        if persist:
            stage5_schedule.init_db()
            stage5_schedule.save_screening(result)
        return result

    s1 = stage1_segment.run(s0["image"], s0["mask"], original=s0["original"], raw=image)
    s2 = stage2_grade.run(image, s1, tta=tta, stage0_image=s0["original"])
    s3 = stage3_explain.run(s0, s1, s2)

    # Attention agreement is the third review signal; it joins the fusion flags
    # here because it needs Stage 3 to exist.
    fusion = s2["fusion"]
    if s3["attention_agreement"]["flag"]:
        fusion["flag_reasons"].append(
            f"attention agreement {s3['attention_agreement']['score']:.2f}: Grad-CAM mass is not on the detected lesions")
        fusion["flag_for_review"] = True

    partial = {"accepted": True, "stage2": s2}
    tier = stage5_schedule.priority_tier(partial, intake)
    total = int((time.perf_counter() - started) * 1000)

    result = {
        "session_id": session_id,
        "captured_at": captured_at,
        "model_version": point["model_version"],
        "calibration_fingerprint": point["calibration_fingerprint"],
        "accepted": True,
        "stop_reason": None,
        "stage0": {k: v for k, v in s0.items() if k not in ("image", "mask", "original")},
        "stage1": {k: v for k, v in s1.items() if k != "masks"},
        "stage2": {k: v for k, v in s2.items() if k != "grader_input"},
        "stage3": {k: v for k, v in s3.items() if k != "heat_referable"},
        "stage5": tier,
        "recommendation": RECOMMENDATIONS[tier["tier"]],
        "timing_ms": {
            "stage0": s0["elapsed_ms"], "stage1": s1["elapsed_ms"], "stage2": s2["elapsed_ms"],
            "stage3": s3["elapsed_ms"], "gradcam": s3["gradcam_ms"], "total": total,
        },
    }
    if write_report:
        # A full disk or an unwritable reports directory must not destroy a
        # finished clinical result: the grade and its evidence are returned
        # either way, with report=None and a stated reason the caller can show.
        # /report/<id>.pdf then answers 507 rather than a bare 404.
        try:
            result["report"] = report_module.write_report(result)
        except OSError as exc:
            result["report"] = None
            result["report_error"] = f"the report could not be written ({exc.strerror or exc}); the result above is complete and was still recorded"
            logger.error("report write failed for %s: %s", session_id, exc)
        result["timing_ms"]["report"] = int((time.perf_counter() - started) * 1000) - total
    if persist:
        stage5_schedule.init_db()
        stage5_schedule.save_screening(result, (intake or {}).get("patient_id"))
    return result
