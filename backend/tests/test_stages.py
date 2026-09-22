"""One test class per stage plus an end-to-end timing test.

The tests assert properties — masks inside the FOV, monotone ordinal decode,
the ICDR rule table, fusion flags, scheduler starvation and bumping — rather
than that the code runs, because every bug worth catching here produces
correctly shaped output while being wrong.

Run:  python -m pytest backend/tests -q
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "samples"
# Ask config where the weights actually are (VENUS_WEIGHTS_DIR moves them), so
# the skip tracks the served path and a weights-less run can be reproduced.
from backend.venus.config import GATE_WEIGHTS, GRADER_WEIGHTS  # noqa: E402

HAS_WEIGHTS = GRADER_WEIGHTS.exists() and GATE_WEIGHTS.exists()
needs_weights = pytest.mark.skipif(not HAS_WEIGHTS, reason="trained weights not present")


@pytest.fixture(scope="session")
def healthy_image():
    from backend.venus.imaging import decode_image
    return decode_image((SAMPLES / "normal_right_eye.jpg").read_bytes())


@pytest.fixture(scope="session")
def dr_image():
    from backend.venus.imaging import decode_image
    return decode_image((SAMPLES / "dr_exudates.png").read_bytes())


# ------------------------------------------------------------- Stage 0 --

class TestStage0Gate:
    def test_fov_mask_is_one_filled_component(self, healthy_image):
        from backend.venus.stage0_gate import fov_mask
        mask = fov_mask(healthy_image)
        count, _ = cv2.connectedComponents(mask)
        assert count == 2  # background + one FOV
        # Convex: no holes and no rim bites
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) == 1

    def test_normalise_fov_is_square_512(self, healthy_image):
        from backend.venus.stage0_gate import normalise_fov
        image, mask, geometry = normalise_fov(healthy_image)
        assert image.shape == (512, 512, 3) and mask.shape == (512, 512)
        assert 0.3 < geometry["coverage"] <= 1.0 and geometry["circularity"] > 0.7
        # Outside the FOV is black after normalisation? No: only after run(); check mask coverage instead
        assert cv2.countNonZero(mask) > 0.5 * 512 * 512

    def test_blur_is_rejected_with_reason(self, healthy_image):
        from backend.venus.stage0_gate import normalise_fov, quality_features, quality_label
        blurred = cv2.GaussianBlur(healthy_image, (0, 0), 9)
        image, mask, geometry = normalise_fov(blurred)
        label, score, reason, _ = quality_label(quality_features(image, mask, geometry))
        assert label == "reject" and "blurry" in reason.lower()

    def test_good_image_not_enhanced_usable_is(self, healthy_image):
        from backend.venus.stage0_gate import enhance, normalise_fov, quality_features, quality_label
        image, mask, geometry = normalise_fov(healthy_image)
        label, _, _, _ = quality_label(quality_features(image, mask, geometry))
        assert label == "good"
        enhanced = enhance(image, mask)
        assert enhanced.shape == image.shape
        assert not np.any(enhanced[mask == 0])  # enhancement never bleeds outside the FOV

    @needs_weights
    def test_modality_gate_rejects_noise_and_accepts_fundus(self, healthy_image):
        from backend.venus.stage0_gate import modality_check
        noise = (np.random.default_rng(0).random((300, 300, 3)) * 255).astype(np.uint8)
        assert modality_check(noise)["accepted"] is False
        assert modality_check(healthy_image)["accepted"] is True

    @needs_weights
    def test_modality_gate_rejects_dermoscopy(self):
        from backend.venus.imaging import decode_image
        from backend.venus.stage0_gate import modality_check
        derm = decode_image((SAMPLES / "not_fundus_dermoscopy.jpg").read_bytes())
        assert modality_check(derm)["accepted"] is False


# ------------------------------------------------------------- Stage 1 --

class TestStage1Segment:
    @pytest.fixture(scope="class")
    def seg(self, healthy_image):
        from backend.venus import stage0_gate, stage1_segment
        image, mask, _ = stage0_gate.normalise_fov(healthy_image)
        return stage1_segment.run(image, mask), mask

    def test_optic_disc_inside_fov_with_plausible_radius(self, seg):
        result, mask = seg
        cx, cy = result["optic_disc"]["centre"]
        assert mask[cy, cx] > 0
        assert 15 < result["optic_disc"]["radius"] < 80

    def test_fovea_is_two_to_three_disc_diameters_from_disc(self, seg):
        result, _ = seg
        (cx, cy), (fx, fy) = result["optic_disc"]["centre"], result["fovea"]["centre"]
        dd = 2 * result["optic_disc"]["radius"]
        distance = float(np.hypot(fx - cx, fy - cy))
        assert 1.7 * dd <= distance <= 3.3 * dd

    def test_vessel_fraction_is_physiological(self, seg):
        result, _ = seg
        assert 0.02 <= result["vessels"]["fraction"] <= 0.15

    def test_masks_lie_inside_fov_and_lesions_are_sparse_on_healthy(self, seg):
        result, mask = seg
        for key in ("vessels", "MA", "HE", "EX", "SE"):
            assert not np.any(result["masks"][key][mask == 0])
        assert result["lesions"]["HE"]["count"] <= 1
        assert result["lesions"]["MA"]["count"] <= 2

    def test_quadrant_counts_sum_to_hemorrhages(self):
        from backend.venus.stage1_segment import quadrant_counts
        comps = [{"centroid": [300, 100]}, {"centroid": [400, 256]}, {"centroid": [256, 400]}, {"centroid": [50, 256]}]
        counts = quadrant_counts(comps, [256, 256], 512)
        assert counts == [1, 1, 1, 1]


# ------------------------------------------------------------- Stage 2 --

class TestStage2Grade:
    def _stage1(self, ma=0, he=0, ex=0, se=0, he_area=0.001, quadrants=None, ex_area=0.001):
        return {
            "lesions": {
                "MA": {"count": ma, "area_fraction": 0.001 if ma else 0.0},
                "HE": {"count": he, "area_fraction": he_area if he else 0.0},
                "EX": {"count": ex, "area_fraction": ex_area if ex else 0.0},
                "SE": {"count": se, "area_fraction": 0.001 if se else 0.0},
            },
            "hemorrhages_per_quadrant": quadrants or [he, 0, 0, 0],
        }

    def test_rule_grader_follows_icdr_table(self):
        from backend.venus.stage2_grade import rule_grade
        assert rule_grade(self._stage1(), 0.0)["grade"] == 0
        assert rule_grade(self._stage1(ma=5), 0.0)["grade"] == 1
        assert rule_grade(self._stage1(ma=5, he=3), 0.0)["grade"] == 2
        assert rule_grade(self._stage1(ex=4), 0.0)["grade"] == 2
        assert rule_grade(self._stage1(he=90, quadrants=[22, 25, 21, 22]), 0.0)["grade"] == 3
        assert rule_grade(self._stage1(he=90, quadrants=[22, 25, 21, 5]), 0.0)["grade"] == 2
        assert rule_grade(self._stage1(), 0.8)["grade"] == 4
        assert rule_grade(self._stage1(he=3, he_area=0.06), 0.0)["grade"] == 4

    def test_component_thresholds_ignore_single_specks(self):
        from backend.venus.stage2_grade import rule_grade
        out = rule_grade(self._stage1(ma=1, ex=1, ex_area=0.00005), 0.0)
        assert out["grade"] == 0
        assert any("below component threshold" in t for t in out["trace"])

    def test_calibration_is_monotone_and_bounded(self):
        from backend.venus.stage2_grade import calibrate
        from backend.venus.config import operating_point
        point = operating_point()
        probs = [calibrate(x, point) for x in (1e-6, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.999)]
        assert probs == sorted(probs) and 0.0 < probs[0] and probs[-1] < 1.0

    def test_fusion_flags_disagreement_and_abstain(self):
        from backend.venus.stage2_grade import fuse
        from backend.venus.config import operating_point
        t = operating_point()["thresholds"]["referable"]
        cnn = {"grade": 3, "grade_label": "Severe NPDR", "referable_probability": 0.95}
        rule = {"grade": 0, "counts": {"MA": 0, "HE": 0, "EX": 0, "SE": 0}}
        out = fuse(cnn, rule)
        assert out["flag_for_review"] and out["disagreement_levels"] == 3
        assert any("no lesion" in r for r in out["flag_reasons"])
        cnn2 = {"grade": 2, "grade_label": "Moderate NPDR", "referable_probability": round(t + 0.02, 4)}
        rule2 = {"grade": 2, "counts": {"MA": 3, "HE": 2, "EX": 0, "SE": 0}}
        out2 = fuse(cnn2, rule2)
        assert out2["abstain"] and out2["flag_for_review"] and out2["referable"]
        cnn3 = {"grade": 0, "grade_label": "No DR", "referable_probability": 0.05}
        out3 = fuse(cnn3, {"grade": 0, "counts": {"MA": 0, "HE": 0, "EX": 0, "SE": 0}})
        assert not out3["flag_for_review"] and not out3["referable"]

    def test_operating_point_fingerprint_is_verified(self, tmp_path, monkeypatch):
        from backend.venus import config
        config.operating_point.cache_clear()
        point = json.loads(config.OPERATING_POINT_PATH.read_text(encoding="utf-8"))
        bad = tmp_path / point["calibration_manifest"]
        bad.write_text("tampered", encoding="utf-8")
        monkeypatch.setattr(config, "MANIFEST_DIR", tmp_path)
        with pytest.raises(config.OperatingPointError):
            config.operating_point()
        config.operating_point.cache_clear()

    @needs_weights
    def test_cnn_grader_decodes_ordinal_thresholds(self, dr_image):
        from backend.venus import stage0_gate
        from backend.venus.stage2_grade import cnn_grade
        s0 = stage0_gate.run(dr_image)
        out = cnn_grade(dr_image, stage0_image=s0["original"])
        cum = out["ordinal_thresholds"]
        assert out["grade"] == sum(1 for p in cum if p >= 0.5)
        assert abs(sum(out["grade_probabilities"]) - 1.0) < 1e-3
        assert out["referable_probability"] > 0.5  # the exudate image is referable


# ------------------------------------------------------------- Stage 3 --

class TestStage3Explain:
    def test_attention_agreement_chance_and_lift(self):
        from backend.venus.stage3_explain import attention_agreement
        mask = np.full((512, 512), 255, np.uint8)
        lesion = np.zeros((512, 512), np.uint8)
        cv2.circle(lesion, (256, 256), 6, 255, -1)
        stage1 = {"masks": {"MA": lesion, "HE": np.zeros_like(lesion), "EX": np.zeros_like(lesion), "SE": np.zeros_like(lesion)}}
        heat_on = np.zeros((512, 512), np.float32)
        cv2.circle(heat_on, (256, 256), 20, 1.0, -1)
        on = attention_agreement(heat_on, stage1, mask, counted=1, referable=True)
        assert on["score"] > 0.9 and on["flag"] is False and on["lift"] > 1.5
        heat_off = np.zeros((512, 512), np.float32)
        cv2.circle(heat_off, (100, 100), 20, 1.0, -1)
        off = attention_agreement(heat_off, stage1, mask, counted=1, referable=True)
        assert off["score"] < 0.05 and off["flag"] is True
        # Not referable: the score is reported but never flags.
        assert attention_agreement(heat_off, stage1, mask, counted=1, referable=False)["flag"] is False

    @needs_weights
    def test_gradcam_is_inside_fov_and_normalised(self, dr_image):
        from backend.venus import stage0_gate, stage1_segment, stage2_grade, stage3_explain
        s0 = stage0_gate.run(dr_image)
        s1 = stage1_segment.run(s0["image"], s0["mask"], original=s0["original"], raw=dr_image)
        s2 = stage2_grade.run(dr_image, s1, stage0_image=s0["original"])
        s3 = stage3_explain.run(s0, s1, s2)
        heat = s3["heat_referable"]
        assert heat.shape == (512, 512) and 0.0 <= heat.min() and heat.max() <= 1.0
        assert not np.any(heat[s0["mask"] == 0])
        assert set(s3["overlays"]) >= {"gradcam_referable", "lesions", "vessels", "original", "enhanced"}


# ------------------------------------------------------------- Stage 4 --

class TestStage4Simulate:
    def test_ai_reduces_doctor_load_and_missed_cases_come_from_confusion_matrix(self):
        from backend.venus.stage4_simulate import Params, simulate
        p = Params(sample_fraction=0.1, ai_enabled=True, sensitivity=0.9, specificity=0.6, flag_rate=0.1)
        ai = simulate(p)
        base = simulate(Params(sample_fraction=0.1, ai_enabled=False))
        assert ai["doctor_hours"] < base["doctor_hours"]
        assert base["referable_missed_by_ai"] == 0
        expected_missed = ai["referable_cases"] * (1 - 0.9) * (1 - 0.1)
        assert abs(ai["referable_missed_by_ai"] - expected_missed) < 0.35 * expected_missed + 20

    def test_perfect_ai_misses_nothing(self):
        from backend.venus.stage4_simulate import Params, simulate
        out = simulate(Params(sample_fraction=0.05, sensitivity=1.0, specificity=1.0, flag_rate=0.0))
        assert out["referable_missed_by_ai"] == 0 and out["unnecessary_referrals"] == 0

    def test_more_doctors_shorten_waits(self):
        from backend.venus.stage4_simulate import Params, simulate
        few = simulate(Params(sample_fraction=0.1, ophthalmologists=2))
        many = simulate(Params(sample_fraction=0.1, ophthalmologists=8))
        assert many["wait_capture_to_result_days"]["p95"] <= few["wait_capture_to_result_days"]["p95"]
        assert many["ophthalmologist_utilisation"] <= 1.0 and few["ophthalmologist_utilisation"] <= 1.0

    def test_sweep_pareto_front_is_monotone(self):
        from backend.venus.stage4_simulate import sweep
        out = sweep(cameras=(1,), doctors=(2, 4), sample_fraction=0.04)
        front = out["pareto_front"]
        costs = [r["cost_inr_total"] for r in front]
        missed = [r["missed_total"] for r in front]
        assert costs == sorted(costs) and missed == sorted(missed, reverse=True)


# ------------------------------------------------------------- Stage 5 --

class TestStage5Schedule:
    def _result(self, grade, p, flag=False, nv=0.0):
        return {"accepted": True, "stage2": {"fusion": {"grade": grade, "p_referable": p, "referable": p >= 0.334,
                                                        "flag_for_review": flag, "flag_reasons": ["x"] if flag else []},
                                             "cnn": {"nv_probability": nv}}}

    def test_tiers_follow_the_table(self):
        from backend.venus.stage5_schedule import priority_tier
        assert priority_tier(None)["tier"] == "P0"
        assert priority_tier(self._result(4, 0.9))["tier"] == "P1"
        assert priority_tier(self._result(2, 0.2, nv=0.7))["tier"] == "P1"
        assert priority_tier(self._result(3, 0.9), {"symptoms": ["floaters"]})["tier"] == "P1"
        assert priority_tier(self._result(2, 0.8, flag=True))["tier"] == "P3"
        assert priority_tier(self._result(2, 0.8))["tier"] == "P2"
        assert priority_tier(self._result(0, 0.05))["tier"] == "P4"

    def test_risk_factors_raise_by_one_never_lower(self):
        from backend.venus.stage5_schedule import priority_tier
        assert priority_tier(self._result(0, 0.05), {"pregnant": True})["tier"] == "P2"
        assert priority_tier(self._result(2, 0.8), {"symptoms": ["blurred_vision"]})["tier"] == "P1"
        routine = priority_tier(self._result(0, 0.05), {"hba1c": 9.5})
        assert routine["tier"] == "P4" and routine["deadline_text"] == "6-month recall"
        assert priority_tier(self._result(4, 0.99), {"hba1c": 5})["tier"] == "P1"

    def _slots(self, now, facility_type, per_day, days):
        out = []
        for d in range(1, days + 1):
            for k in range(per_day):
                t = now + timedelta(days=d, hours=k)
                out.append({"id": f"{facility_type}-{d}-{k}", "facility_id": facility_type, "starts_at": t, "booked": 0, "appointment_id": None})
        return out

    def test_allocator_books_earliest_within_deadline_and_ages_lower_tiers(self):
        from backend.venus.stage5_schedule import allocate
        now = datetime(2026, 9, 20, 9, tzinfo=timezone.utc)
        facilities = {"clinic": {"type": "clinic", "pos": (0, 0)}}
        slots = self._slots(now, "clinic", 1, 40)
        fresh_p2 = {"id": "A", "tier": "P2", "risk_score": 0.9, "queued_at": now, "deadline_days": 30, "village_pos": (0, 0)}
        old_p3 = {"id": "B", "tier": "P3", "risk_score": 0.4, "queued_at": now - timedelta(days=5), "deadline_days": 3, "village_pos": (0, 0)}
        decisions = allocate([fresh_p2, old_p3], slots, now, facilities)
        by_id = {d["appointment_id"]: d for d in decisions}
        # The overdue P3 takes the first slot even though P2 has a higher risk score.
        assert by_id["B"]["starts_at"] < by_id["A"]["starts_at"]

    def test_allocator_bumps_lower_priority_when_deadline_is_full(self):
        from backend.venus.stage5_schedule import allocate
        now = datetime(2026, 9, 20, 9, tzinfo=timezone.utc)
        facilities = {"district_hospital": {"type": "district_hospital", "pos": (0, 0)}}
        slots = self._slots(now, "district_hospital", 1, 40)
        # Fill the first 7 days with routine-ish P2 bookings (allowed at the hospital).
        for d in range(7):
            holder = {"id": f"H{d}", "tier": "P2", "queued_at": now, "deadline_days": 30}
            slots[d]["booked"], slots[d]["appointment_id"], slots[d]["holder"] = 1, holder["id"], holder
        urgent = {"id": "U", "tier": "P1", "risk_score": 0.99, "queued_at": now, "deadline_days": 7, "village_pos": (0, 0)}
        decision = allocate([urgent], slots, now, facilities)[0]
        assert decision["slot_id"] is not None and decision["bumped"] is not None
        assert decision["starts_at"] <= now + timedelta(days=7)
        bumped = decision["bumped"]
        assert bumped["starts_at"] > decision["starts_at"]  # re-slotted later, within its own deadline


# ---------------------------------------------------------- end to end --

@needs_weights
class TestEndToEnd:
    def test_pipeline_under_thirty_seconds_and_traceable(self, tmp_path, monkeypatch):
        from backend.venus import config, pipeline, report as report_module
        monkeypatch.setattr(report_module, "REPORT_DIR", tmp_path)
        pipeline.warm_up()
        payload = (SAMPLES / "dr_exudates.png").read_bytes()
        result = pipeline.screen_image(payload, persist=False)
        assert result["accepted"] and result["timing_ms"]["total"] < 30_000
        assert result["model_version"] == config.MODEL_VERSION
        assert result["calibration_fingerprint"] == config.operating_point()["calibration_fingerprint"]
        assert (tmp_path / result["report"]["pdf"]).stat().st_size > 10_000
        sidecar = json.loads((tmp_path / result["report"]["json"]).read_text(encoding="utf-8"))
        assert sidecar["stage2"]["fusion"]["p_referable"] == result["stage2"]["fusion"]["p_referable"]

    def test_rejected_image_runs_nothing_diagnostic(self):
        from backend.venus import pipeline
        payload = (SAMPLES / "retake_blurred.jpg").read_bytes()
        result = pipeline.screen_image(payload, persist=False)
        assert result["accepted"] is False and "stage2" not in result
        assert result["stage5"]["tier"] == "P0" and result["stop_reason"]
