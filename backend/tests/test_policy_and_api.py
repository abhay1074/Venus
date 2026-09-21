"""The validation-chosen review policy, the two-network lesion switch, the
streaming AUPR used to score the U-Nets, and the read-only API endpoints that
feed the Validation screen.

Run:  python -m pytest backend/tests -q
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]


def _policy(monkeypatch, **fields):
    """Serve a specific review policy to the stage code for one test."""
    from backend.venus import config
    policy = {"abstain_band_logit": None, "abstain_band": 0.05, "attention_floor": 0.15, "attention_min_lift": 1.5, "chosen_on": "test", **fields}
    monkeypatch.setattr(config, "review_policy", lambda: policy)
    from backend.venus import stage2_grade, stage3_explain
    monkeypatch.setattr(stage2_grade, "review_policy", lambda: policy)
    monkeypatch.setattr(stage3_explain, "review_policy", lambda: policy)
    return policy


class TestReviewPolicy:
    def test_logit_band_is_symmetric_in_logit_space(self, monkeypatch):
        """Threshold ~0.1, half-width 0.35 logit: 0.13 abstains, 0.06 does not,
        although 0.06 would under the architecture's ±0.05 probability band."""
        from backend.venus import config, stage2_grade
        point = dict(config.operating_point()); point["thresholds"] = {**point["thresholds"], "referable": 0.1}
        monkeypatch.setattr(stage2_grade, "operating_point", lambda: point)
        _policy(monkeypatch, abstain_band_logit=0.35, attention_min_lift=None)
        rule = {"grade": 2, "counts": {"MA": 3, "HE": 2, "EX": 0, "SE": 0}}
        hi = stage2_grade.fuse({"grade": 2, "grade_label": "Moderate NPDR", "referable_probability": 0.13}, rule)
        lo = stage2_grade.fuse({"grade": 1, "grade_label": "Mild NPDR", "referable_probability": 0.06}, rule)
        assert hi["abstain"] and not lo["abstain"]
        assert abs(hi["abstain_low"] - 0.0726) < 1e-3 and abs(hi["abstain_high"] - 0.1363) < 1e-3
        assert hi["abstain_high"] - 0.1 > 0.1 - hi["abstain_low"]          # wider above: symmetric in logit, not in probability
        _policy(monkeypatch)                                                 # architecture default: ±0.05 probability
        assert stage2_grade.fuse({"grade": 1, "grade_label": "Mild NPDR", "referable_probability": 0.06}, rule)["abstain"]

    def test_attention_floor_comes_from_policy_and_drops_the_lift_condition(self, monkeypatch):
        from backend.venus.stage3_explain import attention_agreement
        size = 128
        mask = np.zeros((size, size), np.uint8); mask[8:-8, 8:-8] = 255
        lesion = np.zeros((size, size), np.uint8); lesion[60:68, 60:68] = 255
        stage1 = {"masks": {"MA": lesion, "HE": np.zeros_like(lesion), "EX": np.zeros_like(lesion), "SE": np.zeros_like(lesion)}}
        # Heat spread evenly over the FOV: score ~= chance level, lift ~= 1.
        heat = (mask > 0).astype(np.float32)
        _policy(monkeypatch, attention_floor=0.30, attention_min_lift=None)
        a = attention_agreement(heat, stage1, mask, counted=1, referable=True)
        assert a["floor"] == 0.30 and a["score"] < 0.30 and a["flag"] is True
        _policy(monkeypatch, attention_floor=0.30, attention_min_lift=1.5)
        b = attention_agreement(heat, stage1, mask, counted=1, referable=True)
        assert b["lift"] < 1.5 and b["flag"] is True
        _policy(monkeypatch, attention_floor=0.30, attention_min_lift=0.5)   # lift condition not met -> no flag
        c = attention_agreement(heat, stage1, mask, counted=1, referable=True)
        assert c["flag"] is False
        # Never flags a non-referable call or one with no counted lesion.
        assert attention_agreement(heat, stage1, mask, counted=0, referable=True)["flag"] is False
        assert attention_agreement(heat, stage1, mask, counted=1, referable=False)["flag"] is False

    def test_served_policy_file_is_consistent_with_itself(self):
        path = ROOT / "backend/config/review_policy.json"
        if not path.exists():
            pytest.skip("no review policy in this checkout")
        p = json.loads(path.read_text(encoding="utf-8"))
        assert 0 < p["abstain_band_logit"] < 1 and 0 < p["attention_floor"] < 1
        assert 0 < p["resulting_flag_rate"] < 0.5
        assert p["error_rate_flagged"] > p["error_rate_unflagged"], "a review rule must concentrate errors in the queue"
        chosen = p["attention_floor"]
        assert str(chosen) in {str(float(k)) for k in p["attention_candidates"]} or chosen == 0.15


class TestLesionNetworks:
    def test_hires_switch_is_off_by_default_and_status_says_so(self):
        from backend.venus import config, stage1_segment
        assert not config.LESION_THRESHOLDS_HIRES_PATH.exists() or json.loads(config.LESION_THRESHOLDS_HIRES_PATH.read_text())["serves"]
        status = stage1_segment.unet_status()
        assert "hires" in status and set(status["hires"]) >= {"loaded", "serves", "note"}
        assert stage1_segment.lesion_method() in {"classical", "unet"} or stage1_segment.lesion_method().startswith("unet (")

    def test_recorded_experiment_explains_why_it_is_not_shipped(self):
        path = ROOT / "backend/config/experiments/lesion_unet_1024.json"
        if not path.exists():
            pytest.skip("experiment record not present")
        e = json.loads(path.read_text(encoding="utf-8"))
        d = e["decision"]
        assert d["shipped"] is False and len(d["why"]) > 100
        assert d["cpu_timing_median_ms"]["with_1024_for_MA"] > d["cpu_timing_median_ms"]["512_only"]
        assert e["test_aupr"]["MA"] > 0.079, "the experiment's one measured gain"


class TestStreamingAupr:
    def test_histogram_aupr_matches_sklearn(self):
        pytest.importorskip("tensorflow")
        from backend.training import train_lesion_unet as U
        from sklearn.metrics import average_precision_score
        rng = np.random.default_rng(1)
        y = rng.random(40000) < 0.01
        p = np.clip(rng.beta(2, 5, 40000) + 0.3 * y, 0, 1).astype(np.float32)

        class Model:
            i = 0

            def predict_on_batch(self, x):
                out = np.stack([p.reshape(4, 100, 100)[Model.i:Model.i + x.shape[0]]] * 4, -1); Model.i += x.shape[0]; return out

        masks = (y.reshape(4, 100, 100) * 15).astype(np.uint8)
        out = U.evaluate(Model(), np.zeros((4, 100, 100, 3), np.uint8), masks, 2)
        assert abs(out["MA"]["aupr"] - average_precision_score(y, p)) < 2e-4
        assert out["MA"]["positives"] == int(y.sum()) and 0.05 <= out["MA"]["threshold"] <= 0.98


class TestValidationApi:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        with TestClient(app) as c:
            yield c

    def test_validation_extras_serves_the_config_without_row_data(self, client):
        r = client.get("/validation-extras"); assert r.status_code == 200
        d = r.json()
        assert "review_policy" in d and "rows" not in d.get("validation_flags", {})
        assert "experiments" in d and all(isinstance(v, dict) for v in d["experiments"].values())

    def test_operating_point_endpoint_matches_file(self, client):
        r = client.get("/operating-point"); assert r.status_code == 200
        on_disk = json.loads((ROOT / "backend/config/operating_point.json").read_text(encoding="utf-8"))
        assert r.json()["calibration_fingerprint"] == on_disk["calibration_fingerprint"]

    def test_health_reports_every_network(self, client):
        h = client.get("/health").json()
        assert {"quality_cnn", "lesion_unet", "operating_point"} <= set(h)
        assert "hires" in h["lesion_unet"]
