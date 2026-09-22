"""API hardening for a shared laptop: bind address, upload cap, rate limit,
and report lookups that only serve session ids this process issued.

The threat model is mundane and real: a demo laptop on conference wi-fi, an API
with no authentication, one CPU, and a browser tab that retries. See
docs/PRIVACY.md §6 for what is and is not protected.

Run:  python -m pytest backend/tests/test_hardening.py -q
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("VENUS_SWEEP_ON_START", "false")

ROOT = Path(__file__).resolve().parents[2]


class TestBindAddress:
    def test_loopback_by_default(self):
        from backend.serve import LOOPBACK, chosen_host
        host, warning = chosen_host(bind_all_flag=False, env={})
        assert host == LOOPBACK
        assert warning is None

    @pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on"])
    def test_env_opt_in_exposes_and_warns(self, value):
        from backend.serve import ALL_INTERFACES, chosen_host
        host, warning = chosen_host(bind_all_flag=False, env={"VENUS_BIND_ALL": value})
        assert host == ALL_INTERFACES
        assert "no authentication" in warning

    @pytest.mark.parametrize("value", ["false", "0", "no", "", "maybe"])
    def test_unset_or_falsey_env_stays_loopback(self, value):
        from backend.serve import LOOPBACK, chosen_host
        assert chosen_host(bind_all_flag=False, env={"VENUS_BIND_ALL": value})[0] == LOOPBACK

    def test_command_line_flag_exposes_and_warns(self):
        from backend.serve import ALL_INTERFACES, chosen_host
        host, warning = chosen_host(bind_all_flag=True, env={})
        assert host == ALL_INTERFACES and warning

    def test_serve_script_and_container_use_the_policy(self):
        """The two ways this is started must go through backend.serve, and the
        container's 0.0.0.0 must be the documented opt-in rather than a flag."""
        serve_ps1 = (ROOT / "scripts" / "serve.ps1").read_text(encoding="utf-8")
        assert "backend.serve" in serve_ps1 and "--host 0.0.0.0" not in serve_ps1
        dockerfile = (ROOT / "backend" / "Dockerfile").read_text(encoding="utf-8")
        assert "backend.serve" in dockerfile and "VENUS_BIND_ALL=true" in dockerfile


class TestRateLimit:
    def test_burst_then_refusal_then_recovery(self):
        from backend.venus.ratelimit import RateLimiter
        limiter = RateLimiter(per_minute=60, burst=3)
        now = 1000.0
        assert [limiter.check("a", now) for _ in range(3)] == [0.0, 0.0, 0.0]
        wait = limiter.check("a", now)
        assert wait > 0, "the fourth request in a burst of 3 must be refused"
        assert limiter.check("a", now + wait + 0.01) == 0.0, "it must recover after the stated wait"

    def test_clients_have_separate_budgets(self):
        from backend.venus.ratelimit import RateLimiter
        limiter = RateLimiter(per_minute=60, burst=2)
        now = 500.0
        limiter.check("operator", now); limiter.check("operator", now)
        assert limiter.check("operator", now) > 0
        assert limiter.check("other-laptop", now) == 0.0, "one client must not starve another"

    def test_disabled_when_configured_to_zero(self):
        from backend.venus.ratelimit import RateLimiter
        limiter = RateLimiter(per_minute=0, burst=1)
        assert limiter.enabled is False
        assert all(limiter.check("a", 1.0) == 0.0 for _ in range(50))

    def test_screen_endpoint_returns_429_with_retry_after(self, monkeypatch):
        """A retry loop is refused with a header it can obey, not a 500.

        The refill rate has to be well below one per screen: a screen takes
        ~1.5 s, so at 60/min the bucket refills faster than a caller can drain
        it and nothing would ever be refused (which is the intended behaviour
        for a human at a camera, and the reason the shipped default is 30/min
        with a burst of 10)."""
        from backend.venus.ratelimit import RateLimiter
        from backend import main
        monkeypatch.setattr(main, "screen_limiter", RateLimiter(per_minute=6, burst=2))
        payload = (ROOT / "samples" / "dr_exudates.png").read_bytes()
        with TestClient(main.app) as client:
            codes = []
            for _ in range(4):
                r = client.post("/screen", files={"file": ("x.png", io.BytesIO(payload), "image/png")})
                codes.append(r.status_code)
                last = r
            assert 429 in codes, codes
            assert last.status_code == 429
            assert int(last.headers["Retry-After"]) >= 1
            assert "Retry in" in last.json()["detail"]

    def test_limit_does_not_apply_to_reading_endpoints(self, monkeypatch):
        """Only the expensive endpoint is limited; the review queue is not."""
        from backend.venus.ratelimit import RateLimiter
        from backend import main
        monkeypatch.setattr(main, "screen_limiter", RateLimiter(per_minute=60, burst=1))
        with TestClient(main.app) as client:
            assert all(client.get("/screenings").status_code == 200 for _ in range(12))


class TestUploadCap:
    def test_cap_is_documented_and_enforced(self):
        from backend.main import MAX_UPLOAD_BYTES
        from backend.venus.config import MAX_UPLOAD_MB
        assert MAX_UPLOAD_BYTES == MAX_UPLOAD_MB * 1024 * 1024
        demo = (ROOT / "docs" / "DEMO.md").read_text(encoding="utf-8")
        privacy = (ROOT / "docs" / "PRIVACY.md").read_text(encoding="utf-8")
        assert str(MAX_UPLOAD_MB) in demo or str(MAX_UPLOAD_MB) in privacy, \
            "the upload cap must be documented where an operator would look"


class TestReportPathSafety:
    @pytest.fixture(scope="class")
    def client(self):
        from backend.main import app
        with TestClient(app) as c:
            yield c

    @pytest.mark.parametrize("payload", [
        "../../../backend/config/operating_point",
        "../../CLAUDE",
        "..\\..\\CLAUDE",
        "C:\\Users\\anilm\\OneDrive\\Venus AI SIH 26\\CLAUDE",
        "/etc/passwd",
        "%2e%2e%2f%2e%2e%2fCLAUDE",
        "VS-../../../CLAUDE",
        "VS-ZZZZZZZZZZ",
        "vs-0123456789",
    ])
    def test_traversal_and_foreign_ids_are_refused(self, client, payload):
        for suffix in (".pdf", ".json"):
            r = client.get(f"/report/{payload}{suffix}")
            assert r.status_code == 404, f"{payload}{suffix} -> {r.status_code}"
            assert "operating_point" not in r.text and "%PDF" not in r.text

    def test_the_pattern_matches_what_the_pipeline_mints(self):
        """If the id format ever changes, this fails rather than silently
        refusing every real report."""
        import re
        from backend.main import SESSION_ID_RE
        from backend.venus import pipeline
        source = Path(pipeline.__file__).read_text(encoding="utf-8")
        minted = re.search(r'session_id = f"(VS-)\{uuid\.uuid4\(\)\.hex\[:(\d+)\]\.upper\(\)\}"', source)
        assert minted, "pipeline no longer mints session ids the way this test expects"
        example = "VS-" + "0123456789ABCDEF"[: int(minted.group(2))]
        assert SESSION_ID_RE.fullmatch(example), f"{example} should be accepted by the API pattern"
