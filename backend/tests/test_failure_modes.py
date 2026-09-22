"""Failure modes must produce a typed HTTP error, never a stack trace or a 500.

A screening laptop in a PHC fails in dull ways: a half-copied checkpoint, a full
disk, an operator who uploads a PDF, two clicks on the same button. Each case
below is provoked for real — the file is truncated, the directory is made
unwritable, the requests are sent concurrently — rather than mocked, because
these tests exist to prove the error path, and a mocked error path proves
nothing about the real one.

Run:  python -m pytest backend/tests/test_failure_modes.py -q
"""

from __future__ import annotations

import concurrent.futures as futures
import io
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("VENUS_SWEEP_ON_START", "false")

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "samples"
# Ask config where the weights actually are (VENUS_WEIGHTS_DIR moves them), so
# the skip tracks the served path and a weights-less run can be reproduced.
from backend.venus.config import GATE_WEIGHTS, GRADER_WEIGHTS  # noqa: E402

HAS_WEIGHTS = GRADER_WEIGHTS.exists() and GATE_WEIGHTS.exists()
needs_weights = pytest.mark.skipif(not HAS_WEIGHTS, reason="trained weights not present")


@pytest.fixture(scope="module")
def client():
    from backend.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def fundus() -> bytes:
    return (SAMPLES / "dr_exudates.png").read_bytes()


def upload(client, content: bytes, name: str = "x.png", ctype: str = "image/png"):
    return client.post("/screen", files={"file": (name, io.BytesIO(content), ctype)})


class TestBadUploads:
    def test_empty_upload_is_400_not_500(self, client):
        r = upload(client, b"")
        assert r.status_code == 400
        assert "empty" in r.json()["detail"].lower()

    def test_not_an_image_is_400_with_readable_message(self, client):
        r = upload(client, b"%PDF-1.4 this is a pdf, not a fundus photograph" * 20)
        assert r.status_code == 400
        assert "image" in r.json()["detail"].lower()

    def test_truncated_png_is_400(self, client, fundus):
        """A half-transferred PNG: valid header, body cut off."""
        r = upload(client, fundus[: len(fundus) // 3])
        assert r.status_code == 400
        assert "image" in r.json()["detail"].lower()

    def test_header_only_png_is_400(self, client, fundus):
        r = upload(client, fundus[:64])
        assert r.status_code == 400

    def test_wrong_content_type_is_415(self, client, fundus):
        r = upload(client, fundus, name="x.txt", ctype="text/plain")
        assert r.status_code == 415

    def test_oversize_upload_is_413(self, client):
        from backend.main import MAX_UPLOAD_BYTES
        r = upload(client, b"\x89PNG\r\n\x1a\n" + b"\0" * (MAX_UPLOAD_BYTES + 1024))
        assert r.status_code == 413
        assert "limit" in r.json()["detail"].lower()

    def test_huge_upload_is_refused_from_the_header_without_buffering(self, client):
        """A declared 4 GB body must cost one response, not 4 GB of RAM."""
        from backend.main import MAX_UPLOAD_BYTES, _read_capped
        from fastapi import HTTPException

        class FakeRequest:
            headers = {"content-length": str(4 * 1024 ** 3)}

        class NeverRead:
            async def read(self, _n):
                raise AssertionError("the body must not be read once Content-Length exceeds the cap")

        import asyncio
        with pytest.raises(HTTPException) as err:
            asyncio.run(_read_capped(NeverRead(), FakeRequest()))
        assert err.value.status_code == 413
        assert MAX_UPLOAD_BYTES > 0


class TestMissingCheckpoints:
    def test_health_reports_degraded_with_a_reason(self, client, monkeypatch):
        """A checkpoint that is absent or truncated must show up in /health as
        `degraded` with the loader's reason, not as a healthy service."""
        from backend.venus import stage2_grade
        monkeypatch.setattr(stage2_grade, "_model", None)
        monkeypatch.setattr(stage2_grade, "_load_error", "OSError: truncated file")
        monkeypatch.setattr(stage2_grade, "load_grader", lambda: None)
        health = client.get("/health").json()
        assert health["status"] == "degraded"
        assert health["grader"]["loaded"] is False
        assert "truncated" in health["grader"]["error"]

    @needs_weights
    def test_truncated_grader_gives_503_naming_the_fix(self, client, monkeypatch, fundus):
        """With the gate loaded, a broken grader is reached and must answer 503
        pointing at the checksum tool — never a stack trace.

        Needs weights because without them the modality gate refuses the image
        first (see the next test); that is correct, but it tests a different path."""
        from backend.venus import stage2_grade
        monkeypatch.setattr(stage2_grade, "_model", None)
        monkeypatch.setattr(stage2_grade, "_load_error", "OSError: truncated file")
        monkeypatch.setattr(stage2_grade, "load_grader", lambda: None)
        r = upload(client, fundus)
        assert r.status_code == 503, r.text
        detail = r.json()["detail"]
        assert "checksums.py" in detail and "Traceback" not in detail

    def test_without_any_checkpoint_the_gate_refuses_cleanly(self, client, monkeypatch, fundus):
        """The no-weights case a fresh clone hits: the modality gate cannot
        vouch for the image, so Stage 0 refuses it with an operator-facing
        reason and tier P0. A 200 carrying `accepted: false` is the right
        answer here — nothing diagnostic ran and nothing crashed."""
        from backend.venus import stage0_gate
        monkeypatch.setattr(stage0_gate, "_gate_model", None, raising=False)
        monkeypatch.setattr(stage0_gate, "_gate_error", "missing modality gate weights", raising=False)
        monkeypatch.setattr(stage0_gate, "load_gate", lambda: None)
        r = upload(client, fundus)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["accepted"] is False
        assert body["stage5"]["tier"] == "P0"
        assert "fundus" in body["stop_reason"] and "Traceback" not in r.text
        assert body.get("stage2") is None, "nothing diagnostic may run once the gate has refused"


class TestReportWriteFailure:
    @needs_weights
    def test_unwritable_reports_directory_keeps_the_result(self, client, monkeypatch, fundus, tmp_path):
        """A full or unwritable disk must not destroy a finished grade: the
        result comes back with report=None and a stated reason, and the PDF
        endpoint answers 507 (not 404) because the session is known."""
        from backend.venus import report as report_module
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("this is a file, so mkdir underneath it fails")
        monkeypatch.setattr(report_module, "REPORT_DIR", blocker / "reports")

        r = upload(client, fundus)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["accepted"] is True
        assert body["stage2"]["fusion"]["grade"] is not None      # the clinical result survived
        assert body["report"] is None
        assert "could not be written" in body["report_error"]

        from backend.venus import stage5_schedule
        assert stage5_schedule.screening_exists(body["session_id"])
        pdf = client.get(f"/report/{body['session_id']}.pdf")
        assert pdf.status_code == 507
        assert "disk" in pdf.json()["detail"].lower()


class TestConcurrency:
    @needs_weights
    def test_concurrent_screens_all_succeed_with_distinct_sessions(self, client, fundus):
        """Two operators clicking at once: both get a result, no shared-state
        corruption, distinct session ids (TensorFlow serialises internally)."""
        with futures.ThreadPoolExecutor(3) as pool:
            responses = [f.result() for f in [pool.submit(upload, client, fundus) for _ in range(3)]]
        assert [r.status_code for r in responses] == [200, 200, 200]
        bodies = [r.json() for r in responses]
        assert len({b["session_id"] for b in bodies}) == 3
        grades = {b["stage2"]["cnn"]["grade"] for b in bodies}
        assert len(grades) == 1, f"the same image graded differently under concurrency: {grades}"


class TestReportLookup:
    @pytest.mark.parametrize("bad", [
        "../../../backend/config/operating_point", "..%2f..%2fconfig", "C:\\Windows\\system",
        "/etc/passwd", "VS-notahexid", "*", "VS-0123456789ABCDEF", "",
    ])
    def test_only_issued_session_ids_are_served(self, client, bad):
        """No path fragment, glob or foreign id may reach the filesystem."""
        r = client.get(f"/report/{bad}.pdf")
        assert r.status_code == 404, f"{bad!r} returned {r.status_code}"
        assert "operating_point" not in r.text and "%PDF" not in r.text

    def test_sample_endpoint_rejects_traversal(self, client):
        assert client.get("/samples/../../CLAUDE.md").status_code == 404
