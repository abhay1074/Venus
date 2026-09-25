"""The PDF says what the system is, and says which lesion path produced it.

docs/MODEL_CARD.md, the web UI footer and both PDF generators carry one
sentence verbatim. A report without it is the artefact most likely to be read
out of context (printed, forwarded, pinned to a referral), so this is asserted
on the bytes of a real PDF rather than on the source string.

Run:  python -m pytest backend/tests/test_report.py -q
"""

from __future__ import annotations

import base64
import os
import re
import zlib
from pathlib import Path

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
from backend.venus.config import GATE_WEIGHTS, GRADER_WEIGHTS  # noqa: E402

HAS_WEIGHTS = GRADER_WEIGHTS.exists() and GATE_WEIGHTS.exists()
needs_weights = pytest.mark.skipif(not HAS_WEIGHTS, reason="trained weights not present")

STATEMENT = ("Venus AI is a triage aid for referable diabetic retinopathy. A clinician reviews every case. "
             "It is not a diagnosis and it is not a cleared medical device.")


def pdf_text(path: Path) -> bytes:
    """Text operators of a ReportLab PDF: content streams are ASCII85 + Flate."""
    raw = path.read_bytes()
    out = b""
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", raw, re.S):
        data = re.sub(rb"\s", b"", m.group(1))
        if data.endswith(b"~>"):
            data = data[:-2]
        try:
            out += zlib.decompress(base64.a85decode(data))
        except Exception:  # noqa: BLE001 - image streams are not text
            continue
    return out


class TestStatementEverywhere:
    def test_python_constant_is_the_verbatim_statement(self):
        from backend.venus.report import DISCLAIMER
        assert DISCLAIMER == STATEMENT

    def test_model_card_web_ui_and_matlab_report_carry_it(self):
        card = (ROOT / "docs" / "MODEL_CARD.md").read_text(encoding="utf-8")
        assert STATEMENT in card, "docs/MODEL_CARD.md must open with the statement (regenerate with write_docs)"
        app = (ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
        assert STATEMENT in app, "the web UI footer must carry the statement"
        matlab = (ROOT / "matlab" / "+drscreen" / "report.m").read_text(encoding="utf-8")
        assert "not a cleared medical device" in matlab, "the MATLAB PDF footer must carry the statement"

    @needs_weights
    def test_statement_and_lesion_method_reach_the_pdf(self, tmp_path):
        from backend.venus import pipeline, report
        result = pipeline.screen_image((ROOT / "samples" / "dr_exudates.png").read_bytes(), persist=False, write_report=False)
        assert result["accepted"], result.get("stop_reason")
        path = report.write_pdf(result, tmp_path / "r.pdf")
        text = pdf_text(path)
        assert b"not a cleared medical device" in text
        assert result["model_version"].encode() in text and result["calibration_fingerprint"][:16].encode() in text
        # The lesion line must name the path that actually ran - it once
        # printed "classical detectors" while the U-Net was serving.
        # PDF string literals escape parentheses: "Lesions \(unet\)".
        assert f"Lesions \\({result['stage1']['method']}\\)".encode() in text
