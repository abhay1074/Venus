"""One-page PDF report and JSON sidecar for a screening result.

The page a clinician checks in under a minute: session ID, capture time,
quality label, original image, lesion overlay, Grad-CAM, grade with the ICDR
criterion that fired, calibrated confidence, referral tier, and in the footer
the model version plus the calibration fingerprint the threshold was locked
against. The JSON sidecar carries the same fields for any downstream system.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from backend.venus.config import ICDR_LABELS, REPORT_DIR

# The one sentence that must appear wherever a grade is shown; docs/MODEL_CARD.md
# opens and closes with it and backend/tests asserts it reaches the PDF.
DISCLAIMER = ("Venus AI is a triage aid for referable diabetic retinopathy. A clinician reviews every case. "
              "It is not a diagnosis and it is not a cleared medical device.")

NAVY = colors.HexColor("#102A43")
TEAL = colors.HexColor("#0F766E")
ROSE = colors.HexColor("#BE123C")
AMBER = colors.HexColor("#B45309")
GREY = colors.HexColor("#64748B")
LINE = colors.HexColor("#D8E2EA")


def _image_reader(png_b64: str) -> ImageReader:
    return ImageReader(BytesIO(base64.b64decode(png_b64)))


def _wrap(text: str, width_chars: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width_chars:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def write_pdf(result: dict, path: Path) -> Path:
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    margin = 14 * mm
    y = height - margin

    # Header
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Venus AI — Diabetic Retinopathy Screening Report")
    c.setFont("Helvetica", 9)
    c.setFillColor(GREY)
    y -= 6 * mm
    c.drawString(margin, y, f"Session {result['session_id']}   ·   captured {result['captured_at']}   ·   "
                            f"quality: {result['stage0']['quality']['label']} "
                            f"(score {result['stage0']['quality']['score']:.2f}"
                            f"{', enhanced' if result['stage0']['quality']['enhanced'] else ''})")
    y -= 3 * mm
    c.setStrokeColor(LINE)
    c.line(margin, y, width - margin, y)

    # Verdict block
    fusion = result["stage2"]["fusion"]
    tier = result["stage5"]
    y -= 10 * mm
    c.setFont("Helvetica-Bold", 22)
    c.setFillColor(ROSE if fusion["referable"] else TEAL)
    verdict = "REFERABLE DR" if fusion["referable"] else "NOT REFERABLE"
    if fusion["flag_for_review"]:
        verdict += "  ·  HUMAN REVIEW"
        c.setFillColor(AMBER)
    c.drawString(margin, y, verdict)
    c.setFont("Helvetica", 10)
    c.setFillColor(NAVY)
    y -= 6 * mm
    c.drawString(margin, y, f"ICDR grade {fusion['grade']} — {ICDR_LABELS[fusion['grade']]}      "
                            f"P(referable) = {fusion['p_referable']:.2f} (calibrated; threshold {fusion['threshold']:.2f})"
                            f"{'   ·   ABSTAIN BAND' if fusion['abstain'] else ''}")
    y -= 5 * mm
    c.drawString(margin, y, f"Priority tier {tier['tier']} — {tier['label']}: {tier['deadline_text']}, {tier['facility']}")

    # Images
    y -= 8 * mm
    panel = (width - 2 * margin - 2 * 4 * mm) / 3
    labels = [("Original", "original"), ("Lesion overlay", "lesions"), ("Grad-CAM (referable head)", "gradcam_referable")]
    for i, (label, key) in enumerate(labels):
        x = margin + i * (panel + 4 * mm)
        c.drawImage(_image_reader(result["stage3"]["overlays"][key]), x, y - panel, panel, panel)
        c.setFont("Helvetica", 8)
        c.setFillColor(GREY)
        c.drawString(x, y - panel - 3.5 * mm, label)
    y -= panel + 9 * mm

    # Evidence columns
    def block(title, lines, x, top, col_width):
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(NAVY)
        c.drawString(x, top, title)
        c.setFont("Helvetica", 8.5)
        c.setFillColor(colors.black)
        yy = top - 4.5 * mm
        for line in lines:
            for piece in _wrap(line, col_width):
                c.drawString(x, yy, piece)
                yy -= 3.8 * mm
        return yy

    rule = result["stage2"]["rule"]
    cnn = result["stage2"]["cnn"]
    s1 = result["stage1"]
    agreement = result["stage3"]["attention_agreement"]
    col = (width - 2 * margin - 6 * mm) / 2

    left = [
        f"CNN grader: grade {cnn['grade']} ({cnn['grade_label']}), five-grade probabilities "
        + ", ".join(f"G{i} {p:.2f}" for i, p in enumerate(cnn["grade_probabilities"])),
        f"Rule grader: grade {rule['grade']} ({rule['grade_label']})",
    ] + [f"• {t}" for t in rule["trace"]] + [
        f"Lesions ({s1['method']}): MA {s1['lesions']['MA']['count']}, HE {s1['lesions']['HE']['count']} "
        f"(quadrants {s1['hemorrhages_per_quadrant']}), EX {s1['lesions']['EX']['count']}, SE {s1['lesions']['SE']['count']}",
        f"PDR evidence: NV probability {cnn['nv_probability']:.2f} (classifier, not localised)",
    ]
    right = [
        result["stage2"]["fusion"]["confidence_text"],
        ("Attention agreement: " + (f"{agreement['score']:.2f} (chance {agreement.get('chance_level', 0):.2f}) — {agreement['note']}"
                                    if agreement["score"] is not None else agreement["note"])),
        "Review flags: " + ("; ".join(fusion["flag_reasons"]) if fusion["flag_reasons"] else "none"),
        f"Quality: sharpness {result['stage0']['quality']['features']['sharpness']:.0f}, illumination "
        f"{result['stage0']['quality']['features']['illumination_uniformity']:.2f}, FOV circularity "
        f"{result['stage0']['quality']['features']['fov_circularity']:.2f}",
        f"Pipeline time: {result['timing_ms']['total']} ms on this machine "
        f"(stage 0 {result['timing_ms']['stage0']}, 1 {result['timing_ms']['stage1']}, "
        f"2 {result['timing_ms']['stage2']}, 3 {result['timing_ms']['stage3']})",
    ]
    y_left = block("Grade evidence", left, margin, y, 62)
    y_right = block("Confidence and review", right, margin + col + 6 * mm, y, 62)
    y = min(y_left, y_right) - 4 * mm

    # Recommendation
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(NAVY)
    c.drawString(margin, y, "Recommendation")
    c.setFont("Helvetica", 8.5)
    c.setFillColor(colors.black)
    y -= 4.5 * mm
    for piece in _wrap(result["recommendation"], 130):
        c.drawString(margin, y, piece)
        y -= 3.8 * mm

    # Footer: the same sentence docs/MODEL_CARD.md opens and closes with, beside
    # the version and fingerprint that make this page traceable to its threshold.
    c.setFont("Helvetica", 7)
    c.setFillColor(GREY)
    c.drawString(margin, margin - 1 * mm, DISCLAIMER)
    c.drawString(margin, margin - 4.5 * mm,
                 f"Model {result['model_version']}   ·   calibration fingerprint {result['calibration_fingerprint'][:16]}…   ·   "
                 "Every image is read by an eye-care professional.")
    c.showPage()
    c.save()
    return path


def write_json(result: dict, path: Path) -> Path:
    slim = {k: v for k, v in result.items() if k != "stage3"}
    slim["stage3"] = {k: v for k, v in result["stage3"].items() if k != "overlays"}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(slim, handle, indent=2)
    return path


def write_report(result: dict) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = write_pdf(result, REPORT_DIR / f"{result['session_id']}.pdf")
    sidecar = write_json(result, REPORT_DIR / f"{result['session_id']}.json")
    return {"pdf": pdf.name, "json": sidecar.name}
