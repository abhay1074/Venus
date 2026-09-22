"""Venus AI — HTTP API around the screening pipeline.

    GET  /health                         networks, operating point, DB, timing
    GET  /operating-point                the locked threshold and its provenance
    POST /screen                         multipart image (+ optional intake JSON) -> result
    GET  /screenings                     review queue: every stored screening
    GET  /screenings/{session_id}        one stored result (without overlays)
    GET  /report/{session_id}.pdf        the one-page report
    GET  /report/{session_id}.json       the JSON sidecar
    GET  /samples, /samples/{name}       demo images shipped with the build
    POST /simulate                       one district scenario, AI vs baseline
    GET  /sweep, POST /sweep             cached Pareto sweep / start a new one
    POST /intake                         register a patient (consent required)
    POST /appointments                   tier + book the earliest feasible slot
    GET  /worklist                       doctor worklist sorted by priority
    POST /appointments/{id}/outcome      confirmed | treated | referred_onward | no_show
    GET  /facilities                     facilities and free slots
    POST /reset-demo                     wipe the demo database and reports

Every response that carries a grade also carries the model version and the
calibration fingerprint, so a screen can always be traced to the threshold it
was made with.
"""

from __future__ import annotations

import json
import re
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.venus import __version__, pipeline, stage0_gate, stage1_segment, stage2_grade, stage4_simulate, stage5_schedule
from backend.venus.config import (ALLOWED_IMAGE_TYPES, CONFIG_DIR, MAX_UPLOAD_MB, MODEL_VERSION, PROJECT_ROOT, REPORT_DIR, SAMPLES_DIR,
                                  SCREEN_RATE_BURST, SCREEN_RATE_PER_MIN, OperatingPointError, operating_point)
from backend.venus.ratelimit import RateLimiter

app = FastAPI(title="Venus AI", version=__version__,
              description="Explainable diabetic-retinopathy screening for district programmes (SIH26038).")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:4173", "http://localhost:4173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

_startup: dict = {}

# Session ids are minted by pipeline.screen_image as VS- + 10 hex characters.
# Report lookups match this exactly: a path fragment never reaches the filesystem.
SESSION_ID_RE = re.compile(r"VS-[0-9A-F]{10}")


@app.on_event("startup")
async def startup() -> None:
    global _startup
    stage5_schedule.init_db()
    try:
        _startup = pipeline.warm_up()
    except OperatingPointError as exc:
        _startup = {"error": str(exc)}
    if os.getenv("VENUS_SWEEP_ON_START", "true").lower() == "true" and stage4_simulate.cached_sweep() is None:
        # Full year, in the background (~80 s on a laptop); cached to config/.
        stage4_simulate.start_sweep(sample_fraction=1.0)


# ------------------------------------------------------------- status --

MIN_SCREENINGS_FOR_MEASURED_RATES = 10


def _measured_rates() -> dict:
    """Flag and retake rates measured on stored screenings; used by the
    simulation only once enough screenings exist to mean anything."""
    m = stage5_schedule.flag_rate()
    m["used_by_simulation"] = m["n_gradable"] >= MIN_SCREENINGS_FOR_MEASURED_RATES
    m["minimum_n"] = MIN_SCREENINGS_FOR_MEASURED_RATES
    m["validation_sample"] = stage4_simulate.measured_flag_rates()
    return m


def _served_version() -> str:
    """The version the operating point carries (a site calibration appends
    +site_<name>); the config constant is the fallback before a point exists."""
    try:
        return operating_point()["model_version"]
    except OperatingPointError:
        return MODEL_VERSION


@app.get("/health")
async def health() -> dict:
    try:
        point = operating_point()
        op = {"ok": True, "threshold": point["thresholds"]["referable"],
              "fingerprint": point["calibration_fingerprint"], "model_version": point["model_version"]}
    except OperatingPointError as exc:
        op = {"ok": False, "error": str(exc)}
    return {
        "status": "ok" if op["ok"] and stage2_grade.grader_status()["loaded"] else "degraded",
        "model_version": _served_version(),
        "gate": stage0_gate.gate_status(),
        "grader": stage2_grade.grader_status(),
        "quality_cnn": stage0_gate.quality_status(),
        "lesion_unet": stage1_segment.unet_status(),
        "operating_point": op,
        "warm_up": _startup,
        "measured": _measured_rates(),
        "sweep": stage4_simulate.sweep_status(),
    }


@app.get("/operating-point")
async def get_operating_point() -> dict:
    try:
        return operating_point()
    except OperatingPointError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/validation-extras")
async def get_validation_extras() -> dict:
    """The validation-sample analyses that sit beside the operating point:
    the served human-review policy, the flag-rate / attention study it was
    chosen from, the lesion U-Net's once-scored test numbers, and the
    experiments that were measured and not shipped. All read from the JSON
    the evaluation scripts wrote; nothing is computed here."""
    out = {}
    for key, name in (("review_policy", "review_policy.json"), ("validation_flags", "validation_flags.json"),
                      ("lesion_thresholds", "lesion_thresholds.json"), ("ensemble_check", "ensemble_check.json"),
                      ("site_calibration", "site_calibration_messidor2.json")):
        path = CONFIG_DIR / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                out[key] = json.load(handle)
    if "validation_flags" in out:
        out["validation_flags"].pop("rows", None)
    experiments = {}
    for path in sorted((CONFIG_DIR / "experiments").glob("*.json")) if (CONFIG_DIR / "experiments").exists() else []:
        with open(path, "r", encoding="utf-8") as handle:
            experiments[path.stem] = json.load(handle)
    out["experiments"] = experiments
    return out


# ------------------------------------------------------------ screening --

MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
screen_limiter = RateLimiter(SCREEN_RATE_PER_MIN, SCREEN_RATE_BURST)


def _enforce_rate_limit(request: Request) -> None:
    """One screen costs ~1.5 s of the single CPU the operator is waiting on."""
    client = request.client.host if request.client else "unknown"
    wait = screen_limiter.check(client)
    if wait:
        raise HTTPException(status_code=429, detail=f"Too many screening requests from this client. Retry in {wait:.0f}s.",
                            headers={"Retry-After": str(max(int(wait), 1))})


async def _read_capped(upload: UploadFile, request: Request) -> bytes:
    """Read the upload without ever buffering more than the cap.

    The declared Content-Length is refused first, so a 5 GB upload costs one
    response rather than 5 GB of a shared laptop's RAM; a lying or absent
    header is caught by reading in chunks and stopping one byte over."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES + _MULTIPART_SLACK:
        raise HTTPException(status_code=413, detail=f"The upload exceeds the {MAX_UPLOAD_MB} MB limit.")
    chunks, size = [], 0
    while chunk := await upload.read(1 << 20):
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"The image exceeds the {MAX_UPLOAD_MB} MB limit.")
        chunks.append(chunk)
    return b"".join(chunks)


# Multipart framing (boundaries, headers) rides along with the file in
# Content-Length; allow a little room so a file just under the cap is not
# refused for its envelope.
_MULTIPART_SLACK = 64 * 1024


def _validate_upload(upload: UploadFile, payload: bytes) -> None:
    if upload.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG or WEBP fundus photograph.")
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")


@app.post("/screen")
async def screen(request: Request, file: UploadFile = File(...), intake: Optional[str] = Form(default=None),
                 tta: bool = Form(default=False)) -> dict:
    _enforce_rate_limit(request)
    payload = await _read_capped(file, request)
    _validate_upload(file, payload)
    intake_dict = None
    if intake:
        try:
            intake_dict = json.loads(intake)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="intake must be JSON") from exc
    try:
        return pipeline.screen_image(payload, intake=intake_dict, tta=tta)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OperatingPointError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/screenings")
async def screenings(limit: int = 200) -> list:
    return stage5_schedule.list_screenings(limit)


@app.get("/screenings/{session_id}")
async def screening(session_id: str) -> dict:
    path = REPORT_DIR / f"{session_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="unknown session")
    return json.load(open(path, encoding="utf-8"))


def _report_path(session_id: str, suffix: str) -> Path:
    """Resolve a report file for a session id this process issued.

    The id is matched against the format pipeline.screen_image mints and is
    never joined as a path fragment, so '..', an absolute path or a glob can
    never leave REPORT_DIR. A known session whose file is missing means the
    write failed (full or unwritable disk) and is answered 507, not 404, so
    the operator is told the difference."""
    if not SESSION_ID_RE.fullmatch(session_id):
        raise HTTPException(status_code=404, detail="no report for this session")
    path = REPORT_DIR / f"{session_id}{suffix}"
    if not path.exists():
        if stage5_schedule.screening_exists(session_id):
            raise HTTPException(status_code=507, detail="This case was screened but its report could not be written "
                                                        "(the reports directory was unwritable or the disk was full). "
                                                        "The result is still in the review queue.")
        raise HTTPException(status_code=404, detail="no report for this session")
    return path


@app.get("/report/{session_id}.pdf")
async def report_pdf(session_id: str):
    path = _report_path(session_id, ".pdf")
    return FileResponse(path, media_type="application/pdf", filename=f"venus-{session_id}.pdf")


@app.get("/report/{session_id}.json")
async def report_json(session_id: str):
    path = _report_path(session_id, ".json")
    return FileResponse(path, media_type="application/json")


@app.get("/samples")
async def samples() -> list:
    if not SAMPLES_DIR.exists():
        return []
    meta_path = SAMPLES_DIR / "samples.json"
    meta = json.load(open(meta_path, encoding="utf-8")) if meta_path.exists() else {}
    out = []
    for path in sorted(SAMPLES_DIR.iterdir()):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            out.append({"name": path.name, "url": f"/samples/{path.name}", **meta.get(path.name, {})})
    return out


@app.get("/samples/{name}")
async def sample(name: str):
    path = SAMPLES_DIR / Path(name).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="unknown sample")
    return FileResponse(path)


# ----------------------------------------------------------- simulation --

class SimulateRequest(BaseModel):
    params: dict = {}
    use_measured_rates: bool = True


@app.post("/simulate")
async def simulate(req: SimulateRequest) -> dict:
    measured = _measured_rates() if req.use_measured_rates else {}
    use = bool(measured.get("used_by_simulation"))
    try:
        return stage4_simulate.compare(
            req.params,
            measured_flag_rate=measured.get("flag_rate") if use else None,
            measured_retake_rate=measured.get("retake_rate") if use else None,
        )
    except OperatingPointError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/sweep")
async def get_sweep() -> dict:
    cached = stage4_simulate.cached_sweep()
    status = stage4_simulate.sweep_status()
    if cached is None:
        return {"status": status, "result": None}
    return {"status": status, "result": cached}


class SweepRequest(BaseModel):
    params: dict = {}
    sample_fraction: float = 1.0
    max_p95_wait_days: float = 7.0
    max_missed: Optional[int] = None
    max_missed_fraction: float = 0.20


@app.post("/sweep")
async def post_sweep(req: SweepRequest) -> dict:
    started = stage4_simulate.start_sweep(req.params, sample_fraction=req.sample_fraction,
                                          max_p95_wait_days=req.max_p95_wait_days, max_missed=req.max_missed,
                                          max_missed_fraction=req.max_missed_fraction)
    return {"started": started, "status": stage4_simulate.sweep_status()}


# ----------------------------------------------------------- scheduling --

class IntakeRequest(BaseModel):
    patient_id: Optional[str] = None
    name: Optional[str] = None
    phone: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    village: Optional[str] = None
    phc: Optional[str] = None
    diabetes_years: Optional[float] = None
    hba1c: Optional[float] = None
    insulin: bool = False
    hypertension: bool = False
    pregnant: bool = False
    last_eye_exam: Optional[str] = None
    symptoms: list[str] = []
    consent: bool = False


@app.post("/intake")
async def intake(req: IntakeRequest) -> dict:
    if not req.consent:
        raise HTTPException(status_code=400, detail="Consent is required before a patient record is created.")
    patient_id = stage5_schedule.save_patient(req.model_dump())
    return {"patient_id": patient_id}


class AppointmentRequest(BaseModel):
    patient_id: str
    session_id: Optional[str] = None
    intake: dict = {}


@app.post("/appointments")
async def create_appointment(req: AppointmentRequest) -> dict:
    result = None
    if req.session_id:
        path = REPORT_DIR / f"{req.session_id}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="unknown session")
        result = json.load(open(path, encoding="utf-8"))
    tier = stage5_schedule.priority_tier(result, req.intake)
    if tier["tier"] == "P0":
        return {"tier": tier, "appointment": None, "note": "retake on the same visit; no appointment needed"}
    appointment = stage5_schedule.book(req.patient_id, req.session_id, tier)
    return {"tier": tier, "appointment": appointment}


@app.get("/worklist")
async def get_worklist() -> list:
    return stage5_schedule.worklist()


class OutcomeRequest(BaseModel):
    outcome: str


@app.post("/appointments/{appointment_id}/outcome")
async def outcome(appointment_id: str, req: OutcomeRequest) -> dict:
    if req.outcome not in ("confirmed", "treated", "referred_onward", "no_show"):
        raise HTTPException(status_code=400, detail="unknown outcome")
    try:
        return stage5_schedule.record_outcome(appointment_id, req.outcome)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown appointment") from exc


@app.get("/facilities")
async def get_facilities() -> list:
    return stage5_schedule.facilities()


@app.post("/reset-demo")
async def reset_demo() -> dict:
    stage5_schedule.reset_db()
    removed = 0
    if REPORT_DIR.exists():
        for path in REPORT_DIR.iterdir():
            if path.suffix in (".pdf", ".json"):
                path.unlink()
                removed += 1
    return {"reset": True, "reports_removed": removed}


# Offline deployment: when the front end has been built (frontend/dist), the
# API serves it at / so a PHC laptop runs one process and needs no Node, no
# browser plug-ins and no network. The Vite dev server is still the way to
# develop; this mount is only used when dist/ exists.
_DIST = PROJECT_ROOT / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
