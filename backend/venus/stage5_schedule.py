"""Stage 5 — risk-prioritised appointment and worklist.

Turns a screening result into a booked appointment. The model output sets the
base priority tier; clinical risk factors from the intake form can raise a tier
by one, never lower it. A priority queue keyed on (tier, risk score, waiting
time) feeds a slot allocator that books the earliest open slot at the nearest
facility able to serve the tier, within the tier's deadline, bumping a
lower-priority booking whose own deadline still allows it when nothing is free.

Tiers follow the ICDR follow-up guidance:

    P0 retake     Stage 0 reject                          same visit, PHC
    P1 urgent     PDR, P(NV) > 0.5, sudden vision loss,    7 days, district hospital
                  grade 3 with symptoms
    P2 referable  grade 2-3 or P(referable) >= threshold  30 days, ophthalmology / tele
    P3 review     human-review flag                        3 days tele-review, then re-tier
    P4 routine    grade 0-1, gradable, no symptoms         12-month recall (6 months if
                                                          diabetes > 10 y or HbA1c > 9)

Records live in SQLite: patients, screenings, facilities, slots, appointments.
Patient identifiers live only in `patients`; screenings carry the session ID.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from backend.venus.config import DB_PATH

TIERS = {
    "P0": {"label": "Retake", "deadline_days": 0, "deadline_text": "same visit", "facility": "PHC",
           "facility_types": ["phc"], "rank": 0},
    "P1": {"label": "Urgent", "deadline_days": 7, "deadline_text": "within 7 days",
           "facility": "district hospital, in person", "facility_types": ["district_hospital"], "rank": 1},
    "P3": {"label": "Review", "deadline_days": 3, "deadline_text": "tele-review within 3 days, then re-tiered",
           "facility": "tele-ophthalmology", "facility_types": ["tele", "clinic", "district_hospital"], "rank": 2},
    "P2": {"label": "Referable", "deadline_days": 30, "deadline_text": "within 30 days",
           "facility": "nearest ophthalmology clinic or tele-review",
           "facility_types": ["clinic", "district_hospital", "tele"], "rank": 3},
    "P4": {"label": "Routine", "deadline_days": 365, "deadline_text": "12-month recall",
           "facility": "PHC camera", "facility_types": ["phc"], "rank": 4},
}
# Ordering used by the queue: P0 first because it is resolved on the spot,
# then urgent, review, referable, routine.
TIER_ORDER = ["P0", "P1", "P3", "P2", "P4"]

_lock = threading.RLock()


# ------------------------------------------------------------- tiering --

def priority_tier(result: dict | None, intake: dict | None = None) -> dict:
    """Base tier from the model output, raised by at most one step by risk factors."""
    intake = intake or {}
    symptoms = set(intake.get("symptoms") or [])
    reasons = []

    if result is None or not result.get("accepted", True):
        tier = "P0"
        reasons.append("image rejected at Stage 0; retake on the same visit")
    else:
        fusion = result["stage2"]["fusion"]
        cnn = result["stage2"]["cnn"]
        grade = fusion["grade"]
        if grade == 4 or cnn["nv_probability"] > 0.5:
            tier = "P1"; reasons.append("PDR evidence (grade 4 or P(NV) > 0.5)")
        elif "sudden_vision_loss" in symptoms:
            tier = "P1"; reasons.append("sudden vision loss reported")
        elif grade == 3 and symptoms:
            tier = "P1"; reasons.append("severe NPDR with symptoms")
        elif fusion["flag_for_review"]:
            tier = "P3"; reasons.append("human-review flag: " + "; ".join(fusion["flag_reasons"]))
        elif grade >= 2 or fusion["referable"]:
            tier = "P2"; reasons.append(f"grade {grade} / P(referable) {fusion['p_referable']:.2f} ≥ threshold")
        else:
            tier = "P4"; reasons.append(f"grade {grade}, gradable, no symptoms")

    # Risk factors raise by one step, never lower. Pregnancy and visual
    # symptoms are strong: P4 -> P2, P2 -> P1. Metabolic factors (HbA1c > 9,
    # diabetes > 10 years, hypertension, insulin) shorten a routine recall to
    # six months instead of changing the tier.
    strong, metabolic = [], []
    if intake.get("pregnant"):
        strong.append("pregnancy")
    if symptoms & {"blurred_vision", "floaters"}:
        strong.append("visual symptoms")
    if intake.get("hba1c") not in (None, "") and float(intake["hba1c"]) > 9:
        metabolic.append("HbA1c > 9")
    if intake.get("diabetes_years") not in (None, "") and float(intake["diabetes_years"]) > 10:
        metabolic.append("diabetes > 10 years")
    if intake.get("hypertension"):
        metabolic.append("hypertension")
    if intake.get("insulin"):
        metabolic.append("insulin use")
    raise_one = strong + metabolic

    recall_months = 12
    if strong and tier in ("P4", "P2"):
        tier = {"P4": "P2", "P2": "P1"}[tier]
        reasons.append("raised one tier by risk factors: " + ", ".join(strong))
    elif tier == "P4" and metabolic:
        recall_months = 6
        reasons.append("recall shortened to 6 months: " + ", ".join(metabolic))

    meta = TIERS[tier]
    risk = 0.0
    if result is not None and result.get("accepted", True):
        risk = float(result["stage2"]["fusion"]["p_referable"])
    risk += 0.05 * len(raise_one)
    return {
        "tier": tier,
        "label": meta["label"],
        "deadline_days": meta["deadline_days"] if not (tier == "P4" and recall_months == 6) else 183,
        "deadline_text": meta["deadline_text"] if not (tier == "P4" and recall_months == 6) else "6-month recall",
        "facility": meta["facility"],
        "risk_score": round(min(risk, 1.0), 3),
        "risk_factors": raise_one,
        "reasons": reasons,
    }


# ------------------------------------------------------------- storage --

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY, name TEXT, phone TEXT, age INTEGER, sex TEXT, village TEXT, phc TEXT,
    diabetes_years REAL, hba1c REAL, insulin INTEGER, hypertension INTEGER, pregnant INTEGER,
    last_eye_exam TEXT, symptoms TEXT, consent INTEGER NOT NULL, consent_at TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS screenings (
    session_id TEXT PRIMARY KEY, patient_id TEXT, captured_at TEXT, accepted INTEGER,
    grade INTEGER, referable INTEGER, p_referable REAL, flag INTEGER, tier TEXT, quality TEXT,
    result_json TEXT, elapsed_ms INTEGER
);
CREATE TABLE IF NOT EXISTS facilities (
    id TEXT PRIMARY KEY, name TEXT, type TEXT, lat REAL, lon REAL, slots_per_day INTEGER
);
CREATE TABLE IF NOT EXISTS slots (
    id TEXT PRIMARY KEY, facility_id TEXT, starts_at TEXT, booked INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS appointments (
    id TEXT PRIMARY KEY, patient_id TEXT, session_id TEXT, tier TEXT, risk_score REAL,
    queued_at TEXT, deadline_at TEXT, slot_id TEXT, facility_id TEXT, starts_at TEXT,
    status TEXT, outcome TEXT, bump_history TEXT, sms_log TEXT
);
"""

FACILITIES = [
    {"id": "DH-01", "name": "District Hospital Ophthalmology", "type": "district_hospital", "lat": 0.0, "lon": 0.0, "slots_per_day": 24},
    {"id": "CL-01", "name": "Taluk Eye Clinic North", "type": "clinic", "lat": 0.35, "lon": 0.10, "slots_per_day": 16},
    {"id": "CL-02", "name": "Taluk Eye Clinic South", "type": "clinic", "lat": -0.30, "lon": -0.15, "slots_per_day": 16},
    {"id": "TELE", "name": "Tele-ophthalmology reading centre", "type": "tele", "lat": 0.0, "lon": 0.0, "slots_per_day": 60},
    {"id": "PHC-07", "name": "PHC 7 fundus camera", "type": "phc", "lat": 0.20, "lon": 0.30, "slots_per_day": 40},
    {"id": "PHC-12", "name": "PHC 12 fundus camera", "type": "phc", "lat": -0.25, "lon": 0.20, "slots_per_day": 40},
]
# Village -> approximate position, so "nearest facility" means something.
VILLAGES = {
    "Kharia": (0.30, 0.25), "Belpur": (-0.20, 0.15), "Rampur": (0.05, -0.05),
    "Sonpur": (0.40, 0.05), "Dhanora": (-0.35, -0.10), "Mohanpur": (0.10, 0.35),
}


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db():
    """Commit on success, roll back on error, and always close: sqlite3's own
    context manager commits but never closes, and an open handle keeps the
    file locked on Windows (so reset_db could not delete it)."""
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(now: datetime | None = None, days: int = 45) -> None:
    now = now or datetime.now(timezone.utc)
    with _lock, db() as conn:
        conn.executescript(SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM facilities").fetchone()[0] == 0:
            conn.executemany("INSERT INTO facilities VALUES (:id,:name,:type,:lat,:lon,:slots_per_day)", FACILITIES)
        if conn.execute("SELECT COUNT(*) FROM slots").fetchone()[0] == 0:
            rows = []
            for f in FACILITIES:
                per_day = f["slots_per_day"]
                for d in range(1, days + 1):
                    day = (now + timedelta(days=d)).replace(hour=9, minute=0, second=0, microsecond=0)
                    if day.weekday() == 6:
                        continue
                    step = timedelta(minutes=max(int(8 * 60 / per_day), 3))
                    for k in range(per_day):
                        starts = day + k * step
                        rows.append((f"{f['id']}-{starts.strftime('%Y%m%d%H%M')}", f["id"], starts.isoformat(), 0))
            conn.executemany("INSERT INTO slots VALUES (?,?,?,?)", rows)


def reset_db() -> None:
    with _lock:
        if DB_PATH.exists():
            DB_PATH.unlink()
    init_db()


# -------------------------------------------------------- allocation --

def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _queue_key(tier: str, risk: float, waiting_hours: float, deadline_hours: float):
    """Lower sorts first. Ageing: past its deadline, a lower tier outranks a
    fresh higher tier, so no tier is starved."""
    rank = TIER_ORDER.index(tier)
    overdue = waiting_hours > deadline_hours
    return (0 if overdue else 1, rank, -risk, -waiting_hours)


def allocate(queue: list[dict], slots: list[dict], now: datetime, facilities: dict) -> list[dict]:
    """Pure scheduling function, unit-tested on synthetic queues.

    queue: [{id, tier, risk_score, queued_at, village_pos, patient_id, ...}]
    slots: [{id, facility_id, starts_at, booked, appointment_id}]  (mutated)
    Returns decisions [{appointment_id, slot_id, facility_id, starts_at, bumped}].
    """
    decisions = []
    ordered = sorted(queue, key=lambda q: _queue_key(
        q["tier"], q["risk_score"],
        (now - q["queued_at"]).total_seconds() / 3600,
        TIERS[q["tier"]]["deadline_days"] * 24 if q.get("deadline_days") is None else q["deadline_days"] * 24))
    for item in ordered:
        window = timedelta(days=item.get("deadline_days", TIERS[item["tier"]]["deadline_days"]))
        deadline = item["queued_at"] + window
        if deadline <= now:
            # Already overdue (a no-show, or a long wait): the ageing term has
            # put it at the front of the queue; give it a fresh window from now
            # so it is booked at the earliest slot rather than waitlisted.
            deadline = now + max(window, timedelta(days=1))
        allowed = TIERS[item["tier"]]["facility_types"]
        candidates = [s for s in slots if not s["booked"] and s["starts_at"] > now
                      and s["starts_at"] <= deadline and facilities[s["facility_id"]]["type"] in allowed]
        if candidates:
            candidates.sort(key=lambda s: (s["starts_at"].date(),
                                           _distance(item["village_pos"], facilities[s["facility_id"]]["pos"]),
                                           s["starts_at"]))
            chosen = candidates[0]
            chosen["booked"], chosen["appointment_id"] = 1, item["id"]
            decisions.append({"appointment_id": item["id"], "slot_id": chosen["id"], "facility_id": chosen["facility_id"],
                              "starts_at": chosen["starts_at"], "bumped": None})
            continue
        # No slot inside the deadline: bump the lowest-priority booking whose
        # own deadline still allows re-slotting, then re-queue it.
        booked = [s for s in slots if s["booked"] and s["starts_at"] > now and s["starts_at"] <= deadline
                  and facilities[s["facility_id"]]["type"] in allowed and s.get("holder")]
        booked.sort(key=lambda s: TIER_ORDER.index(s["holder"]["tier"]), reverse=True)
        bumped = None
        for s in booked:
            holder = s["holder"]
            if TIER_ORDER.index(holder["tier"]) <= TIER_ORDER.index(item["tier"]):
                break
            holder_deadline = holder["queued_at"] + timedelta(days=holder.get("deadline_days", TIERS[holder["tier"]]["deadline_days"]))
            spare = [t for t in slots if not t["booked"] and t["starts_at"] > s["starts_at"] and t["starts_at"] <= holder_deadline
                     and facilities[t["facility_id"]]["type"] in TIERS[holder["tier"]]["facility_types"]]
            if spare:
                spare.sort(key=lambda t: t["starts_at"])
                new = spare[0]
                new["booked"], new["appointment_id"], new["holder"] = 1, holder["id"], holder
                s["booked"], s["appointment_id"], s["holder"] = 1, item["id"], item
                bumped = {"appointment_id": holder["id"], "from_slot": s["id"], "to_slot": new["id"], "starts_at": new["starts_at"]}
                decisions.append({"appointment_id": item["id"], "slot_id": s["id"], "facility_id": s["facility_id"],
                                  "starts_at": s["starts_at"], "bumped": bumped})
                break
        if bumped is None:
            decisions.append({"appointment_id": item["id"], "slot_id": None, "facility_id": None,
                              "starts_at": None, "bumped": None})
    return decisions


# --------------------------------------------------------- persistence --

def save_patient(intake: dict) -> str:
    patient_id = intake.get("patient_id") or f"PT-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now(timezone.utc).isoformat()
    with _lock, db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO patients VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (patient_id, intake.get("name"), intake.get("phone"), intake.get("age"), intake.get("sex"),
             intake.get("village"), intake.get("phc"), intake.get("diabetes_years"), intake.get("hba1c"),
             int(bool(intake.get("insulin"))), int(bool(intake.get("hypertension"))), int(bool(intake.get("pregnant"))),
             intake.get("last_eye_exam"), json.dumps(intake.get("symptoms") or []),
             int(bool(intake.get("consent"))), now if intake.get("consent") else None, now))
    return patient_id


def save_screening(result: dict, patient_id: str | None = None) -> None:
    with _lock, db() as conn:
        accepted = bool(result.get("accepted", True))
        fusion = result["stage2"]["fusion"] if accepted else None
        conn.execute(
            "INSERT OR REPLACE INTO screenings VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (result["session_id"], patient_id, result["captured_at"], int(accepted),
             fusion["grade"] if fusion else None, int(fusion["referable"]) if fusion else None,
             fusion["p_referable"] if fusion else None, int(fusion["flag_for_review"]) if fusion else None,
             result["stage5"]["tier"], result["stage0"]["quality"]["label"] if result["stage0"].get("quality") else "reject",
             json.dumps({k: v for k, v in result.items() if k != "stage3"} | {"stage3": {k: v for k, v in result.get("stage3", {}).items() if k != "overlays"}}),
             result["timing_ms"]["total"]))


def list_screenings(limit: int = 200) -> list[dict]:
    with _lock, db() as conn:
        rows = conn.execute("SELECT session_id, patient_id, captured_at, accepted, grade, referable, p_referable, flag, tier, quality, elapsed_ms, result_json "
                            "FROM screenings ORDER BY captured_at DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        res = json.loads(d.pop("result_json"))
        d["cnn_grade"] = res["stage2"]["cnn"]["grade"] if res.get("stage2") else None
        d["rule_grade"] = res["stage2"]["rule"]["grade"] if res.get("stage2") else None
        d["flag_reasons"] = res["stage2"]["fusion"]["flag_reasons"] if res.get("stage2") else []
        d["attention_agreement"] = res.get("stage3", {}).get("attention_agreement", {}).get("score")
        out.append(d)
    return out


def flag_rate() -> dict:
    """Measured human-review flag rate over stored gradable screenings."""
    with _lock, db() as conn:
        row = conn.execute("SELECT COUNT(*), COALESCE(SUM(flag),0), COALESCE(SUM(referable),0) FROM screenings WHERE accepted=1").fetchone()
        rejected = conn.execute("SELECT COUNT(*) FROM screenings WHERE accepted=0").fetchone()[0]
    n, flagged, referable = row
    return {"n_gradable": n, "flagged": flagged, "flag_rate": round(flagged / n, 4) if n else None,
            "referable_rate": round(referable / n, 4) if n else None, "rejected": rejected,
            "retake_rate": round(rejected / (n + rejected), 4) if (n + rejected) else None}


def book(patient_id: str, session_id: str | None, tier_info: dict, now: datetime | None = None) -> dict:
    """Queue one patient and run the allocator against the live slot table."""
    now = now or datetime.now(timezone.utc)
    init_db(now)
    with _lock, db() as conn:
        patient = conn.execute("SELECT * FROM patients WHERE id=?", (patient_id,)).fetchone()
        village = patient["village"] if patient else None
        pos = VILLAGES.get(village, (0.0, 0.0))
        facilities = {f["id"]: {"type": f["type"], "pos": (f["lat"], f["lon"]), "name": f["name"]}
                      for f in conn.execute("SELECT * FROM facilities")}
        slot_rows = conn.execute("SELECT s.*, a.tier AS holder_tier, a.queued_at AS holder_queued, a.deadline_at AS holder_deadline, a.id AS holder_id "
                                 "FROM slots s LEFT JOIN appointments a ON a.slot_id = s.id AND a.status='booked' "
                                 "WHERE s.starts_at > ?", (now.isoformat(),)).fetchall()
        slots = []
        for s in slot_rows:
            entry = {"id": s["id"], "facility_id": s["facility_id"], "starts_at": datetime.fromisoformat(s["starts_at"]),
                     "booked": int(s["booked"]), "appointment_id": s["holder_id"]}
            if s["holder_id"]:
                entry["holder"] = {"id": s["holder_id"], "tier": s["holder_tier"],
                                   "queued_at": datetime.fromisoformat(s["holder_queued"]),
                                   "deadline_days": (datetime.fromisoformat(s["holder_deadline"]) - datetime.fromisoformat(s["holder_queued"])).days}
            slots.append(entry)
        appointment_id = f"AP-{uuid.uuid4().hex[:8].upper()}"
        item = {"id": appointment_id, "tier": tier_info["tier"], "risk_score": tier_info["risk_score"],
                "queued_at": now, "deadline_days": tier_info["deadline_days"], "village_pos": pos, "patient_id": patient_id}
        decisions = allocate([item], slots, now, facilities)
        decision = decisions[0]
        deadline_at = (now + timedelta(days=tier_info["deadline_days"])).isoformat()
        sms = []
        if decision["slot_id"]:
            facility = facilities[decision["facility_id"]]["name"]
            when = decision["starts_at"].strftime("%d %b %Y %H:%M")
            sms.append({"at": now.isoformat(), "to": patient["phone"] if patient else None,
                        "text": f"Venus AI: eye appointment booked {when} at {facility}. Bring your diabetes record. Reply 1 to confirm, 2 to reschedule."})
            conn.execute("UPDATE slots SET booked=1 WHERE id=?", (decision["slot_id"],))
            status = "booked"
        else:
            status = "waitlisted"
            sms.append({"at": now.isoformat(), "to": patient["phone"] if patient else None,
                        "text": "Venus AI: no slot inside your deadline yet; you are on the priority waitlist and will be called."})
        bump_history = []
        if decision["bumped"]:
            b = decision["bumped"]
            conn.execute("UPDATE appointments SET slot_id=?, starts_at=?, bump_history=json_insert(COALESCE(bump_history,'[]'), '$[#]', ?) WHERE id=?",
                         (b["to_slot"], b["starts_at"].isoformat(),
                          json.dumps({"at": now.isoformat(), "by": appointment_id, "from": b["from_slot"], "to": b["to_slot"]}), b["appointment_id"]))
            conn.execute("UPDATE slots SET booked=1 WHERE id=?", (b["to_slot"],))
            bump_history.append({"at": now.isoformat(), "bumped": b["appointment_id"], "to_slot": b["to_slot"]})
        conn.execute("INSERT INTO appointments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (appointment_id, patient_id, session_id, tier_info["tier"], tier_info["risk_score"], now.isoformat(), deadline_at,
                      decision["slot_id"], decision["facility_id"],
                      decision["starts_at"].isoformat() if decision["starts_at"] else None, status, None,
                      json.dumps(bump_history), json.dumps(sms)))
        return get_appointment(appointment_id, conn)


def get_appointment(appointment_id: str, conn: sqlite3.Connection | None = None) -> dict:
    own = conn is None
    conn = conn or connect()
    row = conn.execute("SELECT a.*, f.name AS facility_name, p.name AS patient_name, p.village AS village "
                       "FROM appointments a LEFT JOIN facilities f ON f.id=a.facility_id LEFT JOIN patients p ON p.id=a.patient_id "
                       "WHERE a.id=?", (appointment_id,)).fetchone()
    if own:
        conn.close()
    if row is None:
        raise KeyError(appointment_id)
    d = dict(row)
    d["bump_history"] = json.loads(d["bump_history"] or "[]")
    d["sms_log"] = json.loads(d["sms_log"] or "[]")
    d["tier_label"] = TIERS[d["tier"]]["label"]
    return d


def worklist(limit: int = 200) -> list[dict]:
    with _lock, db() as conn:
        rows = conn.execute("SELECT a.id FROM appointments a ORDER BY a.queued_at DESC LIMIT ?", (limit,)).fetchall()
        items = [get_appointment(r["id"], conn) for r in rows]
    # Open bookings first, then by tier, risk and slot time; closed ones at the end.
    items.sort(key=lambda a: (0 if a["status"] in ("booked", "waitlisted") else 1,
                              TIER_ORDER.index(a["tier"]), -(a["risk_score"] or 0), a["starts_at"] or "9"))
    return items


def record_outcome(appointment_id: str, outcome: str) -> dict:
    """confirmed | treated | referred_onward | no_show. A no-show re-queues at
    the same tier with its waiting time preserved."""
    now = datetime.now(timezone.utc)
    with _lock, db() as conn:
        appt = get_appointment(appointment_id, conn)
        if outcome == "no_show":
            conn.execute("UPDATE slots SET booked=0 WHERE id=?", (appt["slot_id"],))
            conn.execute("UPDATE appointments SET status='no_show', outcome='no_show' WHERE id=?", (appointment_id,))
            tier_info = {"tier": appt["tier"], "risk_score": appt["risk_score"],
                         "deadline_days": TIERS[appt["tier"]]["deadline_days"]}
        else:
            conn.execute("UPDATE appointments SET status='completed', outcome=? WHERE id=?", (outcome, appointment_id))
            return get_appointment(appointment_id, conn)
    # Re-book outside the connection context, preserving the original queue time.
    rebooked = book(appt["patient_id"], appt["session_id"], tier_info, now=datetime.fromisoformat(appt["queued_at"]))
    return rebooked


def facilities() -> list[dict]:
    init_db()
    with _lock, db() as conn:
        rows = conn.execute("SELECT f.*, (SELECT COUNT(*) FROM slots s WHERE s.facility_id=f.id AND s.booked=0) AS free_slots FROM facilities f").fetchall()
    return [dict(r) for r in rows]
