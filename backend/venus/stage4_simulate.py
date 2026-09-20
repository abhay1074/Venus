"""Stage 4 — district screening programme simulation.

A discrete-event model of one district screening `annual_patients` diabetics a
year, parameterised by the grader's MEASURED sensitivity, specificity and
human-review flag rate, so false negatives and false positives emerge from the
model rather than being assumed. It answers the district health officer's
question: how many cameras, operators and ophthalmologist-hours does this
programme need, and what does the AI change.

    patient generator (Poisson, ~400 / working day)
        -> PHC queue (N cameras, M operators per PHC)
        -> capture + Stage 0 (retake loop, p_retake)
        -> AI triage (sens, spec, flag rate)        [baseline: no AI, every image read]
             referable or flagged -> ophthalmologist queue (K doctors, tele-review)
                 confirmed -> treatment referral (in-person)
                 false positive -> discharge
             non-referable -> discharge + 12-month recall
             missed cases -> vision-loss counter

Each patient entity carries a true DR state drawn from prevalence, a quality
draw, and timestamps at every station. Time is in working minutes; a working
day has `phc_hours_per_day` hours for PHCs and `doctor_hours_per_day` hours of
ophthalmologist time. Queues persist across days, so a year-end backlog is a
real output, not an artefact.

This is the SimEvents model of the architecture written as a plain event loop
(heapq). The parameter struct, entity attributes and outputs are the same, so
the Simulink version can be built block for block from this file.
"""

from __future__ import annotations

import heapq
import json
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import numpy as np

from backend.venus.config import CONFIG_DIR, operating_point

SWEEP_CACHE = CONFIG_DIR / "sweep_cache.json"
VALIDATION_FLAGS = CONFIG_DIR / "validation_flags.json"


def measured_flag_rates() -> dict | None:
    """Flag and retake rates measured on validation images by backend.eval.flag_rate."""
    if not VALIDATION_FLAGS.exists():
        return None
    v = json.load(open(VALIDATION_FLAGS, encoding="utf-8"))
    flag_rate = v["flag_rate"]
    policy_path = CONFIG_DIR / "review_policy.json"
    if policy_path.exists():
        # The served review rule is the validation-chosen policy; its flag
        # rate on the same sample is what the programme absorbs.
        flag_rate = json.load(open(policy_path, encoding="utf-8")).get("resulting_flag_rate", flag_rate)
    return {"flag_rate": flag_rate, "retake_rate": v["retake_rate"], "n": v["n_sampled"],
            "grader": v.get("grader"), "written_at": v.get("written_at"), "source": "validation manifest sample"}


@dataclass
class Params:
    annual_patients: int = 100_000
    working_days: int = 250
    referable_prevalence: float = 0.06
    any_dr_prevalence: float = 0.18
    phcs: int = 30
    cameras_per_phc: int = 1
    operators_per_phc: int = 1
    capture_minutes: float = 4.0
    retake_probability: float = 0.05          # measured Stage 0 reject rate when available
    phc_hours_per_day: float = 8.0
    ai_enabled: bool = True
    sensitivity: float = 0.90                 # coupled from Stage 2 external test
    specificity: float = 0.60
    flag_rate: float = 0.12                   # human-review flag rate from fusion
    human_reader_sensitivity: float = 0.85    # baseline: a human tele-reader is not perfect either
    human_reader_specificity: float = 0.90
    ophthalmologists: int = 3
    doctor_hours_per_day: float = 6.0
    tele_review_minutes: float = 3.0
    in_person_minutes: float = 20.0
    operator_hour_cost: float = 150.0         # INR
    doctor_hour_cost: float = 1500.0          # INR
    camera_annual_cost: float = 50_000.0      # INR, amortised portable fundus camera
    seed: int = 42
    sample_fraction: float = 1.0              # < 1 simulates a fraction of the year (sweeps)


def params_from(overrides: dict | None = None, point: dict | None = None, measured_flag_rate: float | None = None,
                measured_retake_rate: float | None = None) -> Params:
    """Defaults coupled to the locked operating point; overrides from the UI."""
    point = point or operating_point()
    p = Params()
    at = point["external_test"]["at_locked_threshold"]
    p.sensitivity = float(at["sensitivity"])
    p.specificity = float(at["specificity"])
    # Flag and retake rates measured on validation images, when available;
    # explicit measured_* arguments (e.g. from stored screenings) override.
    measured = measured_flag_rates()
    if measured is not None:
        p.flag_rate = float(measured["flag_rate"])
        p.retake_probability = float(measured["retake_rate"])
    if measured_flag_rate is not None:
        p.flag_rate = float(measured_flag_rate)
    if measured_retake_rate is not None:
        p.retake_probability = float(measured_retake_rate)
    for key, value in (overrides or {}).items():
        if hasattr(p, key) and value is not None:
            setattr(p, key, type(getattr(p, key))(value))
    return p


# ------------------------------------------------------------- engine --

class _DailyServers:
    """A pool of servers that each work `budget` minutes per `day_len` day."""

    def __init__(self, count: int, budget: float, day_len: float, horizon: float):
        self.count = max(int(count), 1)
        self.budget = budget
        self.day_len = day_len
        self.horizon = horizon
        self.free_at = [0.0] * self.count
        self.used_today = [0.0] * self.count
        self.today = [0] * self.count
        self.busy_minutes = 0.0

    def start(self, now: float, service: float) -> float:
        """Assign the earliest-free server; return the completion time."""
        i = min(range(self.count), key=self.free_at.__getitem__)
        t = max(now, self.free_at[i])
        day = int(t // self.day_len)
        if day != self.today[i]:
            self.today[i], self.used_today[i] = day, 0.0
        if self.used_today[i] + service > self.budget:
            # Out of hours today: start at the beginning of the next day.
            day += 1
            t = day * self.day_len
            self.today[i], self.used_today[i] = day, 0.0
        self.used_today[i] += service
        self.free_at[i] = t + service
        if t < self.horizon:
            # Work started inside the simulated period counts; anything that
            # only starts after the horizon is backlog, not utilisation.
            self.busy_minutes += min(service, self.horizon - t)
        return t + service


def simulate(p: Params) -> dict:
    rng = np.random.default_rng(p.seed)
    day_len = p.phc_hours_per_day * 60.0
    days = max(int(round(p.working_days * p.sample_fraction)), 1)
    n = int(round(p.annual_patients * p.sample_fraction))
    horizon = days * day_len

    # Arrivals: Poisson process over working time, spread across PHCs.
    arrivals = np.sort(rng.uniform(0, horizon, size=n))
    phc_of = rng.integers(0, p.phcs, size=n)
    referable = rng.random(n) < p.referable_prevalence
    any_dr = referable | (rng.random(n) < (p.any_dr_prevalence - p.referable_prevalence) / max(1 - p.referable_prevalence, 1e-6))
    retake = rng.random(n) < p.retake_probability

    if p.ai_enabled:
        ai_positive = np.where(referable, rng.random(n) < p.sensitivity, rng.random(n) < (1 - p.specificity))
        flagged = rng.random(n) < p.flag_rate
        to_doctor = ai_positive | flagged
    else:
        ai_positive = np.zeros(n, bool)
        flagged = np.zeros(n, bool)
        to_doctor = np.ones(n, bool)
    reader_calls_referable = np.where(referable, rng.random(n) < p.human_reader_sensitivity,
                                      rng.random(n) < (1 - p.human_reader_specificity))

    phc_pools = [_DailyServers(min(p.cameras_per_phc, p.operators_per_phc), day_len, day_len, horizon) for _ in range(p.phcs)]
    doctors = _DailyServers(p.ophthalmologists, p.doctor_hours_per_day * 60.0, day_len, horizon)

    # Doctor queue is priority: AI-positive cases before review-only flags,
    # FIFO within a class. Events: (time, seq, kind, index).
    events: list = []
    seq = 0
    for i in range(n):
        heapq.heappush(events, (float(arrivals[i]), seq, "arrive", i)); seq += 1
    doctor_queue: list = []          # (priority, ready_time, seq, i)
    capture_done = np.full(n, np.nan)
    result_time = np.full(n, np.nan)
    doctor_wait = np.full(n, np.nan)
    phc_wait = np.full(n, np.nan)
    doctor_free_events = 0
    pending_wakeups: set = set()

    def dispatch_doctor(now: float):
        """Serve queued cases while a doctor can start them now."""
        nonlocal seq
        while doctor_queue:
            i_free = min(range(doctors.count), key=doctors.free_at.__getitem__)
            free_at = doctors.free_at[i_free]
            if free_at > now + 1e-9:
                # Someone is busy; schedule ONE wake-up for when they free up.
                if free_at not in pending_wakeups:
                    pending_wakeups.add(free_at)
                    heapq.heappush(events, (free_at, seq, "doctor_free", -1)); seq += 1
                return
            _, ready, _, i = heapq.heappop(doctor_queue)
            service = p.tele_review_minutes + (p.in_person_minutes if reader_calls_referable[i] else 0.0)
            done = doctors.start(now, service)
            doctor_wait[i] = done - ready
            heapq.heappush(events, (done, seq, "result", i)); seq += 1

    while events:
        now, _, kind, i = heapq.heappop(events)
        if kind == "arrive":
            pool = phc_pools[phc_of[i]]
            service = p.capture_minutes * (2 if retake[i] else 1)
            done = pool.start(now, service)
            phc_wait[i] = done - service - now
            heapq.heappush(events, (done, seq, "captured", i)); seq += 1
        elif kind == "captured":
            capture_done[i] = now
            if to_doctor[i]:
                priority = 0 if ai_positive[i] else 1
                heapq.heappush(doctor_queue, (priority, now, seq, i)); seq += 1
                dispatch_doctor(now)
            else:
                result_time[i] = now
        elif kind == "doctor_free":
            doctor_free_events += 1
            pending_wakeups.discard(now)
            dispatch_doctor(now)
        elif kind == "result":
            result_time[i] = now

    # Outcomes ------------------------------------------------------------
    # A result after the horizon is the year-end backlog.
    done_mask = ~np.isnan(result_time) & (result_time <= horizon)
    wait_days = (result_time[done_mask] - capture_done[done_mask]) / day_len
    missed = referable & ~to_doctor                          # never seen by a human
    seen = to_doctor & done_mask                             # actually read within the year
    missed_by_reader = referable & seen & ~reader_calls_referable
    unresulted = referable & to_doctor & ~done_mask          # still in the backlog at year end
    unnecessary = to_doctor & ~referable                     # doctor time on a non-referable
    confirmed = referable & seen & reader_calls_referable
    backlog = int((~done_mask).sum())
    scale = 1.0 / p.sample_fraction

    operator_hours = sum(pool.busy_minutes for pool in phc_pools) / 60.0
    doctor_hours = doctors.busy_minutes / 60.0
    doctor_capacity_hours = p.ophthalmologists * p.doctor_hours_per_day * days
    camera_cost = p.camera_annual_cost * p.phcs * p.cameras_per_phc * (days / p.working_days)
    cost = operator_hours * p.operator_hour_cost + doctor_hours * p.doctor_hour_cost + camera_cost
    screened = int(n)

    return {
        "params": asdict(p),
        "simulated": {"patients": screened, "days": days, "scaled_to_year": scale != 1.0},
        "wait_capture_to_result_days": {
            "mean": round(float(wait_days.mean()), 3) if wait_days.size else None,
            "p95": round(float(np.percentile(wait_days, 95)), 3) if wait_days.size else None,
            "max": round(float(wait_days.max()), 3) if wait_days.size else None,
        },
        "phc_wait_minutes_mean": round(float(np.nanmean(phc_wait)), 2),
        "doctor_wait_days_mean": round(float(np.nanmean(doctor_wait) / day_len), 3) if np.isfinite(np.nanmean(doctor_wait)) else 0.0,
        "ophthalmologist_utilisation": round(float(doctor_hours / max(doctor_capacity_hours, 1e-6)), 3),
        "backlog_at_end": int(backlog * scale),
        "sent_to_ophthalmologist": int(to_doctor.sum() * scale),
        "sent_fraction": round(float(to_doctor.mean()), 4),
        "referable_cases": int(referable.sum() * scale),
        "referable_missed_by_ai": int(missed.sum() * scale),
        "referable_missed_by_reader": int(missed_by_reader.sum() * scale),
        "referable_unresulted_at_year_end": int(unresulted.sum() * scale),
        "referable_confirmed": int(confirmed.sum() * scale),
        "programme_sensitivity": round(float(confirmed.sum() / max(referable.sum(), 1)), 4),
        "unnecessary_referrals": int(unnecessary.sum() * scale),
        "operator_hours": round(operator_hours * scale, 1),
        "doctor_hours": round(doctor_hours * scale, 1),
        "cost_inr_total": round(cost * scale),
        "cost_inr_per_screened_patient": round(cost / max(screened, 1), 2),
        "patients_per_ophthalmologist_hour": round(screened / max(doctor_hours, 1e-6), 1),
    }


# -------------------------------------------------------- comparisons --

def compare(overrides: dict | None = None, measured_flag_rate=None, measured_retake_rate=None) -> dict:
    """Run the AI scenario and the no-AI baseline with the same random draws."""
    started = time.perf_counter()
    ai = params_from(overrides, measured_flag_rate=measured_flag_rate, measured_retake_rate=measured_retake_rate)
    ai.ai_enabled = True
    base = params_from(overrides, measured_flag_rate=measured_flag_rate, measured_retake_rate=measured_retake_rate)
    base.ai_enabled = False
    ai_result = simulate(ai)
    base_result = simulate(base)
    return {
        "ai": ai_result,
        "baseline": base_result,
        "delta": {
            "doctor_hours_saved": round(base_result["doctor_hours"] - ai_result["doctor_hours"], 1),
            "cost_saved_inr": round(base_result["cost_inr_total"] - ai_result["cost_inr_total"]),
            "wait_p95_days_change": round((ai_result["wait_capture_to_result_days"]["p95"] or 0)
                                          - (base_result["wait_capture_to_result_days"]["p95"] or 0), 3),
            "additional_missed_vs_baseline": _missed_total(ai_result) - _missed_total(base_result),
            "programme_sensitivity_ai": ai_result["programme_sensitivity"],
            "programme_sensitivity_baseline": base_result["programme_sensitivity"],
        },
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


def _missed_total(r: dict) -> int:
    """Every referable patient without a correct result by year end: missed by
    the AI, missed by the reader, or never read because of the backlog."""
    return r["referable_missed_by_ai"] + r["referable_missed_by_reader"] + r["referable_unresulted_at_year_end"]


def sweep(overrides: dict | None = None, cameras=(1, 2, 3), doctors=range(1, 9), roc_points=None,
          sample_fraction: float = 0.25, max_missed: int | None = None, max_missed_fraction: float = 0.20,
          max_p95_wait_days: float = 7.0, measured_flag_rate=None, measured_retake_rate=None, progress=None) -> dict:
    """Cameras per PHC x ophthalmologists x AI threshold along the measured ROC.

    Objective: minimum cost subject to missed referable cases below a limit
    (default: 20% of referable cases, i.e. programme sensitivity >= 80%, the
    district officer's choice) and 95th-percentile wait under 7 days. Returns
    every run plus the Pareto front of cost against missed cases, and the four
    slide numbers: doctors K with AI vs K' without, at cost C vs C'.
    """
    started = time.perf_counter()
    point = operating_point()
    roc_points = roc_points or point["external_test"]["roc_points_for_simulation"]
    runs = []
    total = len(cameras) * len(list(doctors)) * (len(roc_points) + 1)
    k = 0
    for cam in cameras:
        for doc in doctors:
            # Baseline (no AI) for this staffing.
            p = params_from(overrides, measured_flag_rate=measured_flag_rate, measured_retake_rate=measured_retake_rate)
            p.cameras_per_phc, p.operators_per_phc, p.ophthalmologists = cam, cam, doc
            p.ai_enabled, p.sample_fraction = False, sample_fraction
            r = simulate(p)
            runs.append(_summarise(r, cam, doc, None))
            k += 1
            if progress: progress(k, total)
            for rp in roc_points:
                p = params_from(overrides, measured_flag_rate=measured_flag_rate, measured_retake_rate=measured_retake_rate)
                p.cameras_per_phc, p.operators_per_phc, p.ophthalmologists = cam, cam, doc
                p.sensitivity, p.specificity = rp["sensitivity"], rp["specificity"]
                p.ai_enabled, p.sample_fraction = True, sample_fraction
                r = simulate(p)
                runs.append(_summarise(r, cam, doc, rp))
                k += 1
                if progress: progress(k, total)

    # Pareto front: cost vs missed (lower is better on both).
    ai_runs = [r for r in runs if r["ai"]]
    front = []
    for r in sorted(ai_runs, key=lambda x: (x["cost_inr_total"], x["missed_total"])):
        if not front or r["missed_total"] < front[-1]["missed_total"]:
            front.append(r)

    referable_cases = runs[0]["referable_cases"] if runs else 0
    if max_missed is None:
        max_missed = int(round(max_missed_fraction * referable_cases))

    def feasible(r):
        return (r["wait_p95_days"] is not None and r["wait_p95_days"] <= max_p95_wait_days
                and r["missed_total"] <= max_missed)

    best_ai = min((r for r in ai_runs if feasible(r)), key=lambda r: r["cost_inr_total"], default=None)
    best_base = min((r for r in runs if not r["ai"] and feasible(r)), key=lambda r: r["cost_inr_total"], default=None)
    slide = None
    if best_ai and best_base:
        slide = {
            "doctors_with_ai": best_ai["ophthalmologists"], "doctors_without_ai": best_base["ophthalmologists"],
            "cost_with_ai": best_ai["cost_inr_total"], "cost_without_ai": best_base["cost_inr_total"],
            "missed_with_ai": best_ai["missed_total"], "missed_without_ai": best_base["missed_total"],
            "operating_point_with_ai": best_ai["sensitivity_target"],
            "cameras_per_phc": best_ai["cameras_per_phc"],
        }
    out = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "sample_fraction": sample_fraction,
        "constraints": {"max_missed": max_missed, "max_missed_fraction": max_missed_fraction,
                        "referable_cases": referable_cases, "max_p95_wait_days": max_p95_wait_days},
        "coupled_from": {"model_version": point["model_version"], "fingerprint": point["calibration_fingerprint"],
                         "roc_points": roc_points},
        "runs": runs,
        "pareto_front": front,
        "best_ai": best_ai,
        "best_baseline": best_base,
        "slide_numbers": slide,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
    return out


def _summarise(r: dict, cam: int, doc: int, rp: dict | None) -> dict:
    return {
        "ai": rp is not None,
        "cameras_per_phc": cam,
        "ophthalmologists": doc,
        "sensitivity_target": rp["sensitivity_target"] if rp else None,
        "sensitivity": rp["sensitivity"] if rp else None,
        "specificity": rp["specificity"] if rp else None,
        "cost_inr_total": r["cost_inr_total"],
        "cost_inr_per_patient": r["cost_inr_per_screened_patient"],
        "referable_cases": r["referable_cases"],
        "missed_total": _missed_total(r),
        "missed_by_ai": r["referable_missed_by_ai"],
        "missed_by_reader": r["referable_missed_by_reader"],
        "unresulted": r["referable_unresulted_at_year_end"],
        "programme_sensitivity": r["programme_sensitivity"],
        "unnecessary_referrals": r["unnecessary_referrals"],
        "wait_mean_days": r["wait_capture_to_result_days"]["mean"],
        "wait_p95_days": r["wait_capture_to_result_days"]["p95"],
        "backlog_at_end": r["backlog_at_end"],
        "utilisation": r["ophthalmologist_utilisation"],
        "doctor_hours": r["doctor_hours"],
    }


# ---------------------------------------------------- background sweep --

_sweep_state = {"running": False, "progress": 0, "total": 0, "error": None}
_sweep_lock = threading.Lock()


def sweep_status() -> dict:
    cached = None
    if SWEEP_CACHE.exists():
        try:
            cached = json.load(open(SWEEP_CACHE, encoding="utf-8"))
        except (OSError, ValueError):
            cached = None
    return {**_sweep_state, "cached": cached is not None,
            "cached_at": cached["written_at"] if cached else None}


def cached_sweep() -> dict | None:
    if SWEEP_CACHE.exists():
        return json.load(open(SWEEP_CACHE, encoding="utf-8"))
    return None


def start_sweep(overrides: dict | None = None, **kwargs) -> bool:
    with _sweep_lock:
        if _sweep_state["running"]:
            return False
        _sweep_state.update({"running": True, "progress": 0, "total": 0, "error": None})

    def progress(k, total):
        _sweep_state["progress"], _sweep_state["total"] = k, total

    def worker():
        try:
            result = sweep(overrides, progress=progress, **kwargs)
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(SWEEP_CACHE, "w", encoding="utf-8") as handle:
                json.dump(result, handle)
        except Exception as exc:  # pragma: no cover
            _sweep_state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            _sweep_state["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return True
