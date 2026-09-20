import { useEffect, useState } from "react";
import { CalendarCheck, MessageSquare, RefreshCw, UserPlus } from "lucide-react";
import { createAppointment, createIntake, getFacilities, getWorklist, recordOutcome, reportUrl } from "../api/venus.js";

const VILLAGES = ["Kharia", "Belpur", "Rampur", "Sonpur", "Dhanora", "Mohanpur"];
const SYMPTOMS = [["blurred_vision", "Blurred vision"], ["floaters", "Floaters"], ["sudden_vision_loss", "Sudden vision loss"]];
const TIER_STYLE = {
  P0: "bg-slate-100 text-slate-800", P1: "bg-rose-100 text-rose-900", P2: "bg-orange-100 text-orange-900",
  P3: "bg-amber-100 text-amber-900", P4: "bg-emerald-100 text-emerald-900",
};

export default function AppointmentsPage({ sessionId, lastResult }) {
  const [form, setForm] = useState({
    name: "", phone: "", age: "", sex: "F", village: "Kharia", phc: "PHC 7", diabetes_years: "", hba1c: "",
    insulin: false, hypertension: false, pregnant: false, last_eye_exam: "", symptoms: [], consent: false,
  });
  const [session, setSession] = useState(sessionId || "");
  const [booking, setBooking] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [worklist, setWorklist] = useState([]);
  const [facilities, setFacilities] = useState([]);

  const refresh = async () => {
    setWorklist(await getWorklist());
    setFacilities(await getFacilities());
  };
  useEffect(() => {
    refresh();
  }, []);
  useEffect(() => {
    if (sessionId) setSession(sessionId);
  }, [sessionId]);

  const toggle = (id) => setForm((f) => ({ ...f, symptoms: f.symptoms.includes(id) ? f.symptoms.filter((x) => x !== id) : [...f.symptoms, id] }));

  const submit = async () => {
    setBusy(true);
    setError("");
    try {
      const intake = {
        ...form,
        age: form.age === "" ? null : Number(form.age),
        diabetes_years: form.diabetes_years === "" ? null : Number(form.diabetes_years),
        hba1c: form.hba1c === "" ? null : Number(form.hba1c),
      };
      const { patient_id } = await createIntake(intake);
      const result = await createAppointment({ patient_id, session_id: session || null, intake });
      setBooking({ patient_id, ...result });
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const outcome = async (id, value) => {
    await recordOutcome(id, value);
    await refresh();
  };

  return (
    <div className="grid gap-5 lg:grid-cols-[380px_1fr]">
      <div className="space-y-4">
        <div className="card">
          <div className="card-head">
            <h2 className="card-title flex items-center gap-2"><UserPlus size={16} /> Patient intake</h2>
            <p className="text-xs text-slate-500">Under two minutes; works without an image, so a camp organiser can pre-register.</p>
          </div>
          <div className="grid grid-cols-2 gap-3 p-4 text-sm">
            <Field label="Name" value={form.name} onChange={(v) => setForm({ ...form, name: v })} />
            <Field label="Phone" value={form.phone} onChange={(v) => setForm({ ...form, phone: v })} />
            <Field label="Age" type="number" value={form.age} onChange={(v) => setForm({ ...form, age: v })} />
            <label><span className="label">Sex</span><select className="input mt-1" value={form.sex} onChange={(e) => setForm({ ...form, sex: e.target.value })}><option>F</option><option>M</option><option>Other</option></select></label>
            <label><span className="label">Village</span><select className="input mt-1" value={form.village} onChange={(e) => setForm({ ...form, village: e.target.value })}>{VILLAGES.map((v) => <option key={v}>{v}</option>)}</select></label>
            <Field label="PHC" value={form.phc} onChange={(v) => setForm({ ...form, phc: v })} />
            <Field label="Diabetes, years" type="number" value={form.diabetes_years} onChange={(v) => setForm({ ...form, diabetes_years: v })} />
            <Field label="Last HbA1c" type="number" value={form.hba1c} onChange={(v) => setForm({ ...form, hba1c: v })} />
            <Field label="Last eye exam" type="date" value={form.last_eye_exam} onChange={(v) => setForm({ ...form, last_eye_exam: v })} />
            <div className="flex flex-col justify-end gap-1 text-xs">
              <label className="flex items-center gap-2"><input type="checkbox" checked={form.insulin} onChange={(e) => setForm({ ...form, insulin: e.target.checked })} /> Insulin</label>
              <label className="flex items-center gap-2"><input type="checkbox" checked={form.hypertension} onChange={(e) => setForm({ ...form, hypertension: e.target.checked })} /> Hypertension</label>
              <label className="flex items-center gap-2"><input type="checkbox" checked={form.pregnant} onChange={(e) => setForm({ ...form, pregnant: e.target.checked })} /> Pregnant</label>
            </div>
            <div className="col-span-2">
              <span className="label">Symptoms</span>
              <div className="mt-1 flex flex-wrap gap-2">
                {SYMPTOMS.map(([id, l]) => (
                  <button key={id} onClick={() => toggle(id)} className={`pill ring-1 ${form.symptoms.includes(id) ? "bg-amber-100 text-amber-900 ring-amber-300" : "bg-white text-slate-600 ring-slate-300"}`}>{l}</button>
                ))}
              </div>
            </div>
            <label className="col-span-2"><span className="label">Screening session (optional)</span><input className="input mt-1 font-mono text-xs" placeholder="VS-…" value={session} onChange={(e) => setSession(e.target.value)} /></label>
            {lastResult && !session && (
              <button className="col-span-2 text-left text-xs text-venus-teal hover:underline" onClick={() => setSession(lastResult.session_id)}>Use last screening {lastResult.session_id}</button>
            )}
            <label className="col-span-2 flex items-start gap-2 rounded-lg bg-slate-50 p-2 text-xs ring-1 ring-slate-200">
              <input type="checkbox" checked={form.consent} onChange={(e) => setForm({ ...form, consent: e.target.checked })} className="mt-0.5" />
              <span>The patient consents to their details and retinal image being stored for screening and follow-up. Consent time is recorded.</span>
            </label>
            <button className="btn-primary col-span-2 justify-center" disabled={busy || !form.consent} onClick={submit}>
              {busy ? <RefreshCw size={16} className="animate-spin" /> : <CalendarCheck size={16} />} Compute tier and book the earliest slot
            </button>
            {error && <p className="col-span-2 text-xs text-rose-700">{error}</p>}
          </div>
        </div>

        {booking && (
          <div className="card p-4 text-sm">
            <div className="flex items-center justify-between">
              <span className={`pill ${TIER_STYLE[booking.tier.tier]}`}>{booking.tier.tier} · {booking.tier.label}</span>
              <span className="font-mono text-xs text-slate-500">{booking.patient_id}</span>
            </div>
            <ul className="mt-2 space-y-1 text-xs text-slate-700">{booking.tier.reasons.map((r) => <li key={r}>• {r}</li>)}</ul>
            {booking.appointment ? (
              <div className="mt-3 rounded-lg bg-emerald-50 p-3 ring-1 ring-emerald-200">
                <div className="font-semibold text-emerald-900">{booking.appointment.status === "booked" ? "Booked" : "Waitlisted"}</div>
                <div className="text-xs text-emerald-900">{booking.appointment.facility_name} · {booking.appointment.starts_at ? new Date(booking.appointment.starts_at).toLocaleString() : "no slot inside the deadline yet"}</div>
                {booking.appointment.bump_history.length > 0 && <div className="mt-1 text-xs text-amber-800">Bumped a lower-priority booking to make room (logged).</div>}
                {booking.appointment.sms_log.map((s, i) => (
                  <div key={i} className="mt-2 flex items-start gap-2 rounded bg-white p-2 text-xs text-slate-700 ring-1 ring-slate-200"><MessageSquare size={14} className="mt-0.5 shrink-0 text-venus-teal" /> {s.text}</div>
                ))}
              </div>
            ) : (
              <div className="mt-3 rounded-lg bg-slate-50 p-3 text-xs ring-1 ring-slate-200">{booking.note}</div>
            )}
          </div>
        )}

        <div className="card p-4">
          <div className="label">Facilities · free slots (45-day calendar)</div>
          <ul className="mt-2 space-y-1 text-xs">
            {facilities.map((f) => <li key={f.id} className="flex justify-between"><span>{f.name} <span className="text-slate-400">({f.type})</span></span><b>{f.free_slots}</b></li>)}
          </ul>
          <p className="mt-2 text-[11px] text-slate-500">Slot capacity per facility comes from the same numbers the district sweep produces, so the schedule the simulation says is feasible is the one the system runs.</p>
        </div>
      </div>

      <div className="card">
        <div className="card-head flex items-center justify-between">
          <div>
            <h2 className="card-title">Doctor worklist</h2>
            <p className="text-xs text-slate-500">Sorted by tier, then risk, with ageing so a waiting P3 outranks a fresh P2. Record the outcome; a no-show re-queues at the same tier with its waiting time preserved.</p>
          </div>
          <button className="btn-secondary" onClick={refresh}><RefreshCw size={14} /></button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase text-slate-500">
              <tr><th className="px-4 py-2">Tier</th><th className="px-4 py-2">Patient</th><th className="px-4 py-2">Risk</th><th className="px-4 py-2">Slot</th><th className="px-4 py-2">Facility</th><th className="px-4 py-2">Status</th><th className="px-4 py-2">Report</th><th className="px-4 py-2">Outcome</th></tr>
            </thead>
            <tbody>
              {worklist.length === 0 && <tr><td colSpan={8} className="px-4 py-8 text-center text-slate-500">No appointments yet.</td></tr>}
              {worklist.map((a) => (
                <tr key={a.id} className="border-t border-venus-line">
                  <td className="px-4 py-2"><span className={`pill ${TIER_STYLE[a.tier]}`}>{a.tier} {a.tier_label}</span></td>
                  <td className="px-4 py-2">{a.patient_name || a.patient_id}<div className="text-[11px] text-slate-500">{a.village}</div></td>
                  <td className="px-4 py-2">{a.risk_score?.toFixed(2)}</td>
                  <td className="px-4 py-2 text-xs">{a.starts_at ? new Date(a.starts_at).toLocaleString() : "—"}</td>
                  <td className="px-4 py-2 text-xs">{a.facility_name || "—"}</td>
                  <td className="px-4 py-2 text-xs">{a.outcome && a.outcome !== a.status ? `${a.status} · ${a.outcome}` : a.status}{a.bump_history?.length ? " · bumped" : ""}</td>
                  <td className="px-4 py-2 text-xs">{a.session_id ? <a className="text-venus-blue hover:underline" href={reportUrl(a.session_id)} target="_blank" rel="noreferrer">PDF</a> : "—"}</td>
                  <td className="px-4 py-2">
                    {a.status === "booked" && (
                      <select className="input py-1 text-xs" defaultValue="" onChange={(e) => e.target.value && outcome(a.id, e.target.value)}>
                        <option value="">record…</option><option value="confirmed">confirmed</option><option value="treated">treated</option><option value="referred_onward">referred onward</option><option value="no_show">no-show</option>
                      </select>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function Field({ label, value, onChange, type = "text" }) {
  return (
    <label><span className="label">{label}</span><input className="input mt-1" type={type} value={value} onChange={(e) => onChange(e.target.value)} /></label>
  );
}
