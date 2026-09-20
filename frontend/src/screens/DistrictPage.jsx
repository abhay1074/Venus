import { useEffect, useState } from "react";
import { CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { Play, RefreshCw } from "lucide-react";
import { getSweep, simulate, startSweep } from "../api/venus.js";

const FIELDS = [
  ["annual_patients", "Annual patients", 1000],
  ["referable_prevalence", "Referable prevalence", 0.01],
  ["phcs", "PHCs in district", 1],
  ["cameras_per_phc", "Cameras per PHC", 1],
  ["ophthalmologists", "Ophthalmologists", 1],
  ["capture_minutes", "Capture + Stage 0, min", 0.5],
  ["tele_review_minutes", "Tele-review, min", 0.5],
  ["in_person_minutes", "In-person review, min", 1],
  ["doctor_hours_per_day", "Doctor hours / day", 0.5],
  ["human_reader_sensitivity", "Human reader sensitivity", 0.01],
  ["doctor_hour_cost", "Doctor cost, ₹/h", 100],
  ["operator_hour_cost", "Operator cost, ₹/h", 10],
];

export default function DistrictPage({ health }) {
  const [params, setParams] = useState({
    annual_patients: 100000, referable_prevalence: 0.06, phcs: 30, cameras_per_phc: 1, ophthalmologists: 3,
    capture_minutes: 4, tele_review_minutes: 3, in_person_minutes: 20, doctor_hours_per_day: 6,
    human_reader_sensitivity: 0.85, doctor_hour_cost: 1500, operator_hour_cost: 150,
  });
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [sweep, setSweep] = useState(null);
  const [constraints, setConstraints] = useState({ max_missed_fraction: 0.2, max_p95_wait_days: 7 });

  const loadSweep = async () => setSweep(await getSweep());
  useEffect(() => {
    loadSweep();
    const t = setInterval(async () => {
      const s = await getSweep();
      setSweep(s);
    }, 5000);
    return () => clearInterval(t);
  }, []);

  const run = async () => {
    setRunning(true);
    setError("");
    try {
      setResult(await simulate(params));
    } catch (e) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  };

  const coupled = result?.ai?.params;
  const sw = sweep?.result;
  const front = sw?.pareto_front || [];
  const aiRuns = (sw?.runs || []).filter((r) => r.ai);
  const baseRuns = (sw?.runs || []).filter((r) => !r.ai);

  return (
    <div className="space-y-5">
      <div className="card p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="card-title">District screening programme</h2>
            <p className="mt-1 max-w-3xl text-sm text-slate-600">
              A discrete-event model of one district for one year: Poisson arrivals at {params.phcs} PHCs, capture with a retake loop, AI triage,
              an ophthalmologist queue, discharge and recall. The AI block applies the grader's <b>measured</b> sensitivity and specificity from the
              frozen external test, and the human-review flag rate measured on stored screenings — nothing is typed in.
            </p>
          </div>
          <div className="rounded-lg bg-slate-50 p-3 text-xs ring-1 ring-slate-200">
            <div className="label">Coupled from Stage 2</div>
            <div className="mt-1">sensitivity <b>{(coupled?.sensitivity ?? health.measured?.sensitivity ?? 0.911).toFixed(3)}</b> · specificity <b>{(coupled?.specificity ?? 0.607).toFixed(3)}</b></div>
            <div>
              flag rate <b>{coupled ? coupled.flag_rate.toFixed(3) : health.measured?.used_by_simulation ? health.measured.flag_rate.toFixed(3) : health.measured?.validation_sample ? health.measured.validation_sample.flag_rate.toFixed(3) : "0.120 (default)"}</b> · retake{" "}
              <b>{coupled ? coupled.retake_probability.toFixed(3) : health.measured?.used_by_simulation ? health.measured.retake_rate.toFixed(3) : health.measured?.validation_sample ? health.measured.validation_sample.retake_rate.toFixed(3) : "0.050 (default)"}</b>
            </div>
            <div className="mt-1 text-[11px] text-slate-500">
              {health.measured?.used_by_simulation
                ? `measured on ${health.measured.n_gradable} stored screenings`
                : health.measured?.validation_sample
                ? `measured on ${health.measured.validation_sample.n} validation images (${health.measured.validation_sample.grader})`
                : `defaults until ${health.measured?.minimum_n ?? 10} screenings are stored (${health.measured?.n_gradable ?? 0} so far)`}
            </div>
          </div>
        </div>
        <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4 lg:grid-cols-6">
          {FIELDS.map(([key, label, step]) => (
            <label key={key} className="text-xs">
              <span className="label">{label}</span>
              <input className="input mt-1" type="number" step={step} value={params[key]} onChange={(e) => setParams({ ...params, [key]: Number(e.target.value) })} />
            </label>
          ))}
        </div>
        <div className="mt-4 flex items-center gap-3">
          <button className="btn-primary" onClick={run} disabled={running}>{running ? <RefreshCw size={16} className="animate-spin" /> : <Play size={16} />} Run one year, AI vs no-AI</button>
          {error && <span className="text-sm text-rose-700">{error}</span>}
          {result && <span className="text-xs text-slate-500">{result.elapsed_ms} ms for 2 × {result.ai.simulated.patients.toLocaleString()} patients</span>}
        </div>
      </div>

      {result && (
        <div className="grid gap-4 lg:grid-cols-2">
          <ScenarioCard title="With AI triage" r={result.ai} accent="teal" />
          <ScenarioCard title="Baseline: every image read by a human" r={result.baseline} accent="slate" />
          <div className="card p-4 lg:col-span-2">
            <div className="label">What the AI changes at this staffing</div>
            <div className="mt-2 grid grid-cols-2 gap-3 md:grid-cols-4">
              <Delta label="Ophthalmologist hours saved" value={result.delta.doctor_hours_saved.toLocaleString()} unit="h / year" good={result.delta.doctor_hours_saved > 0} />
              <Delta label="Cost saved" value={`₹${(result.delta.cost_saved_inr / 1e5).toFixed(1)} L`} unit="per year" good={result.delta.cost_saved_inr > 0} />
              <Delta label="95th-pct wait change" value={result.delta.wait_p95_days_change.toFixed(1)} unit="days" good={result.delta.wait_p95_days_change < 0} />
              <Delta label="Additional missed vs baseline" value={result.delta.additional_missed_vs_baseline} unit="referable cases" good={result.delta.additional_missed_vs_baseline <= 0} />
            </div>
            <p className="mt-3 text-xs text-slate-500">
              Programme sensitivity (referable cases correctly resulted within the year): <b>{(result.delta.programme_sensitivity_ai * 100).toFixed(0)}%</b> with AI vs{" "}
              <b>{(result.delta.programme_sensitivity_baseline * 100).toFixed(0)}%</b> without. Missed cases are not assumed: they emerge from the confusion matrix
              (AI misses ≈ 1 in 11 referable cases at the locked point; the human reader in both arms misses {(100 * (1 - params.human_reader_sensitivity)).toFixed(0)}% of what reaches them)
              plus every referable patient still in the backlog at year end.
            </p>
          </div>
        </div>
      )}

      <div className="card p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="card-title">Resource optimisation — Pareto front</h2>
            <p className="text-xs text-slate-500">
              Sweep over cameras per PHC (1–3) × ophthalmologists (1–8) × AI threshold along the measured ROC (80–97.5% sensitivity). Objective: minimum
              cost subject to a missed-case limit (default 20% of referable cases, i.e. programme sensitivity ≥ 80%) and a 95th-percentile wait under 7 days — both editable, because they are the district officer’s choice, not ours.
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-500">
            {sweep?.status?.running ? (
              <span className="flex items-center gap-2"><RefreshCw size={14} className="animate-spin" /> running {sweep.status.progress}/{sweep.status.total}</span>
            ) : sw ? (
              <span>cached {new Date(sw.written_at).toLocaleString()} · {sw.runs.length} runs · {(sw.elapsed_ms / 1000).toFixed(0)} s</span>
            ) : (
              <span>no sweep yet</span>
            )}
            <label className="flex items-center gap-1">max missed
              <input className="input w-16 py-1" type="number" step="0.05" min="0.05" max="0.5" value={constraints.max_missed_fraction} onChange={(e) => setConstraints({ ...constraints, max_missed_fraction: Number(e.target.value) })} /> of referable
            </label>
            <label className="flex items-center gap-1">p95 wait ≤
              <input className="input w-14 py-1" type="number" step="1" min="1" value={constraints.max_p95_wait_days} onChange={(e) => setConstraints({ ...constraints, max_p95_wait_days: Number(e.target.value) })} /> d
            </label>
            <button className="btn-secondary" onClick={async () => { await startSweep({ params, ...constraints }); loadSweep(); }} disabled={sweep?.status?.running}>Re-run sweep</button>
          </div>
        </div>

        {sw?.slide_numbers && (
          <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
            <Slide label="Ophthalmologists needed" ai={sw.slide_numbers.doctors_with_ai} base={sw.slide_numbers.doctors_without_ai} />
            <Slide label="Programme cost / year" ai={`₹${(sw.slide_numbers.cost_with_ai / 1e5).toFixed(1)} L`} base={`₹${(sw.slide_numbers.cost_without_ai / 1e5).toFixed(1)} L`} />
            <Slide label={`Referable cases missed (limit ${sw.constraints.max_missed})`} ai={sw.slide_numbers.missed_with_ai} base={sw.slide_numbers.missed_without_ai} />
            <Slide label="AI operating point" ai={`${(sw.slide_numbers.operating_point_with_ai * 100).toFixed(1)}% sens`} base="human only" />
          </div>
        )}
        {sw && !sw.slide_numbers && (
          <p className="mt-3 rounded-lg bg-amber-50 p-3 text-xs text-amber-900 ring-1 ring-amber-200">
            No AI configuration met both constraints (missed ≤ {sw.constraints.max_missed} of {sw.constraints.referable_cases} referable, p95 wait ≤ {sw.constraints.max_p95_wait_days} d). Relax a limit and re-run; the Pareto front below still shows the trade-off.
          </p>
        )}

        {sw && (
          <div className="mt-4 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" dataKey="cost_l" name="cost" unit=" L" label={{ value: "Programme cost, ₹ lakh / year", position: "bottom", offset: 0, fontSize: 12 }} tick={{ fontSize: 11 }} />
                <YAxis type="number" dataKey="missed_total" name="missed" label={{ value: "Referable cases missed / year", angle: -90, position: "insideLeft", fontSize: 12 }} tick={{ fontSize: 11 }} />
                <ZAxis type="number" dataKey="ophthalmologists" range={[30, 200]} name="doctors" />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} content={<PointTip />} />
                <Legend verticalAlign="top" height={24} wrapperStyle={{ fontSize: 12 }} />
                <Scatter name="AI configurations" data={aiRuns.map(withCost)} fill="#0F766E" fillOpacity={0.55} />
                <Scatter name="No-AI baselines" data={baseRuns.map(withCost)} fill="#94a3b8" fillOpacity={0.7} shape="triangle" />
                <Scatter name="Pareto front" data={front.map(withCost)} fill="#BE123C" line={{ stroke: "#BE123C" }} lineType="joint" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}
        {sw && (
          <div className="mt-2 overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="text-left text-slate-500"><tr><th className="py-1 pr-3">Front point</th><th className="pr-3">Cameras/PHC</th><th className="pr-3">Doctors</th><th className="pr-3">Sens target</th><th className="pr-3">Cost ₹L</th><th className="pr-3">Missed (AI / reader / backlog)</th><th className="pr-3">Programme sens</th><th className="pr-3">Unnecessary referrals</th><th className="pr-3">p95 wait d</th><th className="pr-3">Backlog</th><th>Utilisation</th></tr></thead>
              <tbody>
                {front.map((r, i) => (
                  <tr key={i} className="border-t border-venus-line">
                    <td className="py-1 pr-3">{i + 1}</td><td className="pr-3">{r.cameras_per_phc}</td><td className="pr-3">{r.ophthalmologists}</td><td className="pr-3">{(r.sensitivity_target * 100).toFixed(1)}%</td>
                    <td className="pr-3">{(r.cost_inr_total / 1e5).toFixed(1)}</td><td className="pr-3">{r.missed_total} <span className="text-slate-400">({r.missed_by_ai} / {r.missed_by_reader} / {r.unresulted})</span></td><td className="pr-3">{(r.programme_sensitivity * 100).toFixed(0)}%</td><td className="pr-3">{r.unnecessary_referrals.toLocaleString()}</td>
                    <td className="pr-3">{r.wait_p95_days?.toFixed(1)}</td><td className="pr-3">{r.backlog_at_end.toLocaleString()}</td><td>{(r.utilisation * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function withCost(r) {
  return { ...r, cost_l: Number((r.cost_inr_total / 1e5).toFixed(1)) };
}

function PointTip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const r = payload[0].payload;
  return (
    <div className="rounded-lg bg-white p-2 text-xs shadow ring-1 ring-slate-200">
      <div className="font-semibold">{r.ai ? `AI @ ${(r.sensitivity_target * 100).toFixed(1)}% sens` : "No AI"} · {r.cameras_per_phc} cam · {r.ophthalmologists} doctors</div>
      <div>cost ₹{r.cost_l} L · missed {r.missed_total} · p95 wait {r.wait_p95_days?.toFixed(1)} d · backlog {r.backlog_at_end}</div>
    </div>
  );
}

function ScenarioCard({ title, r, accent }) {
  const w = r.wait_capture_to_result_days;
  return (
    <div className="card p-4">
      <div className={`text-sm font-bold ${accent === "teal" ? "text-venus-teal" : "text-slate-600"}`}>{title}</div>
      <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
        <Row k="Wait, capture → result" v={`${w.mean?.toFixed(1)} d mean · ${w.p95?.toFixed(1)} d p95`} />
        <Row k="Ophthalmologist utilisation" v={`${(r.ophthalmologist_utilisation * 100).toFixed(0)}%`} />
        <Row k="Backlog at year end" v={r.backlog_at_end.toLocaleString()} />
        <Row k="Sent to ophthalmologist" v={`${r.sent_to_ophthalmologist.toLocaleString()} (${(r.sent_fraction * 100).toFixed(0)}%)`} />
        <Row k="Referable cases in population" v={r.referable_cases.toLocaleString()} />
        <Row k="Missed by AI / reader / backlog" v={`${r.referable_missed_by_ai} / ${r.referable_missed_by_reader} / ${r.referable_unresulted_at_year_end}`} />
        <Row k="Programme sensitivity" v={`${(r.programme_sensitivity * 100).toFixed(0)}%`} />
        <Row k="Unnecessary referrals" v={r.unnecessary_referrals.toLocaleString()} />
        <Row k="Doctor hours / year" v={r.doctor_hours.toLocaleString()} />
        <Row k="Cost per screened patient" v={`₹${r.cost_inr_per_screened_patient}`} />
        <Row k="Patients per doctor-hour" v={r.patients_per_ophthalmologist_hour} />
      </div>
    </div>
  );
}

function Row({ k, v }) {
  return (
    <div className="flex flex-col"><span className="text-[11px] uppercase tracking-wide text-slate-500">{k}</span><span className="font-semibold text-venus-navy">{v}</span></div>
  );
}

function Delta({ label, value, unit, good }) {
  return (
    <div className={`rounded-lg p-3 ring-1 ${good ? "bg-emerald-50 ring-emerald-200" : "bg-rose-50 ring-rose-200"}`}>
      <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="text-xl font-extrabold text-venus-navy">{value}</div>
      <div className="text-xs text-slate-500">{unit}</div>
    </div>
  );
}

function Slide({ label, ai, base }) {
  return (
    <div className="rounded-lg bg-venus-navy p-3 text-white">
      <div className="text-[11px] uppercase tracking-wide text-slate-300">{label}</div>
      <div className="mt-1 text-2xl font-black">{ai}</div>
      <div className="text-xs text-slate-300">vs {base} without AI</div>
    </div>
  );
}
