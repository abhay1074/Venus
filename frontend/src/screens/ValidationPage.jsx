import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CheckCircle2, XCircle } from "lucide-react";
import { getOperatingPoint } from "../api/venus.js";

// Published references the numbers sit beside, with the same metric definitions.
const BENCHMARKS = [
  { name: "Gulshan et al. 2016 (JAMA), Messidor-2", sens: 0.903, spec: 0.981, auc: 0.99, n: "128k training images, adjudicated labels" },
  { name: "Problem-statement target", sens: 0.9, spec: 0.85, auc: 0.95, n: "referable DR, ICDR ≥ 2" },
];

export default function ValidationPage() {
  const [op, setOp] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    getOperatingPoint().then(setOp).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="card p-5 text-sm text-rose-700">{error}</div>;
  if (!op) return <div className="card p-5 text-sm text-slate-500">Loading the operating point…</div>;

  const t = op.external_test;
  const at = t.at_locked_threshold;
  const ci = t.ci95_bootstrap_2000;
  const sec = op.secondary_heldout;
  const gm = t.grade_metrics_for_contrast;
  const rel = t.reliability_diagram.filter((b) => b.n > 0).map((b) => ({ bin: `${b.bin[0].toFixed(1)}–${b.bin[1].toFixed(1)}`, confidence: b.confidence, accuracy: b.accuracy, n: b.n }));
  const roc = t.roc_points_for_simulation.map((p) => ({ ...p, fpr: 1 - p.specificity }));

  return (
    <div className="space-y-5">
      <div className="card p-5">
        <h2 className="card-title">Pre-registered validation protocol · grader {op.grader_tag} · model {op.model_version}</h2>
        <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm text-slate-700">
          <li><b>Calibration set:</b> {op.source.description} (n = {op.source.n_calibration}, {op.source.n_patients_calibration} patients). Its SHA-256 is the fingerprint the serving code verifies on every start.</li>
          <li><b>Platt scaling</b> fitted on it: ECE {op.calibration.ece_raw} → <b>{op.calibration.ece_calibrated}</b>. The raw grader output ranks; calibration is what makes P(referable) reportable as a probability.</li>
          <li><b>Threshold</b> chosen on the calibration set at 90% sensitivity and locked: <b>{op.thresholds.referable.toFixed(4)}</b> (85% alternative {op.thresholds.referable_85pc_alternative.toFixed(4)}, chosen at the same time; calibration-set specificity {op.thresholds.calibration_at_90.specificity.toFixed(3)}). Abstain band ± {op.thresholds.abstain_band}.</li>
          <li><b>External test:</b> {t.description} (n = {t.n}, {t.n_referable} referable), scored <b>{t.scored_once ? "once" : "again with --force (stated)"}</b>; 2,000 bootstrap resamples for 95% CIs.</li>
        </ol>
        <p className="mt-2 text-xs text-slate-500">calibration fingerprint {op.calibration_fingerprint}{op.external_test_fingerprint ? ` · external test fingerprint ${op.external_test_fingerprint}` : ""} · written {new Date(op.written_at).toLocaleString()}</p>
      </div>

      <div className="grid gap-3 md:grid-cols-4">
        <Metric label="Referable-DR AUC (external)" value={t.auc.toFixed(3)} ci={ci.auc} target={op.targets.auc} met={op.targets.auc_met} />
        <Metric label="Sensitivity @ locked threshold" value={at.sensitivity.toFixed(3)} ci={ci.sensitivity} target={op.targets.sensitivity} met={op.targets.sensitivity_met} />
        <Metric label="Specificity @ locked threshold" value={at.specificity.toFixed(3)} ci={ci.specificity} target={op.targets.specificity} met={op.targets.specificity_met} />
        <Metric label="Expected calibration error" value={t.ece.toFixed(3)} target={op.targets.ece} met={op.targets.ece_met} lowerIsBetter />
      </div>

      {op.additional_external_tests && Object.entries(op.additional_external_tests).map(([name, ext]) => {
        const e = ext.at_locked_threshold; const c = ext.ci95_bootstrap_2000;
        const roc90 = ext.roc_points_for_simulation.find((r) => r.sensitivity_target === 0.9);
        return (
          <div key={name} className="card p-4">
            <div className="label">Additional external test · {name.replace("external_test_", "")} · scored once at the locked threshold</div>
            <p className="mt-1 text-xs text-slate-500">{ext.description} (n = {ext.n}, {ext.n_referable} referable{ext.n_patients ? `, ${ext.n_patients} patients` : ""})</p>
            <div className="mt-2 grid grid-cols-2 gap-2 text-center text-sm md:grid-cols-5">
              <Small k="AUC" v={`${ext.auc.toFixed(3)} [${c.auc[0]}, ${c.auc[1]}]`} /><Small k="Sens" v={`${e.sensitivity.toFixed(3)} [${c.sensitivity[0]}, ${c.sensitivity[1]}]`} />
              <Small k="Spec" v={`${e.specificity.toFixed(3)} [${c.specificity[0]}, ${c.specificity[1]}]`} /><Small k="PPV @18%" v={e.ppv_at_indian_prevalence.toFixed(3)} /><Small k="ECE" v={ext.ece.toFixed(3)} />
            </div>
            {roc90 && <p className="mt-2 text-xs text-slate-600">On this set's own ROC, 90% sensitivity corresponds to specificity {roc90.specificity.toFixed(3)}: discrimination transfers across sources; the locked calibration over-refers here rather than missing cases — the measurement behind a site-specific calibration set before deployment.</p>}
          </div>
        );
      })}

      {(sec || gm) && (
        <div className="grid gap-3 md:grid-cols-2">
          {sec && (
            <div className="card p-4">
              <div className="label">Secondary held-out set · within-source</div>
              <p className="mt-1 text-xs text-slate-500">{sec.description} (n = {sec.n}, {sec.n_referable} referable)</p>
              <div className="mt-2 grid grid-cols-4 gap-2 text-center text-sm">
                <Small k="AUC" v={sec.auc.toFixed(3)} /><Small k="Sens" v={sec.at_locked_threshold.sensitivity.toFixed(3)} />
                <Small k="Spec" v={sec.at_locked_threshold.specificity.toFixed(3)} /><Small k="ECE" v={sec.ece.toFixed(3)} />
              </div>
            </div>
          )}
          {gm && (
            <div className="card p-4">
              <div className="label">Five-grade output, for contrast (not a claim)</div>
              <div className="mt-2 grid grid-cols-3 gap-2 text-center text-sm">
                <Small k="Quadratic weighted κ" v={gm.quadratic_weighted_kappa.toFixed(3)} /><Small k="Exact grade" v={(gm.exact_grade_accuracy * 100).toFixed(1) + "%"} /><Small k="Within one grade" v={(gm.within_one_grade * 100).toFixed(1) + "%"} />
              </div>
              {sec?.grade_metrics_for_contrast && <p className="mt-2 text-[11px] text-slate-500">Held-out EyePACS: QWK {sec.grade_metrics_for_contrast.quadratic_weighted_kappa.toFixed(3)}, exact {(sec.grade_metrics_for_contrast.exact_grade_accuracy * 100).toFixed(1)}%.</p>}
            </div>
          )}
        </div>
      )}

      <div className="grid gap-5 lg:grid-cols-2">
        <div className="card p-5">
          <div className="label">Reliability diagram · external test</div>
          <div className="mt-2 h-64">
            <ResponsiveContainer>
              <BarChart data={rel} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="confidence" name="mean predicted P" fill="#94a3b8" />
                <Bar dataKey="accuracy" name="observed referable fraction" fill="#0F766E" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="card p-5">
          <div className="label">Operating points along the measured ROC (fed to the district sweep)</div>
          <div className="mt-2 h-64">
            <ResponsiveContainer>
              <LineChart data={roc} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="fpr" type="number" domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} tick={{ fontSize: 11 }} label={{ value: "1 − specificity", position: "insideBottom", offset: -2, fontSize: 11 }} />
                <YAxis dataKey="sensitivity" domain={[0, 1]} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => Number(v).toFixed(3)} />
                <Line type="monotone" dataKey="sensitivity" stroke="#BE123C" dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <div className="card p-5">
          <div className="label">Subgroup · referred fraction by true grade (external test)</div>
          <table className="mt-2 w-full text-sm">
            <tbody>
              {Object.entries(t.referred_fraction_by_true_grade).map(([g, v]) => (
                <tr key={g} className="border-t border-venus-line"><td className="py-1">Grade {g}{g === "2" ? " (hardest referable class)" : ""}</td><td className="py-1 text-right font-semibold">{(v * 100).toFixed(1)}%</td></tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-slate-500">PPV at 18% Indian prevalence: <b>{at.ppv_at_indian_prevalence.toFixed(3)}</b> · NPV {at.npv_at_indian_prevalence.toFixed(3)} · referred fraction on this set {(at.referred_fraction * 100).toFixed(0)}% · at the 85% alternative: sens {t.at_85pc_threshold.sensitivity.toFixed(3)}, spec {t.at_85pc_threshold.specificity.toFixed(3)}.</p>
        </div>
        <div className="card p-5">
          <div className="label">Benchmark against published figures</div>
          <table className="mt-2 w-full text-sm">
            <thead className="text-left text-xs text-slate-500"><tr><th>System</th><th>Sens</th><th>Spec</th><th>AUC</th></tr></thead>
            <tbody>
              <tr className="border-t border-venus-line font-semibold text-venus-navy"><td>Venus AI ({op.grader_tag}, external test)</td><td>{at.sensitivity.toFixed(3)}</td><td>{at.specificity.toFixed(3)}</td><td>{t.auc.toFixed(3)}</td></tr>
              {BENCHMARKS.map((b) => <tr key={b.name} className="border-t border-venus-line"><td>{b.name}<div className="text-[11px] text-slate-500">{b.n}</div></td><td>{b.sens}</td><td>{b.spec}</td><td>{b.auc}</td></tr>)}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-slate-500">Published 90/98 used ~128,000 graded images. A missed target is reported with its CI, not a re-tuned threshold.</p>
        </div>
      </div>

      <div className="card p-5">
        <div className="label">Stated plainly</div>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-700">
          {op.notes.map((n) => <li key={n}>{n}</li>)}
          <li>Mild DR (grade 1 vs 0) is not a claim of this system; the referable decision (grade ≥ 2) is what the threshold, the CIs and the simulation describe.</li>
        </ul>
      </div>
    </div>
  );
}

function Small({ k, v }) {
  return <div className="rounded bg-slate-50 py-2 ring-1 ring-slate-200"><div className="text-lg font-bold text-venus-navy">{v}</div><div className="text-[11px] text-slate-500">{k}</div></div>;
}

function Metric({ label, value, ci, target, met, lowerIsBetter }) {
  return (
    <div className={`card p-4 ${met ? "" : "border-amber-300"}`}>
      <div className="label">{label}</div>
      <div className="mt-1 flex items-baseline gap-2"><span className="text-2xl font-extrabold text-venus-navy">{value}</span>{ci && <span className="text-xs text-slate-500">[{ci[0]}, {ci[1]}]</span>}</div>
      <div className={`mt-1 flex items-center gap-1 text-xs ${met ? "text-emerald-700" : "text-amber-800"}`}>
        {met ? <CheckCircle2 size={14} /> : <XCircle size={14} />} target {lowerIsBetter ? "<" : ">"} {target} · {met ? "met" : "missed, reported with CI"}
      </div>
    </div>
  );
}
