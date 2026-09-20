import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CheckCircle2, XCircle } from "lucide-react";
import { getOperatingPoint } from "../api/venus.js";

// Published references the numbers sit beside, with the same metric definitions.
const BENCHMARKS = [
  { name: "Gulshan et al. 2016 (JAMA), Messidor-2", sens: 0.903, spec: 0.981, auc: 0.99, n: "128k training images, adjudicated labels" },
  { name: "Problem-statement target", sens: 0.90, spec: 0.85, auc: 0.95, n: "referable DR, ICDR ≥ 2" },
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
  const rel = t.reliability_diagram.filter((b) => b.n > 0).map((b) => ({ bin: `${b.bin[0].toFixed(1)}–${b.bin[1].toFixed(1)}`, confidence: b.confidence, accuracy: b.accuracy, n: b.n }));
  const roc = t.roc_points_for_simulation.map((p) => ({ ...p, fpr: 1 - p.specificity }));

  return (
    <div className="space-y-5">
      <div className="card p-5">
        <h2 className="card-title">Pre-registered validation protocol</h2>
        <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm text-slate-700">
          <li>The frozen external set ({op.frozen_source.description}) was split <b>by patient</b> into a calibration half (n = {op.frozen_source.n_calibration}) and a test half (n = {op.frozen_source.n_test}). The split file's SHA-256 is the fingerprint the serving code verifies on every start.</li>
          <li>Platt scaling fitted on the calibration half: ECE {op.calibration.ece_raw} → <b>{op.calibration.ece_calibrated}</b>. The raw grader score ranks well but is not a probability; this is what makes P(referable) reportable.</li>
          <li>Threshold chosen on the calibration half at 90% sensitivity and locked: <b>{op.thresholds.referable.toFixed(4)}</b> (85% alternative {op.thresholds.referable_85pc_alternative.toFixed(4)}, chosen at the same time). Abstain band ± {op.thresholds.abstain_band}.</li>
          <li>Test half scored <b>{t.scored_once ? "once" : "again with --force (stated)"}</b>; 2,000 bootstrap resamples for 95% CIs.</li>
        </ol>
        <p className="mt-2 text-xs text-slate-500">Model {op.model_version} · fingerprint {op.calibration_fingerprint} · written {new Date(op.written_at).toLocaleString()}</p>
      </div>

      <div className="grid gap-3 md:grid-cols-4">
        <Metric label="Referable-DR AUC" value={t.auc.toFixed(3)} ci={ci.auc} target={op.targets.auc} met={op.targets.auc_met} />
        <Metric label="Sensitivity @ locked threshold" value={at.sensitivity.toFixed(3)} ci={ci.sensitivity} target={op.targets.sensitivity} met={op.targets.sensitivity_met} />
        <Metric label="Specificity @ locked threshold" value={at.specificity.toFixed(3)} ci={ci.specificity} target={op.targets.specificity} met={op.targets.specificity_met} />
        <Metric label="Expected calibration error" value={t.ece.toFixed(3)} target={op.targets.ece} met={op.targets.ece_met} lowerIsBetter />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <div className="card p-5">
          <div className="label">Reliability diagram · test half</div>
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
          <div className="label">Subgroup · referred fraction by true grade (test half)</div>
          <table className="mt-2 w-full text-sm">
            <tbody>
              {Object.entries(t.referred_fraction_by_true_grade).map(([g, v]) => (
                <tr key={g} className="border-t border-venus-line"><td className="py-1">Grade {g}{g === "2" ? " (hardest referable class)" : ""}</td><td className="py-1 text-right font-semibold">{(v * 100).toFixed(1)}%</td></tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-slate-500">PPV at 18% Indian prevalence: <b>{at.ppv_at_indian_prevalence.toFixed(3)}</b> · NPV {at.npv_at_indian_prevalence.toFixed(3)} · referred fraction on this 60%-referable set {(at.referred_fraction * 100).toFixed(0)}%.</p>
        </div>
        <div className="card p-5">
          <div className="label">Benchmark against published figures</div>
          <table className="mt-2 w-full text-sm">
            <thead className="text-left text-xs text-slate-500"><tr><th>System</th><th>Sens</th><th>Spec</th><th>AUC</th></tr></thead>
            <tbody>
              <tr className="border-t border-venus-line font-semibold text-venus-navy"><td>Venus AI (this build, EyePACS test half)</td><td>{at.sensitivity.toFixed(3)}</td><td>{at.specificity.toFixed(3)}</td><td>{t.auc.toFixed(3)}</td></tr>
              {BENCHMARKS.map((b) => <tr key={b.name} className="border-t border-venus-line"><td>{b.name}<div className="text-[11px] text-slate-500">{b.n}</div></td><td>{b.sens}</td><td>{b.spec}</td><td>{b.auc}</td></tr>)}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-slate-500">Published 90/98 used ~128,000 graded images; with public data, 90% sensitivity at 60–88% specificity is the realistic range. A missed target is reported with its CI, not a re-tuned threshold.</p>
        </div>
      </div>

      <div className="card p-5">
        <div className="label">Stated plainly</div>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-700">
          {op.notes.map((n) => <li key={n}>{n}</li>)}
          <li>Lesion segmentation in this build is classical (Frangi, morphology) and is reported as evidence, not as a validated detector; the rule grade it feeds is a consistency check on the CNN, and disagreement routes to a human.</li>
          <li>Mild DR (grade 1 vs 0) is not a claim of this system; the referable decision (grade ≥ 2) is what the threshold, the CIs and the simulation describe.</li>
        </ul>
      </div>
    </div>
  );
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
