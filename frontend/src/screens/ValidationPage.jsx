import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CheckCircle2, XCircle } from "lucide-react";
import { getOperatingPoint, getValidationExtras } from "../api/venus.js";

// Published references the numbers sit beside, with the same metric definitions.
const BENCHMARKS = [
  { name: "Gulshan et al. 2016 (JAMA), Messidor-2", sens: 0.903, spec: 0.981, auc: 0.99, n: "128k training images, adjudicated labels" },
  { name: "Problem-statement target", sens: 0.9, spec: 0.85, auc: 0.95, n: "referable DR, ICDR ≥ 2" },
];

export default function ValidationPage() {
  const [op, setOp] = useState(null);
  const [extras, setExtras] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    getOperatingPoint().then(setOp).catch((e) => setError(e.message));
    getValidationExtras().then(setExtras).catch(() => setExtras({}));
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

      {extras?.review_policy && <ReviewPolicyCard policy={extras.review_policy} flags={extras.validation_flags} />}

      {(extras?.lesion_thresholds || extras?.ensemble_check) && (
        <div className="grid gap-5 lg:grid-cols-2">
          {extras.lesion_thresholds && <LesionCard lesion={extras.lesion_thresholds} />}
          {extras.ensemble_check && <EnsembleCard check={extras.ensemble_check} />}
        </div>
      )}

      {extras?.site_calibration && <SiteCalibrationCard site={extras.site_calibration} />}

      {extras?.experiments?.lesion_unet_1024 && <UnetExperimentCard exp={extras.experiments.lesion_unet_1024} />}

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

function ReviewPolicyCard({ policy, flags }) {
  const pct = (x) => `${(x * 100).toFixed(1)}%`;
  const reasons = Object.entries(policy.flag_reasons || {});
  const att = flags?.attention_agreement;
  return (
    <div className="card p-5">
      <div className="label">Human-review policy · chosen on the validation sample, served as config/review_policy.json</div>
      <p className="mt-1 text-xs text-slate-500">{policy.chosen_on} · grader {policy.grader} · written {new Date(policy.written_at).toLocaleString()}</p>
      <div className="mt-3 grid grid-cols-2 gap-2 text-center text-sm md:grid-cols-5">
        <Small k="abstain band (logit, ± around threshold)" v={`±${policy.abstain_band_logit}`} />
        <Small k="attention floor (referable calls)" v={policy.attention_floor} />
        <Small k="resulting flag rate" v={pct(policy.resulting_flag_rate)} />
        <Small k="CNN error rate: flagged vs not" v={`${pct(policy.error_rate_flagged)} / ${pct(policy.error_rate_unflagged)}`} />
        <Small k="of CNN errors flagged" v={pct(policy.share_of_cnn_errors_flagged)} />
      </div>
      <p className="mt-3 text-sm text-slate-700">
        Flags by reason: {reasons.map(([k, v]) => `${k} ${v}`).join(", ")}.
        {att && ` Attention agreement on referable calls: median ${att.correct?.median} when the CNN is right (n = ${att.correct?.n}) vs ${att.incorrect?.median} when it is wrong (n = ${att.incorrect?.n}).`}
        {" "}The attention floor is the highest cut at which at least 60% of the flagged referable calls are CNN errors; the band is symmetric in logit space because ±0.05 in probability is not, at a threshold near 0.1. The district simulation uses this flag rate.
      </p>
    </div>
  );
}

function LesionCard({ lesion }) {
  const keys = ["MA", "HE", "EX", "SE"];
  const names = { MA: "microaneurysms", HE: "hemorrhages", EX: "hard exudates", SE: "soft exudates" };
  return (
    <div className="card p-5">
      <div className="label">Lesion U-Net · DDR test split, scored once · thresholds from DDR valid (max F1)</div>
      <table className="mt-2 w-full text-sm">
        <thead className="text-left text-xs text-slate-500"><tr><th>lesion</th><th>pixel AUPR</th><th>Dice at threshold</th><th>threshold</th></tr></thead>
        <tbody>
          {keys.map((k) => (
            <tr key={k} className="border-t border-venus-line"><td>{k} <span className="text-xs text-slate-500">{names[k]}</span></td><td>{lesion.test_aupr?.[k]?.toFixed(3)}</td><td>{lesion.test_dice?.[k]?.toFixed(3)}</td><td>{lesion.thresholds?.[k]}</td></tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-slate-500">{lesion.tag}{lesion.frame_size ? ` · ${lesion.frame_size} px frames` : " · 512 px frames"} · written {new Date(lesion.written_at).toLocaleString()}. Microaneurysms are 1–3 px at 512 px: the weakest class, stated as such.</p>
    </div>
  );
}

function EnsembleCard({ check }) {
  const rows = check.comparisons || [];
  const tags = rows[0] ? Object.keys(rows[0].single) : [];
  return (
    <div className="card p-5">
      <div className="label">Measured and not shipped · a second seed, test-time augmentation</div>
      <p className="mt-1 text-xs text-slate-500">{check.question} Calibration and validation sets only; the external tests were not re-scored.</p>
      <table className="mt-2 w-full text-sm">
        <thead className="text-left text-xs text-slate-500"><tr><th>set</th><th>TTA</th>{tags.map((t) => <th key={t}>{t}</th>)}<th>mean ensemble</th><th>Δ vs best single (95% CI)</th></tr></thead>
        <tbody>
          {rows.map((r, i) => {
            const e = r.ensemble_mean || {}; const d = e.paired_auc_difference || {};
            return (
              <tr key={i} className="border-t border-venus-line">
                <td>{r.manifest} <span className="text-xs text-slate-500">n = {r.n}</span></td><td>{r.tta ? "on" : "off"}</td>
                {tags.map((t) => <td key={t}>{r.single[t]?.auc?.toFixed(3)}</td>)}
                <td className="font-semibold">{e.auc?.toFixed(3)}</td><td>{d.mean > 0 ? "+" : ""}{d.mean?.toFixed(4)} [{d.ci95?.[0]}, {d.ci95?.[1]}]</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-slate-500">Referable-DR AUC. The ensemble adds about what TTA adds and nothing on top of it, at twice the CPU cost per image: the served grader stays one network, and the external test's single scoring stands.</p>
    </div>
  );
}

function SiteCalibrationCard({ site }) {
  const L = site.locked_operating_point; const F = site.site_calibration_full_half;
  const sizes = Object.entries(site.site_calibration_by_sample_size || {}).filter(([, v]) => v);
  const f3 = (x) => x.toFixed(3);
  const ci = (m) => `${f3(m.mean)} [${f3(m.p5)}, ${f3(m.p95)}]`;
  return (
    <div className="card p-5">
      <div className="label">What a site calibration set buys · Messidor-2 · evaluation of a deployment step, served point unchanged</div>
      <p className="mt-1 text-xs text-slate-500">{site.protocol}. n = {site.n_images} images, {site.n_patients} patients, {site.n_referable} referable.</p>
      <table className="mt-2 w-full text-sm">
        <thead className="text-left text-xs text-slate-500"><tr><th>operating point</th><th>sensitivity</th><th>specificity</th><th>ECE</th></tr></thead>
        <tbody>
          <tr className="border-t border-venus-line"><td className="py-1">locked (EyePACS calibration set), on the held-out halves</td><td>{f3(L.sensitivity.mean)}</td><td>{f3(L.specificity.mean)}</td><td>{f3(L.ece.mean)}</td></tr>
          <tr className="border-t border-venus-line font-semibold text-venus-navy"><td className="py-1">site calibration on the other half (n ≈ {F.n_mean})</td><td>{f3(F.sensitivity.mean)}</td><td>{f3(F.specificity.mean)}</td><td>{f3(F.ece.mean)}</td></tr>
          {sizes.map(([n, s]) => <tr key={n} className="border-t border-venus-line"><td className="py-1">site sample of {n} labelled images</td><td>{ci(s.sensitivity)}</td><td>{ci(s.specificity)}</td><td>{f3(s.ece.mean)}</td></tr>)}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-slate-500">Brackets: 5th–95th percentile over the repeats. Discrimination transfers across acquisition sources; calibration does not. Re-fitting Platt scaling and the 90% threshold on a few hundred labelled site images restores the intended operating point — the deployment step, with its price.</p>
    </div>
  );
}

function UnetExperimentCard({ exp }) {
  const d = exp.decision || {}; const w = d.validation_with_1024_for_MA; const o = d.validation_512_only;
  if (!w || !o) return null;
  const pct = (x) => `${(x * 100).toFixed(1)}%`;
  const row = (label, a, b) => <tr className="border-t border-venus-line"><td className="py-1">{label}</td><td>{a}</td><td>{b}</td></tr>;
  return (
    <div className="card p-5">
      <div className="label">Measured and not shipped · a 1024 px lesion network for microaneurysms</div>
      <p className="mt-1 text-xs text-slate-500">{exp.architecture} · {exp.epochs} epochs · DDR test scored once: MA AUPR {exp.test_aupr.MA} (512 px network: 0.079), HE {exp.test_aupr.HE}, EX {exp.test_aupr.EX}, SE {exp.test_aupr.SE}. Served for MA only on the same {o.n_gradable} raw validation images:</p>
      <table className="mt-2 w-full text-sm">
        <thead className="text-left text-xs text-slate-500"><tr><th></th><th>512 px only (served)</th><th>+ 1024 px for MA</th></tr></thead>
        <tbody>
          {row("rule grader alone, exact / within one", `${pct(o.rule_grader_alone.exact_grade_agreement_with_truth)} / ${pct(o.rule_grader_alone.within_one_of_truth)}`, `${pct(w.rule_grader_alone.exact_grade_agreement_with_truth)} / ${pct(w.rule_grader_alone.within_one_of_truth)}`)}
          {row("attention agreement, median correct / incorrect", `${o.attention_agreement.correct.median} / ${o.attention_agreement.incorrect.median}`, `${w.attention_agreement.correct.median} / ${w.attention_agreement.incorrect.median}`)}
          {row("review policy: floor · flag rate", `${o.review_policy.attention_floor} · ${pct(o.review_policy.resulting_flag_rate)}`, `${w.review_policy.attention_floor} · ${pct(w.review_policy.resulting_flag_rate)}`)}
          {row("CNN error rate, flagged vs unflagged", `${pct(o.review_policy.error_rate_flagged)} vs ${pct(o.review_policy.error_rate_unflagged)}`, `${pct(w.review_policy.error_rate_flagged)} vs ${pct(w.review_policy.error_rate_unflagged)}`)}
          {row("share of CNN referable errors flagged", pct(o.review_policy.share_of_cnn_errors_flagged), pct(w.review_policy.share_of_cnn_errors_flagged))}
          {row("CPU time per image, median", `${(d.cpu_timing_median_ms["512_only"] / 1000).toFixed(1)} s`, `${(d.cpu_timing_median_ms.with_1024_for_MA / 1000).toFixed(1)} s`)}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-slate-500">{d.why}</p>
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
