import { useEffect, useState } from "react";
import { AlertTriangle, CalendarPlus, CheckCircle2, Clock, FileText, ShieldAlert, ShieldCheck } from "lucide-react";
import { reportUrl } from "../api/venus.js";

const GRADE_LABELS = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"];
const OVERLAYS = [
  ["original", "Original"],
  ["enhanced", "Stage 0 output"],
  ["lesions", "Lesion overlay"],
  ["vessels", "Vessels"],
  ["gradcam_referable", "Grad-CAM · referable"],
  ["gradcam_grade", "Grad-CAM · grade"],
];
const TIER_STYLE = {
  P0: "bg-slate-100 text-slate-800 ring-slate-300",
  P1: "bg-rose-100 text-rose-900 ring-rose-300",
  P2: "bg-orange-100 text-orange-900 ring-orange-300",
  P3: "bg-amber-100 text-amber-900 ring-amber-300",
  P4: "bg-emerald-100 text-emerald-900 ring-emerald-300",
};

export default function ResultView({ result, loading, onBook }) {
  const [view, setView] = useState("lesions");
  useEffect(() => {
    if (result?.accepted) setView("lesions");
  }, [result?.session_id]);

  if (loading) {
    return (
      <div className="card p-5">
        <Header />
        <div className="mt-5 space-y-3">
          {["Stage 0 · gate + quality", "Stage 1 · segmentation", "Stage 2 · CNN + rule grader", "Stage 3 · Grad-CAM + report"].map((s) => (
            <div key={s} className="flex items-center gap-3 text-sm text-slate-500">
              <div className="h-2 w-2 animate-pulse rounded-full bg-venus-teal" /> {s}
            </div>
          ))}
          <div className="h-64 animate-pulse rounded-lg bg-slate-100" />
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="card p-5">
        <Header />
        <div className="mt-5 rounded-lg bg-slate-50 p-5 ring-1 ring-slate-200">
          <ShieldCheck size={26} className="text-venus-navy" />
          <p className="mt-3 font-semibold text-venus-navy">Awaiting an image</p>
          <p className="mt-1 text-sm leading-6 text-slate-500">
            One image enters, one annotated report leaves: quality verdict, ICDR grade from two independent graders, calibrated
            referable probability, lesion overlay, Grad-CAM with an attention-agreement score, and a priority tier.
          </p>
        </div>
      </div>
    );
  }

  const s0 = result.stage0;
  if (!result.accepted) {
    return (
      <div className="card p-5">
        <Header />
        <div className="mt-5 rounded-lg bg-amber-50 p-5 ring-1 ring-amber-300">
          <div className="flex items-start gap-3">
            <ShieldAlert className="mt-0.5 shrink-0 text-amber-700" size={22} />
            <div>
              <p className="text-lg font-bold text-amber-900">{s0.modality.accepted ? "Retake requested" : "Not a retinal image"}</p>
              <p className="mt-1 text-sm text-amber-900">{result.stop_reason}</p>
              <p className="mt-2 text-xs text-amber-800">Nothing diagnostic ran. Tier {result.stage5.tier} — {result.stage5.label} ({result.stage5.deadline_text}).</p>
            </div>
          </div>
        </div>
        <Stage0Panel s0={s0} />
        <Timing timing={result.timing_ms} />
      </div>
    );
  }

  const { cnn, rule, fusion } = result.stage2;
  const s3 = result.stage3;
  const tier = result.stage5;
  const agreement = s3.attention_agreement;
  const verdictClass = fusion.flag_for_review ? "bg-amber-50 ring-amber-300" : fusion.referable ? "bg-rose-50 ring-rose-300" : "bg-emerald-50 ring-emerald-300";

  return (
    <div className="card">
      <div className="p-5">
        <Header session={result.session_id} />
        <div className={`mt-4 rounded-lg p-4 ring-1 ${verdictClass}`}>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="text-2xl font-extrabold text-venus-navy">
                {fusion.referable ? "Referable DR" : "Not referable"}
                {fusion.flag_for_review && <span className="ml-3 text-base font-bold text-amber-800">· human review</span>}
              </div>
              <div className="mt-1 text-sm text-slate-700">
                ICDR grade <b>{fusion.grade}</b> — {GRADE_LABELS[fusion.grade]} · {fusion.agreement_text}
                {fusion.abstain && <span className="ml-2 font-semibold text-amber-800">· abstain band</span>}
              </div>
            </div>
            <div className="text-right">
              <div className="text-3xl font-black text-venus-navy">{fusion.p_referable.toFixed(2)}</div>
              <div className="text-xs text-slate-500">P(referable), calibrated · threshold {fusion.threshold.toFixed(2)}</div>
            </div>
          </div>
          <ProbabilityBar p={fusion.p_referable} threshold={fusion.threshold} band={fusion.abstain_band} />
          {fusion.flag_reasons.length > 0 && (
            <ul className="mt-3 space-y-1 text-sm text-amber-900">
              {fusion.flag_reasons.map((r) => (
                <li key={r} className="flex items-start gap-2"><AlertTriangle size={14} className="mt-0.5 shrink-0" /> {r}</li>
              ))}
            </ul>
          )}
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <span className={`pill ring-1 ${TIER_STYLE[tier.tier]}`}>{tier.tier} · {tier.label}</span>
            <span className="text-sm text-slate-700">{tier.deadline_text} · {tier.facility}</span>
          </div>
        </div>
      </div>

      <div className="grid gap-5 border-t border-venus-line p-5 lg:grid-cols-[1fr_320px]">
        <div>
          <div className="flex flex-wrap gap-1">
            {OVERLAYS.map(([key, label]) => (
              <button key={key} onClick={() => setView(key)} className={`rounded-md px-2.5 py-1 text-xs font-semibold ${view === key ? "bg-venus-navy text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"}`}>
                {label}
              </button>
            ))}
          </div>
          <img src={`data:image/png;base64,${s3.overlays[view]}`} alt={view} className="mt-2 w-full rounded-lg bg-black" />
          <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-slate-600">
            <Legend colour="#ff0000" text="MA" /> <Legend colour="#8c0000" text="HE" /> <Legend colour="#ffdc00" text="EX" />
            <Legend colour="#ffffff" text="SE" border /> <Legend colour="#00c8ff" text="optic disc / fovea" />
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <div className="label">Five-grade probability</div>
            <div className="mt-2 space-y-1">
              {cnn.grade_probabilities.map((p, i) => (
                <div key={i} className="flex items-center gap-2 text-xs">
                  <span className="w-24 text-slate-600">{i} · {GRADE_LABELS[i]}</span>
                  <div className="h-2 flex-1 overflow-hidden rounded bg-slate-100"><div className="h-full bg-venus-teal" style={{ width: `${Math.round(p * 100)}%` }} /></div>
                  <span className="w-8 text-right text-slate-500">{(p * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </div>
          <div>
            <div className="label">Rule grader · ICDR criteria</div>
            <ul className="mt-1 space-y-1 text-xs text-slate-700">
              {rule.trace.map((t) => <li key={t}>• {t}</li>)}
            </ul>
            <div className="mt-2 grid grid-cols-4 gap-1 text-center text-xs">
              {Object.entries(rule.counts).map(([k, v]) => (
                <div key={k} className="rounded bg-slate-50 py-1 ring-1 ring-slate-200"><div className="font-bold text-venus-navy">{v}</div><div className="text-slate-500">{k}</div></div>
              ))}
            </div>
            <div className="mt-1 text-[11px] text-slate-500">Hemorrhages per quadrant: {rule.hemorrhages_per_quadrant.join(" / ")} · P(NV) {cnn.nv_probability.toFixed(2)} (classifier, not localised)</div>
          </div>
          <div>
            <div className="label">Attention agreement</div>
            {agreement.score === null ? (
              <p className="mt-1 text-xs text-slate-600">{agreement.note}</p>
            ) : (
              <div className="mt-1">
                <div className="flex items-center gap-2">
                  <div className="h-2 flex-1 overflow-hidden rounded bg-slate-100"><div className={`h-full ${agreement.flag ? "bg-amber-500" : "bg-venus-blue"}`} style={{ width: `${Math.round(agreement.score * 100)}%` }} /></div>
                  <span className="text-sm font-bold text-venus-navy">{agreement.score.toFixed(2)}</span>
                </div>
                <p className="mt-1 text-[11px] text-slate-500">Fraction of Grad-CAM mass on detected lesions · chance {agreement.chance_level?.toFixed(2)} · {agreement.note}</p>
              </div>
            )}
          </div>
          <Stage0Panel s0={s0} compact />
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-venus-line p-5">
        <Timing timing={result.timing_ms} />
        <div className="flex gap-2">
          <a className="btn-secondary" href={reportUrl(result.session_id)} target="_blank" rel="noreferrer"><FileText size={16} /> PDF report</a>
          <button className="btn-primary" onClick={() => onBook(result.session_id)}><CalendarPlus size={16} /> Book appointment</button>
        </div>
      </div>
      <div className="border-t border-venus-line px-5 py-3 text-xs text-slate-500">{result.recommendation}</div>
    </div>
  );
}

function Header({ session }) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h2 className="card-title">Result</h2>
        <p className="text-xs text-slate-500">{session ? `Session ${session}` : "Grade · evidence · confidence · tier"}</p>
      </div>
      <CheckCircle2 size={18} className="text-venus-teal" />
    </div>
  );
}

function ProbabilityBar({ p, threshold, band }) {
  return (
    <div className="relative mt-3 h-3 w-full rounded-full bg-slate-200">
      <div className="absolute inset-y-0 rounded-full bg-amber-200" style={{ left: `${(threshold - band) * 100}%`, width: `${band * 200}%` }} />
      <div className={`absolute inset-y-0 left-0 rounded-full ${p >= threshold ? "bg-rose-500" : "bg-emerald-500"}`} style={{ width: `${Math.max(p * 100, 1)}%` }} />
      <div className="absolute -top-1 h-5 w-0.5 bg-venus-navy" style={{ left: `${threshold * 100}%` }} title={`threshold ${threshold}`} />
    </div>
  );
}

function Legend({ colour, text, border }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span className="inline-block h-3 w-3 rounded-sm" style={{ background: colour, border: border ? "1px solid #999" : "none" }} /> {text}
    </span>
  );
}

function Stage0Panel({ s0, compact }) {
  const q = s0.quality;
  const f = q?.features;
  return (
    <div className={compact ? "" : "mt-5"}>
      <div className="label">Stage 0 · gate and quality</div>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-xs">
        <span className={`pill ring-1 ${s0.modality.accepted ? "bg-emerald-50 text-emerald-800 ring-emerald-200" : "bg-rose-50 text-rose-800 ring-rose-200"}`}>
          fundus P = {s0.modality.fundus_probability?.toFixed(3) ?? "—"}
        </span>
        {q && (
          <span className={`pill ring-1 ${q.label === "good" ? "bg-emerald-50 text-emerald-800 ring-emerald-200" : q.label === "usable" ? "bg-amber-50 text-amber-800 ring-amber-200" : "bg-rose-50 text-rose-800 ring-rose-200"}`}>
            quality {q.label} · {q.score.toFixed(2)}{q.enhanced ? " · enhanced" : ""}
          </span>
        )}
      </div>
      {f && (
        <div className="mt-2 grid grid-cols-3 gap-1 text-[11px] text-slate-600">
          <div>sharpness <b>{f.sharpness.toFixed(0)}</b></div>
          <div>illumination <b>{f.illumination_uniformity.toFixed(2)}</b></div>
          <div>saturated <b>{(f.saturated_fraction * 100).toFixed(1)}%</b></div>
          <div>dark <b>{(f.dark_fraction * 100).toFixed(1)}%</b></div>
          <div>coverage <b>{f.fov_coverage.toFixed(2)}</b></div>
          <div>circularity <b>{f.fov_circularity.toFixed(2)}</b></div>
        </div>
      )}
    </div>
  );
}

function Timing({ timing }) {
  const parts = ["stage0", "stage1", "stage2", "stage3"].filter((k) => timing[k] !== undefined);
  return (
    <div className="flex items-center gap-2 text-xs text-slate-500">
      <Clock size={14} />
      <span><b className="text-venus-navy">{(timing.total / 1000).toFixed(1)} s</b> total on this machine</span>
      {parts.map((k) => <span key={k}>· {k.replace("stage", "S")} {timing[k]} ms</span>)}
    </div>
  );
}
