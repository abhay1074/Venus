import { useEffect, useState } from "react";
import { AlertTriangle, FileText, RefreshCw } from "lucide-react";
import { getScreenings, reportUrl } from "../api/venus.js";

const GRADE = ["No DR", "Mild", "Moderate", "Severe", "PDR"];

export default function ReviewPage({ onBook }) {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("all");

  const load = async () => {
    setLoading(true);
    try {
      setRows(await getScreenings());
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    load();
  }, []);

  const visible = rows
    .filter((r) => (filter === "flagged" ? r.flag === 1 : filter === "referable" ? r.referable === 1 : filter === "rejected" ? r.accepted === 0 : true))
    .sort((a, b) => (b.flag || 0) - (a.flag || 0) || (b.p_referable || 0) - (a.p_referable || 0));

  const flagged = rows.filter((r) => r.flag === 1).length;
  const gradable = rows.filter((r) => r.accepted === 1).length;

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-4">
        <Stat label="Screenings stored" value={rows.length} />
        <Stat label="Gradable" value={gradable} sub={`${rows.length - gradable} retakes`} />
        <Stat label="Flagged for review" value={flagged} sub={gradable ? `${((flagged / gradable) * 100).toFixed(0)}% of gradable — the workload the simulation absorbs` : "—"} />
        <Stat label="Referable" value={rows.filter((r) => r.referable === 1).length} />
      </div>

      <div className="card">
        <div className="card-head flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="card-title">Human review queue</h2>
            <p className="text-xs text-slate-500">Flagged cases first, then by P(referable). Both grades shown so the reviewer sees the disagreement, not just the verdict.</p>
          </div>
          <div className="flex items-center gap-2">
            {["all", "flagged", "referable", "rejected"].map((f) => (
              <button key={f} onClick={() => setFilter(f)} className={`rounded-md px-2.5 py-1 text-xs font-semibold ${filter === f ? "bg-venus-navy text-white" : "bg-slate-100 text-slate-600"}`}>{f}</button>
            ))}
            <button className="btn-secondary" onClick={load}><RefreshCw size={14} className={loading ? "animate-spin" : ""} /></button>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase text-slate-500">
              <tr>
                <th className="px-4 py-2">Session</th><th className="px-4 py-2">Captured</th><th className="px-4 py-2">Quality</th>
                <th className="px-4 py-2">CNN grade</th><th className="px-4 py-2">Rule grade</th><th className="px-4 py-2">P(referable)</th>
                <th className="px-4 py-2">Attention</th><th className="px-4 py-2">Tier</th><th className="px-4 py-2">Why flagged</th><th className="px-4 py-2"></th>
              </tr>
            </thead>
            <tbody>
              {visible.length === 0 && (
                <tr><td colSpan={10} className="px-4 py-8 text-center text-slate-500">{loading ? "Loading…" : "Nothing here yet — screen an image first."}</td></tr>
              )}
              {visible.map((r) => (
                <tr key={r.session_id} className="border-t border-venus-line">
                  <td className="px-4 py-2 font-mono text-xs">{r.session_id}</td>
                  <td className="px-4 py-2 text-xs text-slate-500">{new Date(r.captured_at).toLocaleString()}</td>
                  <td className="px-4 py-2 text-xs">{r.quality}</td>
                  <td className="px-4 py-2">{r.cnn_grade !== null && r.cnn_grade !== undefined ? `${r.cnn_grade} · ${GRADE[r.cnn_grade]}` : "—"}</td>
                  <td className="px-4 py-2">{r.rule_grade !== null && r.rule_grade !== undefined ? `${r.rule_grade} · ${GRADE[r.rule_grade]}` : "—"}</td>
                  <td className="px-4 py-2 font-semibold">{r.p_referable?.toFixed(2) ?? "—"}</td>
                  <td className="px-4 py-2">{r.attention_agreement?.toFixed(2) ?? "—"}</td>
                  <td className="px-4 py-2"><span className="pill bg-slate-100 text-slate-700 ring-1 ring-slate-300">{r.tier}</span></td>
                  <td className="px-4 py-2 text-xs text-amber-800">
                    {r.flag_reasons?.length ? r.flag_reasons.map((x) => <div key={x} className="flex gap-1"><AlertTriangle size={12} className="mt-0.5 shrink-0" />{x}</div>) : ""}
                  </td>
                  <td className="px-4 py-2 whitespace-nowrap">
                    {r.accepted === 1 && <a className="mr-2 inline-flex items-center gap-1 text-venus-blue hover:underline" href={reportUrl(r.session_id)} target="_blank" rel="noreferrer"><FileText size={14} /> PDF</a>}
                    {r.accepted === 1 && <button className="text-venus-teal hover:underline" onClick={() => onBook(r.session_id)}>book</button>}
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

function Stat({ label, value, sub }) {
  return (
    <div className="card p-4">
      <div className="label">{label}</div>
      <div className="mt-1 text-2xl font-extrabold text-venus-navy">{value}</div>
      {sub && <div className="text-xs text-slate-500">{sub}</div>}
    </div>
  );
}
