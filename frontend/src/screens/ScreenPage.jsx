import { useEffect, useMemo, useRef, useState } from "react";
import Webcam from "react-webcam";
import { AlertTriangle, Camera, Image as ImageIcon, Play, RotateCcw, UploadCloud, X } from "lucide-react";
import { fetchSampleFile, getSamples, screenImage } from "../api/venus.js";
import ResultView from "../components/ResultView.jsx";

const SYMPTOMS = [
  ["blurred_vision", "Blurred vision"],
  ["floaters", "Floaters"],
  ["sudden_vision_loss", "Sudden vision loss"],
];

export default function ScreenPage({ health, lastResult, onResult, onBook }) {
  const inputRef = useRef(null);
  const webcamRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [drag, setDrag] = useState(false);
  const [samples, setSamples] = useState([]);
  const [intake, setIntake] = useState({ diabetes_years: "", hba1c: "", pregnant: false, hypertension: false, symptoms: [] });
  const [tta, setTta] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(lastResult);

  useEffect(() => {
    getSamples().then(setSamples);
  }, []);

  // Deep link: /?sample=<file>&run=1 loads a shipped demo image and screens it
  // once the API is up. Lets a presenter (or a screenshot script) jump straight
  // to a result.
  const autoRan = useRef(false);
  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const name = q.get("sample");
    if (!name || autoRan.current || !health.online) return;
    autoRan.current = true;
    (async () => {
      const f = await fetchSampleFile({ name, url: `/samples/${name}` });
      setFile(f);
      setResult(null);
      if (q.get("run") === "1") {
        setLoading(true);
        try {
          const data = await screenImage(f, null, false);
          setResult(data);
          onResult(data);
        } catch (e) {
          setError(e.message);
        } finally {
          setLoading(false);
        }
      }
    })();
  }, [health.online]);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const accept = (f) => {
    if (!["image/jpeg", "image/png", "image/webp"].includes(f.type)) {
      setError("Use a JPEG, PNG or WEBP fundus photograph.");
      return;
    }
    setError("");
    setFile(f);
    setResult(null);
  };

  const capture = () => {
    const shot = webcamRef.current?.getScreenshot();
    if (!shot) return;
    const [head, body] = shot.split(",");
    const mime = head.match(/:(.*?);/)?.[1] || "image/jpeg";
    const bytes = Uint8Array.from(atob(body), (c) => c.charCodeAt(0));
    accept(new File([bytes], "capture.jpg", { type: mime }));
    setCameraOpen(false);
  };

  const run = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const payload = {
        ...intake,
        diabetes_years: intake.diabetes_years === "" ? null : Number(intake.diabetes_years),
        hba1c: intake.hba1c === "" ? null : Number(intake.hba1c),
      };
      const data = await screenImage(file, payload, tta);
      setResult(data);
      onResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const toggleSymptom = (id) =>
    setIntake((s) => ({ ...s, symptoms: s.symptoms.includes(id) ? s.symptoms.filter((x) => x !== id) : [...s.symptoms, id] }));

  const fileLabel = useMemo(() => (file ? `${file.name} · ${(file.size / 1024).toFixed(0)} KB` : "No image selected"), [file]);

  return (
    <div className="grid gap-6 lg:grid-cols-[400px_1fr]">
      <section className="space-y-4">
        <div className="card">
          <div className="card-head flex items-center justify-between">
            <div>
              <h2 className="card-title">Capture / upload</h2>
              <p className="text-xs text-slate-500">Portable fundus camera or phone upload → Stage 0 gate</p>
            </div>
            {file && (
              <button className="rounded-md p-2 text-slate-500 hover:bg-slate-100" onClick={() => setFile(null)} aria-label="clear">
                <X size={18} />
              </button>
            )}
          </div>
          <div className="p-4">
            <div
              className={`flex min-h-[220px] flex-col items-center justify-center rounded-lg border border-dashed p-3 text-center transition ${
                drag ? "border-venus-blue bg-blue-50" : "border-slate-300 bg-slate-50"
              }`}
              onDragOver={(e) => {
                e.preventDefault();
                setDrag(true);
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDrag(false);
                const f = e.dataTransfer.files?.[0];
                if (f) accept(f);
              }}
            >
              {cameraOpen ? (
                <div className="w-full">
                  <div className="overflow-hidden rounded-lg bg-slate-950">
                    <Webcam ref={webcamRef} audio={false} screenshotFormat="image/jpeg" videoConstraints={{ facingMode: "environment" }} className="w-full" />
                  </div>
                  <div className="mt-3 flex justify-center gap-2">
                    <button className="btn-primary" onClick={capture}>Capture</button>
                    <button className="btn-secondary" onClick={() => setCameraOpen(false)}>Cancel</button>
                  </div>
                </div>
              ) : preview ? (
                <img src={preview} alt="fundus" className="max-h-[300px] w-full rounded-lg object-contain" />
              ) : (
                <div className="flex flex-col items-center text-slate-500">
                  <UploadCloud size={30} className="mb-2 text-venus-teal" />
                  <p className="font-semibold text-venus-navy">Drop a fundus photograph</p>
                  <p className="mt-1 text-xs">JPEG, PNG or WEBP · any camera</p>
                </div>
              )}
            </div>
            <p className="mt-2 truncate text-xs text-slate-500">{fileLabel}</p>
            <div className="mt-3 flex flex-wrap gap-2">
              <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" className="hidden" onChange={(e) => e.target.files?.[0] && accept(e.target.files[0])} />
              <button className="btn-secondary" onClick={() => inputRef.current?.click()}><ImageIcon size={16} /> Select image</button>
              <button className="btn-secondary" onClick={() => setCameraOpen(true)}><Camera size={16} /> Camera</button>
            </div>
            {samples.length > 0 && (
              <div className="mt-4">
                <div className="label">Demo images</div>
                <div className="mt-2 grid grid-cols-3 gap-2">
                  {samples.map((s) => (
                    <button
                      key={s.name}
                      className="group overflow-hidden rounded-lg border border-slate-200 bg-white text-left hover:border-venus-teal"
                      onClick={async () => accept(await fetchSampleFile(s))}
                      title={s.note || s.name}
                    >
                      <img src={`${import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"}${s.url}`} alt={s.name} className="aspect-square w-full object-cover" />
                      <div className="truncate px-2 py-1 text-[11px] text-slate-600">{s.label || s.name}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="card">
          <div className="card-head">
            <h2 className="card-title">Patient context (optional)</h2>
            <p className="text-xs text-slate-500">Risk factors can raise the priority tier, never lower it</p>
          </div>
          <div className="grid grid-cols-2 gap-3 p-4">
            <label className="text-sm">
              <span className="label">Diabetes, years</span>
              <input className="input mt-1" type="number" min="0" value={intake.diabetes_years} onChange={(e) => setIntake({ ...intake, diabetes_years: e.target.value })} />
            </label>
            <label className="text-sm">
              <span className="label">Last HbA1c, %</span>
              <input className="input mt-1" type="number" step="0.1" min="3" max="20" value={intake.hba1c} onChange={(e) => setIntake({ ...intake, hba1c: e.target.value })} />
            </label>
            <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={intake.hypertension} onChange={(e) => setIntake({ ...intake, hypertension: e.target.checked })} /> Hypertension</label>
            <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={intake.pregnant} onChange={(e) => setIntake({ ...intake, pregnant: e.target.checked })} /> Pregnant</label>
            <div className="col-span-2">
              <div className="label">Symptoms</div>
              <div className="mt-1 flex flex-wrap gap-2">
                {SYMPTOMS.map(([id, label]) => (
                  <button
                    key={id}
                    onClick={() => toggleSymptom(id)}
                    className={`pill ring-1 ${intake.symptoms.includes(id) ? "bg-amber-100 text-amber-900 ring-amber-300" : "bg-white text-slate-600 ring-slate-300"}`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
            <label className="col-span-2 flex items-center gap-2 text-sm text-slate-600">
              <input type="checkbox" checked={tta} onChange={(e) => setTta(e.target.checked)} /> Test-time augmentation (5× inference; off for the CPU budget)
            </label>
          </div>
        </div>

        <div className="flex gap-2">
          <button className="btn-primary flex-1 justify-center" disabled={!file || loading || !health.online} onClick={run}>
            {loading ? <RotateCcw size={16} className="animate-spin" /> : <Play size={16} />}
            {loading ? "Running Stage 0 → 3…" : "Screen this image"}
          </button>
        </div>
        {!health.online && (
          <p className="flex items-start gap-2 rounded-lg bg-rose-50 p-3 text-sm text-rose-800 ring-1 ring-rose-200">
            <AlertTriangle size={16} className="mt-0.5 shrink-0" /> The API is offline. Start it with <code className="mx-1">scripts\serve.ps1</code>.
          </p>
        )}
        {error && (
          <p className="flex items-start gap-2 rounded-lg bg-rose-50 p-3 text-sm text-rose-800 ring-1 ring-rose-200">
            <AlertTriangle size={16} className="mt-0.5 shrink-0" /> {error}
          </p>
        )}
      </section>

      <section>
        <ResultView result={result} loading={loading} onBook={onBook} />
      </section>
    </div>
  );
}
