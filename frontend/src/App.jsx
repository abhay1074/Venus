import { useEffect, useState } from "react";
import { Activity, BarChart3, CalendarClock, Eye, FlaskConical, ListChecks, Menu, X } from "lucide-react";
import { getHealth } from "./api/venus.js";
import ScreenPage from "./screens/ScreenPage.jsx";
import ReviewPage from "./screens/ReviewPage.jsx";
import DistrictPage from "./screens/DistrictPage.jsx";
import AppointmentsPage from "./screens/AppointmentsPage.jsx";
import ValidationPage from "./screens/ValidationPage.jsx";

const NAV = [
  { id: "screen", label: "Screen", hint: "Capture · grade · explain", icon: Eye },
  { id: "review", label: "Review queue", hint: "Flagged cases, both grades", icon: ListChecks },
  { id: "district", label: "District", hint: "Programme simulation", icon: BarChart3 },
  { id: "appointments", label: "Appointments", hint: "Priority scheduling", icon: CalendarClock },
  { id: "validation", label: "Validation", hint: "Operating point · evidence", icon: FlaskConical },
];

export default function App() {
  const [page, setPage] = useState(() => {
    const wanted = new URLSearchParams(window.location.search).get("page");
    return NAV.some((n) => n.id === wanted) ? wanted : "screen";
  });
  const [health, setHealth] = useState({ online: false, status: "checking" });
  const [open, setOpen] = useState(false);
  // Result of the last screen, kept so the Appointments page can book it.
  const [lastResult, setLastResult] = useState(null);
  const [bookingSession, setBookingSession] = useState(null);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      const h = await getHealth();
      if (alive) setHealth(h);
    };
    poll();
    const timer = setInterval(poll, 15000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  const goBook = (sessionId) => {
    setBookingSession(sessionId);
    setPage("appointments");
  };

  return (
    <div className="min-h-screen text-venus-ink">
      <header className="sticky top-0 z-30 border-b border-venus-line bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <button className="rounded-md p-2 text-slate-600 hover:bg-slate-100 lg:hidden" onClick={() => setOpen(!open)} aria-label="menu">
              {open ? <X size={20} /> : <Menu size={20} />}
            </button>
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-venus-navy text-white">
              <Activity size={20} />
            </div>
            <div>
              <div className="text-base font-bold leading-tight text-venus-navy">Venus AI</div>
              <div className="text-xs text-slate-500">Explainable diabetic-retinopathy screening · SIH26038</div>
            </div>
          </div>
          <nav className="hidden items-center gap-1 lg:flex">
            {NAV.map((item) => (
              <button
                key={item.id}
                onClick={() => setPage(item.id)}
                className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-semibold transition ${
                  page === item.id ? "bg-venus-navy text-white" : "text-slate-600 hover:bg-slate-100"
                }`}
              >
                <item.icon size={16} /> {item.label}
              </button>
            ))}
          </nav>
          <HealthBadge health={health} />
        </div>
        {open && (
          <div className="border-t border-venus-line bg-white px-4 py-2 lg:hidden">
            {NAV.map((item) => (
              <button
                key={item.id}
                onClick={() => {
                  setPage(item.id);
                  setOpen(false);
                }}
                className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm ${
                  page === item.id ? "bg-venus-navy text-white" : "text-slate-700 hover:bg-slate-100"
                }`}
              >
                <item.icon size={16} />
                <span>
                  <span className="font-semibold">{item.label}</span>
                  <span className="ml-2 text-xs opacity-70">{item.hint}</span>
                </span>
              </button>
            ))}
          </div>
        )}
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        {page === "screen" && <ScreenPage health={health} lastResult={lastResult} onResult={setLastResult} onBook={goBook} />}
        {page === "review" && <ReviewPage onBook={goBook} />}
        {page === "district" && <DistrictPage health={health} />}
        {page === "appointments" && <AppointmentsPage sessionId={bookingSession} lastResult={lastResult} />}
        {page === "validation" && <ValidationPage />}
      </main>

      <footer className="mx-auto max-w-7xl px-4 py-6 text-xs text-slate-500 sm:px-6">
        Screening aid, not a diagnosis. Every image is read by an eye-care professional. Model {health.model_version || "—"}
        {health.operating_point?.fingerprint ? ` · calibration ${health.operating_point.fingerprint.slice(0, 12)}…` : ""}
      </footer>
    </div>
  );
}

function HealthBadge({ health }) {
  const ok = health.online && health.status === "ok";
  const text = !health.online ? "API offline" : health.status === "ok" ? "Pipeline ready" : "Degraded";
  return (
    <div className={`pill ring-1 ${ok ? "bg-emerald-50 text-emerald-800 ring-emerald-200" : "bg-rose-50 text-rose-800 ring-rose-200"}`}>
      <span className={`mr-2 inline-block h-2 w-2 rounded-full ${ok ? "bg-emerald-500" : "bg-rose-500"}`} />
      {text}
    </div>
  );
}
