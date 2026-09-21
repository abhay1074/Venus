import axios from "axios";

// No client-side fallback that invents a result: if the API is down, the
// error is the honest answer.
const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://127.0.0.1:8000",
  timeout: 180000,
});

export const apiBase = client.defaults.baseURL;

function detail(error, fallback) {
  return new Error(error?.response?.data?.detail || error?.message || fallback);
}

export async function getHealth() {
  try {
    const { data } = await client.get("/health");
    return { online: true, ...data };
  } catch {
    return { online: false, status: "unavailable" };
  }
}

export async function getOperatingPoint() {
  const { data } = await client.get("/operating-point");
  return data;
}

export async function getValidationExtras() {
  const { data } = await client.get("/validation-extras");
  return data;
}

export async function screenImage(file, intake, tta = false) {
  const form = new FormData();
  form.append("file", file);
  if (intake) form.append("intake", JSON.stringify(intake));
  form.append("tta", tta ? "true" : "false");
  try {
    const { data } = await client.post("/screen", form, { headers: { "Content-Type": "multipart/form-data" } });
    return data;
  } catch (error) {
    throw detail(error, "Screening service is unavailable.");
  }
}

export async function getSamples() {
  try {
    const { data } = await client.get("/samples");
    return data;
  } catch {
    return [];
  }
}

export async function fetchSampleFile(sample) {
  const response = await client.get(sample.url, { responseType: "blob" });
  return new File([response.data], sample.name, { type: response.data.type || "image/jpeg" });
}

export async function getScreenings() {
  const { data } = await client.get("/screenings");
  return data;
}

export async function getScreening(sessionId) {
  const { data } = await client.get(`/screenings/${sessionId}`);
  return data;
}

export function reportUrl(sessionId) {
  return `${apiBase}/report/${sessionId}.pdf`;
}

export async function simulate(params) {
  try {
    const { data } = await client.post("/simulate", { params });
    return data;
  } catch (error) {
    throw detail(error, "Simulation failed.");
  }
}

export async function getSweep() {
  const { data } = await client.get("/sweep");
  return data;
}

export async function startSweep(body = {}) {
  const { data } = await client.post("/sweep", body);
  return data;
}

export async function createIntake(intake) {
  try {
    const { data } = await client.post("/intake", intake);
    return data;
  } catch (error) {
    throw detail(error, "Could not save the intake.");
  }
}

export async function createAppointment(body) {
  try {
    const { data } = await client.post("/appointments", body);
    return data;
  } catch (error) {
    throw detail(error, "Could not book an appointment.");
  }
}

export async function getWorklist() {
  const { data } = await client.get("/worklist");
  return data;
}

export async function recordOutcome(appointmentId, outcome) {
  const { data } = await client.post(`/appointments/${appointmentId}/outcome`, { outcome });
  return data;
}

export async function getFacilities() {
  const { data } = await client.get("/facilities");
  return data;
}

export async function resetDemo() {
  const { data } = await client.post("/reset-demo");
  return data;
}
