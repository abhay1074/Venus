# Privacy — what Venus AI stores, where it stays, and what it does not do

This describes the code as it is, verified against `backend/venus/stage5_schedule.py`,
`backend/venus/report.py` and `backend/main.py`. Where the India **Digital Personal Data
Protection Act, 2023 (DPDP)** would expect something this build does not do, it is listed as a
gap rather than dressed up. Nothing here is a compliance certification; it is a description
written so that a deploying programme can see exactly what it would be taking on.

## 1. Everything stays on the machine

There is no telemetry, no analytics, no cloud inference and no outbound network call in the
serving path. The API binds `127.0.0.1` by default (see `scripts/serve.ps1` and §6), the models
run locally on the CPU, and the SMS notification in Stage 5 is **simulated** — it is written to
a log column in the database and displayed, never sent to a gateway. The offline bundle
(`scripts\build-offline-bundle.ps1`) runs the whole system with the network cable out.

Images are **uploaded, decoded in memory, graded, and not stored as images anywhere except
inside the generated report**. The database stores no pixels.

## 2. What is in `backend/data/venus.sqlite`

A plain, unencrypted SQLite file. Five tables hold data; two are synthetic scaffolding.

### `patients` — created only by `POST /intake`

| field | kind |
|---|---|
| `name`, `phone` | **direct identifiers** |
| `age`, `sex`, `village`, `phc`, `last_eye_exam` | **quasi-identifiers** — village plus age and sex can single a person out in a small population |
| `diabetes_years`, `hba1c`, `insulin`, `hypertension`, `pregnant`, `symptoms` | **health data about an identified person** |
| `consent`, `consent_at`, `created_at` | consent record (see §4) |
| `id` (`PT-XXXXXXXX`) | pseudonymous key used by the other tables |

### `screenings` — created by every `POST /screen`

`session_id`, `patient_id` (null until intake), `captured_at`, `accepted`, `grade`,
`referable`, `p_referable`, `flag`, `tier`, `quality`, `elapsed_ms`, and `result_json`.

`result_json` is the full result struct **with the image overlays removed**
(`pipeline.screen_image` strips `stage3.overlays` before saving): grades, probabilities, lesion
counts and centroids, quality features, timings, model version and calibration fingerprint. It
contains no image data. Lesion centroids are coordinates inside a 512×512 frame and are not
re-identifying on their own.

### `appointments`

`patient_id`, `session_id`, tier, deadline, slot, facility, status, outcome, `bump_history`,
and `sms_log` — **the SMS log contains the patient's phone number** and the message text.

### `facilities`, `slots`

Fictional district scaffolding defined in code (`District Hospital Ophthalmology`, two taluk
clinics, a tele-reading centre, two PHC cameras). No personal data.

## 3. `backend/reports/` — the only place images persist

`write_report` writes two files per screening:

- `VS-XXXXXXXXXX.pdf` — **contains the fundus image**: the original frame, the lesion overlay
  and the Grad-CAM overlay, plus the grade, the evidence and the footer. This is the one
  artefact on disk from which a retina can be seen.
- `VS-XXXXXXXXXX.json` — the same slimmed result as `result_json`, no images.

The PDF carries no name or phone — it is keyed by session id — but a fundus photograph is
biometric-grade imagery and the session id links to the patient row.

## 4. Consent, as the code actually handles it

`POST /intake` refuses with HTTP 400 and *"Consent is required before a patient record is
created"* unless `consent` is true; `consent` and `consent_at` are then stored on the patient
row. The Appointments screen shows a consent checkbox before the form can be submitted.

**Stated plainly: a screening can happen before any consent is recorded.** `POST /screen`
requires no consent and writes a `screenings` row with `patient_id = NULL`. In the intended
workflow the operator takes consent on paper at the camera and the digital record is created at
intake — but the code does not enforce that order, and a deploying programme must close it
procedurally or in code.

## 5. Retention and deletion

- **Retention: indefinite.** Nothing expires. There is no retention schedule, no scheduled job
  and no maximum age; rows and PDFs stay until someone removes them.
- **Deletion: all or nothing.** `POST /reset-demo` truncates every table and deletes every
  `.pdf` and `.json` in `backend/reports/`. It is a demo-reset button.
- **There is no way to delete one patient.** No endpoint, no CLI. Erasing one person today
  means editing the SQLite file by hand and deleting their report files.

## 6. Security posture

- The API has **no authentication and no authorisation**. Anyone who can reach the port can
  call `GET /screenings` and read every stored result, or fetch any report by session id.
  The mitigation in this build is network scope, not identity: the server binds `127.0.0.1`
  and `VENUS_BIND_ALL=true` is required to expose it (§1.5 of the deployment plan).
- The SQLite file and the PDFs are **not encrypted at rest**. Disk encryption (BitLocker on the
  demo laptop) is the only protection, and that is the host's responsibility, not this code's.
- There is no audit log of who read a record, no breach detection, and no key management.

## 7. Against the DPDP Act, 2023 — what holds and what is missing

Where a deploying entity would be the Data Fiduciary and the patient the Data Principal.

| DPDP expectation | This build |
|---|---|
| Consent before processing, recorded | **Partial.** Recorded at intake with a timestamp; screening can precede it (§4). |
| Itemised notice of purpose, in plain language | **Gap.** The consent text shown in the UI is not stored with the record, so there is no evidence of *what* was agreed to. |
| Purpose limitation, data minimisation | **Largely met.** Fields collected are the ones the tier rules use (HbA1c, pregnancy, symptoms drive the priority). No images in the database. |
| Accuracy | Grades are model output with a recorded model version and calibration fingerprint; a clinician reviews every case and outcomes are recorded. |
| Storage limitation / erase when the purpose is served | **Gap.** Indefinite retention, no schedule (§5). |
| Right to access and correction | **Gap.** No endpoint; data is readable only by whoever has the file. |
| Right to erasure | **Gap.** Only a wipe-everything reset (§5). |
| Right to grievance redressal; Data Protection Officer / contact | **Gap.** Not implemented; no contact is recorded anywhere in the product. |
| Consent withdrawal, as easy as giving it | **Gap.** No withdrawal path. |
| Reasonable security safeguards | **Partial.** Local-only by default; no auth, no encryption at rest, no audit log (§6). |
| Breach notification to the Board and affected Principals | **Gap.** No detection or notification mechanism. |
| Children's data (verifiable parental consent) | **Gap.** No age gate; `age` is collected but never checked. |

**What this means.** As it stands Venus AI is a research and demonstration build that is safe
to run on a single laptop with the data of consenting adults for a demonstration. It is **not
DPDP-compliant as deployed software**, and the gaps above — per-patient erasure, access and
correction, notice retention, withdrawal, retention limits, authentication, encryption at rest,
a named grievance contact — are the work a programme would have to fund before screening real
patients at scale. They are engineering gaps, not unknowns.

## 8. If you are demonstrating this

Use the shipped `samples/` images, which are CC0 / CC BY / public-domain photographs unrelated
to any patient (`samples/README.md`), and the fictional intake. Run `POST /reset-demo`
afterwards to leave nothing behind. Do not put a real patient's name, phone or fundus image
into a demo build.
