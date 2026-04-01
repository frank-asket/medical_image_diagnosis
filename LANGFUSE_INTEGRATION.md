# Langfuse Observability Integration

Production-grade observability for the Medical Image Diagnosis pipeline.
Designed for hospital deployment with HIPAA-aligned data handling.

---

## A. Architecture

### Where Langfuse hooks into the pipeline

Langfuse is injected via a thin **`PipelineTracer`** abstraction
(`medical_diagnosis/observability/langfuse_client.py`) that the orchestrator
creates per `run()` call. When `LANGFUSE_ENABLED` is falsy or the `langfuse`
package is absent, a **`NoopTracer`** is returned - all tracing calls become
zero-cost no-ops.

```
Orchestrator.run()
  │
  ├─ get_tracer()          → LangfuseTracer | NoopTracer
  ├─ _obs.start_trace()    → opens root Langfuse span
  │
  ├─ _obs.span("domain-routing")
  │     └─ DomainRouterAgent.classify()
  │
  ├─ _obs.span("image-gate")           (manual-domain path only)
  │     └─ MedicalImageGateAgent.assess()
  │
  ├─ _obs.event("guardrail-block")     (if pipeline blocked)
  │
  ├─ _obs.span("specialist-analysis")
  │     └─ RadiologyAgent / DermatologyAgent / OphthalmologyAgent
  │
  ├─ _obs.event("guardrail-specialist-validation-failed")  (if schema invalid)
  │
  ├─ _obs.span("diagnosis-adapter")
  │     └─ HeuristicAdapter.infer()
  │
  ├─ _obs.event("narrative-suppression")  (if confidence < threshold)
  │
  ├─ _obs.span("narrative-generation")
  │     └─ DiagnosticNarrativeService.generate_narratives()
  │
  ├─ _obs.span("clinical-qa")
  │     └─ DiagnosticNarrativeService.answer_clinical_question()
  │
  └─ _obs.end_trace()      → closes root span with safe output metadata
```

### Trace model

| Concept          | Langfuse primitive | Lifetime |
|------------------|--------------------|----------|
| Full pipeline run | Root span (trace) | One per `run()` call |
| Follow-up Q&A    | Separate trace linked via `session_id` | One per `answer_question()` |
| Pipeline step    | Child span          | Opened/closed around agent call |
| Guardrail signal | Event               | Instant, logged at decision point |
| Clinician review | Score (on trace)    | Created asynchronously via `/api/feedback` |

### Span structure (one `run()` invocation)

```
trace: "medical-diagnosis-pipeline"
├── span: "domain-routing"            (or "image-gate" for manual domain)
├── event: "guardrail-block"          (conditional)
├── span: "specialist-analysis"
├── event: "guardrail-specialist-validation-failed"  (conditional)
├── span: "diagnosis-adapter"
├── event: "narrative-suppression"    (conditional)
├── span: "narrative-generation"
└── span: "clinical-qa"              (conditional)
```

### Metadata logged per span

| Span | Fields |
|------|--------|
| **Root trace** | `image_fingerprint`, `mode`, `patient_context_hash` |
| **Root output** | `pipeline_status`, `blocked_reason`, `domain`, `diagnosis_label`, `confidence`, `triage_level`, `specialist_schema_valid`, `narratives_suppressed` |
| **domain-routing** | `domain` selected, `has_validation_errors`, `model` name |
| **image-gate** | `has_errors`, `model` name |
| **specialist-analysis** | `confidence`, `model` name, `domain` |
| **diagnosis-adapter** | `diagnosis_label`, `confidence`, `triage_level` |
| **narrative-generation** | `generated: true` |
| **clinical-qa** | `answered: true` |
| **Events** | `reason` string for the specific guardrail signal |

### Conversation tracking

1. `run()` generates a Langfuse trace with a unique `trace_id`.
2. The `trace_id` is stored in the result as `_trace_id` (underscore-prefixed internal metadata, consistent with `_agent_meta` used throughout the codebase).
3. In the webapp, `_trace_id` is popped from the HTTP response and stored server-side in `_bundle_traces[bundle_id]`.
4. When `POST /api/qa` is called, the stored `_trace_id` is injected into the prior bundle so `answer_question()` can create a new trace linked via Langfuse's `session_id`.
5. This gives a session view in Langfuse: diagnose trace → QA trace 1 → QA trace 2 → …, all grouped under the same `session_id`.

### Feedback flow

```
Clinician UI → POST /api/feedback (JSON body)
  │
  ├── Lookup trace_id from _bundle_traces[bundle_id]
  │
  └── submit_clinician_feedback(trace_id, agreement, ...)
        │
        ├── langfuse.create_score("clinician_agreement", NUMERIC)
        ├── langfuse.create_score("clinician_agreement_category", CATEGORICAL)
        ├── langfuse.create_score("clinician_corrected_diagnosis", CATEGORICAL)
        ├── langfuse.create_score("clinician_corrected_triage", CATEGORICAL)
        ├── langfuse.create_score("clinician_confidence_override", NUMERIC)
        └── langfuse.create_score("clinician_comment", CATEGORICAL)
```

All scores are attached to the original diagnosis trace. Langfuse's dashboard
shows aggregate agreement rates, correction frequency, and confidence
calibration across traces.

### Data redaction strategy

| Data type | Treatment |
|-----------|-----------|
| Raw image bytes | **Never logged.** Only a SHA-256 fingerprint (first 16 hex chars) appears in traces. |
| `patient_context` | **Never logged in clear text.** A one-way SHA-256 hash prefix (12 chars) is stored for correlation. |
| Patient identifiers (name, MRN, DOB, SSN, etc.) | **Filtered** by `_PHI_KEYS` set in `langfuse_client.py`. Stripped before any data reaches Langfuse. |
| Specialist JSON output | Structured diagnosis fields (`diagnosis_label`, `confidence`, `triage_level`) are logged. Full findings text is NOT logged into trace output. |
| Narrative text | **Not logged** to Langfuse. Only a `generated: true` flag is recorded. |
| Clinical Q&A question/answer | **Not logged** to Langfuse. Only `answered: true` and `question_length` metadata are recorded. |
| Clinician feedback comments | Truncated to 2000 chars. Stored as Langfuse score `comment` field, not as trace content. |

---

## B. Configuration

Add to `.env` (all optional - system works identically without them):

```bash
# Langfuse observability (optional)
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `LANGFUSE_ENABLED` | `false` | Master switch. Must be `true` / `1` / `yes` to activate. |
| `LANGFUSE_PUBLIC_KEY` | (empty) | Langfuse project public key. |
| `LANGFUSE_SECRET_KEY` | (empty) | Langfuse project secret key. |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | Langfuse instance URL. Use your self-hosted URL or `https://us.cloud.langfuse.com` for US region. |

---

## C. HIPAA Compliance Checklist

> This checklist is for **engineering teams** building toward HIPAA compliance.
> It does NOT constitute legal advice. Engage a compliance officer and legal
> counsel for a formal gap analysis before handling real PHI.

### Administrative safeguards

- [ ] **Privacy Officer designated** for the Langfuse integration scope.
- [ ] **Workforce training** completed for all personnel with Langfuse dashboard access.
- [ ] **Access management policy** documented: who can view traces, who can export, who can delete.
- [ ] **Incident response plan** covers Langfuse data exposure scenarios (leaked API keys, unauthorized dashboard access).
- [ ] **Risk assessment** completed for the Langfuse data flow (self-hosted vs. cloud, data residency).

### Technical safeguards

- [ ] **Encryption in transit**: All Langfuse communication over TLS 1.2+ (enforced by SDK default).
- [ ] **Encryption at rest**: Langfuse instance storage encrypted (self-hosted: configure at infra level; cloud: verify Langfuse's SOC 2 / encryption posture).
- [ ] **No PHI in traces**: Code audit confirms `safe_diagnosis_output()` and `hash_patient_context()` are the only paths to Langfuse; raw patient data never reaches the SDK.
- [ ] **No image bytes logged**: `content_fingerprint()` is the only image representation in any trace.
- [ ] **Audit logging**: Langfuse provides built-in access logs. Ensure they are retained per policy.
- [ ] **Session timeout / MFA** on Langfuse dashboard access.
- [ ] **API key rotation** schedule established (Langfuse public/secret keys).

### Access control

- [ ] **Role-based access** in Langfuse (viewer / editor / admin) configured per team member.
- [ ] **Principle of least privilege**: clinician feedback reviewers get read-only access to scores; they cannot view raw trace metadata.
- [ ] **Separate Langfuse projects** for dev / staging / production environments.
- [ ] **Network segmentation**: production Langfuse instance not accessible from dev networks.

### Data retention

- [ ] **Retention policy** defined (e.g., 90 days for traces, 1 year for scores).
- [ ] **Automated deletion** configured in Langfuse for expired traces.
- [ ] **Data export controls**: export of trace data requires approval.
- [ ] **Right to delete**: process exists to purge all traces associated with a specific session/patient.

### Business Associate Agreement (BAA)

- [ ] **BAA executed with Langfuse** (required if ANY PHI could transit or reside in Langfuse - even metadata). Langfuse Cloud offers BAA on enterprise plans. For self-hosted, the BAA obligation shifts to your infrastructure provider.
- [ ] **BAA covers subprocessors**: if Langfuse Cloud stores data on AWS/GCP, those sub-BAAs must be in place.
- [ ] **Annual BAA review** scheduled.

### De-identification rules (Safe Harbor method, 45 CFR §164.514)

- [ ] All 18 HIPAA identifiers removed or transformed before data reaches Langfuse.
- [ ] `patient_context` is hashed (SHA-256, truncated) - not reversible, not linkable to an individual.
- [ ] No dates (birth, admission, discharge) in trace metadata.
- [ ] No geographic data below state level.
- [ ] No device identifiers, URLs, or biometric identifiers in traces.
- [ ] Re-identification risk assessed and documented.

### PHI minimization policy

- [ ] **Only structured diagnostic metadata** (diagnosis_label, confidence, triage_level, pipeline_status) enters Langfuse.
- [ ] **Narrative text, Q&A content, and specialist findings** are NOT logged - only boolean flags (`generated: true`, `answered: true`).
- [ ] **Clinician feedback comments** are capped at 2000 characters and must not contain patient identifiers (enforced by UI guidance and policy, not code).
- [ ] **Image fingerprints** are SHA-256 truncated (16 hex chars) - non-reversible.

### Incident response

- [ ] **Breach notification timeline**: 60 days to HHS, without unreasonable delay to individuals.
- [ ] **Langfuse API key compromise playbook**: revoke keys, rotate, audit trace access logs, assess whether PHI was exposed.
- [ ] **Trace data leak assessment**: because traces contain only de-identified metadata, a leak is likely NOT a HIPAA breach - but document the analysis for each incident.

---

## D. Tradeoffs and Risks

### Logging model prompts

| Consideration | Position |
|---------------|----------|
| **Value** | Prompt logging enables debugging hallucinations, comparing prompt versions, and regression testing. |
| **Risk** | System prompts contain the clinical decision-support framing. If leaked, an attacker understands the model's reasoning constraints. User prompts may inadvertently contain PHI if `patient_context` is not properly scrubbed. |
| **This integration** | **Prompts are NOT logged.** The Langfuse spans receive only structured metadata (`mode`, `domain`, `confidence`, etc.), never the actual system/user prompt text. |
| **When to enable** | Only in a locked-down self-hosted Langfuse instance with no external access, BAA in place, and a specific debugging need. Use `LANGFUSE_OBSERVE_DECORATOR_IO_CAPTURE_ENABLED=false` (SDK default respects this). |

### Storing model outputs

| Consideration | Position |
|---------------|----------|
| **Value** | Output storage enables quality auditing, clinician agreement analytics, and post-market surveillance (21 CFR Part 820 / IEC 62304). |
| **Risk** | Specialist outputs contain clinical findings text that, combined with image fingerprints and timestamps, could theoretically be linked to a patient. |
| **This integration** | **Only structured scalars** from `provisional_diagnosis` are stored (`diagnosis_label`, `confidence`, `triage_level`). Full findings text, narrative prose, and Q&A answers are NOT stored in Langfuse. |
| **When to expand** | When a formal de-identification risk assessment (Expert Determination method) shows acceptably low re-identification risk for the target deployment. |

### When to disable prompt capture entirely

- During **real patient use** in production (always).
- When the Langfuse instance is **cloud-hosted without BAA**.
- When **audit logging** on the Langfuse instance is not confirmed.
- During any **penetration testing** or **red-team** exercise on the infrastructure.

### Safely enabling clinician performance analytics

Langfuse scores (clinician agreement, corrections, confidence overrides) can
power analytics dashboards. To do this safely:

1. **Aggregate, never individualize.** Show agreement rates per domain, per
   triage level, per time window - never per individual clinician unless the
   deployment explicitly requires peer review and the clinician has consented.
2. **No clinician identifiers in Langfuse.** The current integration does not
   log `user_id`. If added later, use an opaque ID and maintain the
   ID ↔ clinician mapping in a separate, access-controlled system.
3. **Consent framework.** Before any per-clinician analytics, obtain informed
   consent and document the purpose, retention, and access policy.
4. **Bias monitoring.** Track whether AI agreement rates differ across
   demographic proxies (domain, facility, shift) and flag outliers for equity
   review.

### FDA Software Classification Risks

If this system is marketed as a **Software as a Medical Device (SaMD)**:

- Langfuse traces constitute **quality system records** under 21 CFR Part 820.
  Retention policies must align with device lifecycle requirements.
- Trace data may be requested during **FDA audits** or **510(k) / De Novo**
  review as evidence of post-market surveillance.
- Changes to what is logged (adding prompt capture, expanding output storage)
  may trigger a **software change assessment** under IEC 62304.
- The observability layer itself should be covered by **software verification
  and validation (V&V)** documentation.

---

## E. Files Changed / Created

| File | Change |
|------|--------|
| `medical_diagnosis/observability/__init__.py` | **New** - package exports |
| `medical_diagnosis/observability/langfuse_client.py` | **New** - tracer, PHI filter, feedback scoring |
| `medical_diagnosis/config.py` | **Modified** - added 4 `LANGFUSE_*` env vars |
| `medical_diagnosis/orchestrator.py` | **Modified** - tracer instrumentation in `run()` and `answer_question()` |
| `medical_diagnosis/webapp.py` | **Modified** - feedback endpoint, trace_id tracking, `_bundle_traces` store |
| `requirements.txt` | **Modified** - added `langfuse>=2.0.0`, `pydantic>=2.0.0` |
| `LANGFUSE_INTEGRATION.md` | **New** - this document |
