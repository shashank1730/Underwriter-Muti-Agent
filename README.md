## Underwriter Multi‑Agent AI

An AI-powered underwriting copilot that triages queries, analyzes properties, answers policy questions with citations, and drafts empathetic customer communications — all through a single, production-ready Streamlit interface. Built on a LangGraph orchestration layer with specialized agents and verifiable retrieval.

---

### Why this matters (stakeholder pitch)

- **Accelerate underwriting decisions**: Automate high-friction steps — image analysis, risk scoring, document Q&A — so underwriters focus on judgment, not busywork.
- **Reduce loss ratios with early risk signals**: On-demand property image analysis and structured risk factors surface issues (roof, exterior, pools, condition) before bind.
- **Improve CX at scale**: Turn historical complaint data into consistent, empathetic, on-brand responses in seconds.
- **Institutionalize knowledge**: Retrieval-augmented answers cite policy sources, creating auditable, compliant outputs.
- **Deploy fast, iterate faster**: Modular agent design, low-latency LLMs, and a clean Streamlit UI enable rapid pilots and measurable value in weeks.

---

### What it does (key capabilities)

- **High-Value Property Assessment Agent**

  - Scrapes Zillow by address, analyzes listing images via Gemini, and returns a structured property profile.
  - Produces an underwriting-focused **Risk Score (0–5)** with explicit risk factors and reasoning.
  - Saves artifacts (image analysis, risk analysis, person-provided vs AI comparison) for auditability.

- **Q&A Underwriter (RAG Agent)**

  - Answers policy and terms questions using a Chroma vector store built from internal docs.
  - Enforces “answer only with context” and returns top citations (chunk previews + counts).

- **Recommendation Agent (Customer Comms)**
  - Leverages a vector DB of historical home insurance complaints to draft short, empathetic emails.
  - Aligns to brand tone; clearly states that outcomes vary case-by-case.

---

### What a user experiences

In the Streamlit app: type a question or drop a property address. The orchestrator routes to the right agent and shows progressive feedback while processing. Results render as:

- Property: ZPID, image-derived attributes, risk score, factors, and reasoning.
- Policy Q&A: a precise answer followed by clear source citations.
- Recommendations: a concise, customer-ready email draft grounded in similar past cases.

---

### Sample use cases

- **Pre-bind risk triage**: “46 Creekstone Ln, Dawsonville, GA 30534” → get roof type, exterior, pools, condition, and a risk score out-of-the-box.
- **Policy clarity**: “Does insurance cover war damage?” → compliant answer citing relevant sections in the policy library.
- **Customer response**: “My kitchen caught fire, need help” → empathetic, concise email with next steps, grounded in prior resolutions.

---

### Architecture overview

- **UI**: Streamlit chat experience with real-time progress and structured result views.
- **Orchestration**: LangGraph state machine routes to agents via an LLM classifier.
- **LLM**: Google Gemini 2.5 Flash (low latency, deterministic temperature 0 for consistent outputs).
- **Agents**:
  - `ImageAnalysisAgent`: Zillow scrape → image analysis (Gemini) → risk scoring (Gemini).
  - `RAGAnalysisAgent`: HF embeddings → Chroma → retrieval chain → grounded answers.
  - `RecommendationAgent`: Chroma over complaints → Gemini drafting.
- **Vector DB**: Chroma with `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- **State**: Typed state object passed through nodes; artifacts persisted for audit.

Data flow at a glance:

1. User input → 2) Orchestrator classifies intent → 3) Specialized agent executes → 4) Result normalized into `current_result` → 5) UI renders and persists artifacts.

---

### Repo structure

```text
app.py                              # Streamlit entry (runs the chat UI)
src/langgraph_agent/
  main.py                           # Loads Streamlit UI and orchestrates graph execution
  graph/graph_builder.py            # LangGraph state machine and routing logic
  llm/llm.py                        # Gemini model wrapper
  state/                            # Shared state definitions
  tools/
    Image_analysis_agent/           # Zillow + image analysis + risk score
    rag_agent/                      # RAG over policy docs (Chroma)
    recommendation_agent/           # Complaints DB + email drafting
  ui/streamlit_ui/                  # UI components
```

---

### Technical details (for builders)

- **Routing**: The orchestrator prompts Gemini to classify into one of: `image_analysis`, `terms_conditions`, `recommendation_agent`, or `general_response`.
- **Image analysis**: Prompts Gemini to return STRICT JSON; includes a robust JSON cleanup/extraction fallback to handle model formatting variances.
- **Risk scoring**: Produces 0–5 numeric score, normalized structure: score, factors, reasoning, overall assessment.
- **RAG**: Similarity search (k=5) → stuff chain → “answer only with context” guardrails; returns chunk previews and counts for transparency.
- **Recommendation**: Similarity search over embedded complaint corpus; short, empathetic email generation with guardrails.
- **Persistence**: Per-property JSON artifacts: `property_{zpid}_image_analysis.json`, `property_{zpid}_risk_analysis.json`, `property_{zpid}_person_report.json`, and an optional combined `property_{zpid}_complete_analysis.json`.

---

### Getting started

Prerequisites:

- Python 3.10+
- A Google Generative AI API key with access to Gemini 2.5

Install:

```bash
pip install -r requirements.txt
```

Environment:

```bash
echo GOOGLE_API_KEY=your_key_here > .env
```

Run:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit.

---

### Configuration & data

- **RAG corpus**: Persisted Chroma DB under `src/langgraph_agent/tools/rag_agent/chroma_db`.
- **Recommendation DB**: Prebuilt under `src/langgraph_agent/tools/recommendation_agent/recommendation_agent_db`.
- **Zillow scraping**: Address must follow `Street, City, State ZIP`. The scraper returns `zpid` and `images` consumed by the image agent.

---

### Reliability, risk, and compliance

- **Determinism for decisions**: Temperature set to 0; image and risk prompts enforce strict schemas.
- **Verifiability**: Q&A includes document chunk previews and counts to support audit.
- **PII & security**: No PII required; environment keys loaded from `.env` via `python-dotenv`.
- **Failure modes**: JSON parsing fallbacks; safe defaults and explicit error messages surfaced to UI.

---

### KPIs to track

- Underwriter handling time (UHT) reduction per case.
- % of queries definitively answered with citations.
- Pre-bind risk issues found vs baseline.
- Customer response TAT improvement for complaint-driven emails.

---

### Roadmap (suggested)

- Add claim document ingestion (photos, adjuster notes) to enrich risk signals.
- Expand policy library coverage and access control for role-based sources.
- Introduce cost-aware routing and batching for lower inference spend.
- Add evaluation harnesses for factuality and policy adherence.

---

### License

For internal evaluation and prototyping. Adapt licensing as needed for production.
