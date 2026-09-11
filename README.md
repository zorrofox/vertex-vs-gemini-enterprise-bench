# Vertex AI vs. Gemini Enterprise — Grounded Generation Benchmark

An OpenAI-compatible proxy and benchmark suite that compares **two ways of calling Gemini with Google Search grounding** on Google Cloud:

| Path | Endpoint | Billing |
|------|----------|---------|
| **Vertex AI** | `aiplatform.googleapis.com/.../models/{model}:generateContent` + `tools:[google_search]` | Pay-as-you-go (tokens + $35/1k grounded prompts) |
| **Gemini Enterprise (Agentspace)** | `discoveryengine.googleapis.com/.../assistants/{id}:streamAssist` + `webGroundingSpec` | Per-seat subscription (pooled daily quota) |

The suite measures **latency** (TTFT / E2E), **answer quality** (LLM-as-a-Judge: accuracy / recency / citations), and **cost-per-query**, then renders a side-by-side dashboard.

📊 **[Read the full comparison report →](docs/vertex-vs-ge-comparison.en.md)** ([中文版](docs/vertex-vs-ge-comparison.md))

![Latency comparison](docs/images/03-latency.png)

---

## Architecture

```
┌──────────────┐  OpenAI API   ┌───────────┐   ┌─────────────────────────────┐
│ dashboard.py │──────────────▶│           │──▶│ Vertex AI generateContent   │
│ benchmark.py │   (stream)    │  app.py   │   │   + google_search tool      │
│ any client   │               │  (proxy)  │   ├─────────────────────────────┤
└──────────────┘               │           │──▶│ Gemini Enterprise           │
                               └───────────┘   │   :streamAssist             │
                                               └─────────────────────────────┘
```

- **`app.py`** — FastAPI server exposing `/v1/chat/completions`, `/v1/images/generations` and `/v1/videos` (OpenAI wire format). Routes requests to either backend based on the `model` field.
- **`benchmark.py`** — Runs 8 time-sensitive questions against all 4 channels, then grades answers via a 2-stage LLM-as-a-Judge (live ground-truth fetch + structured scoring).
- **`dashboard.py`** — Streamlit UI: live 4-way streaming arena + Plotly analytics over benchmark results.
- **`discovery_api/`** — Core library (config, auth with token caching, SSE parsing, citation extraction, handlers).

### Model routing

| `model` | Backend | Google Search |
|---------|---------|---------------|
| `gemini-standard` | Vertex AI | ❌ |
| `vertex-grounded-search` | Vertex AI | ✅ |
| `discovery-standard` | Gemini Enterprise `:streamAssist` | ❌ |
| `discovery-grounded-search` | Gemini Enterprise `:streamAssist` | ✅ |

### Media generation (Gemini Enterprise only)

| Endpoint | `model` | Backend | Notes |
|----------|---------|---------|-------|
| `POST /v1/images/generations` | `discovery-image` | GE `:streamAssist` + `imageGenerationSpec` | Synchronous, ~40 s. `response_format`: `b64_json` (default) or `url` |
| `POST /v1/videos` → `GET /v1/videos/{id}` → `GET /v1/videos/{id}/content` | `discovery-video` | GE `:streamAssist` + `videoGenerationSpec` | Async job, ~90–120 s, returns 8 s 720p MP4 |

Both follow the OpenAI Images / Videos API shapes, so existing OpenAI clients work unchanged. Gemini Enterprise exposes **no** generation parameters (`size`, `seconds`, aspect ratio are accepted but ignored) — the model (Nano Banana / Veo) is chosen by the GE app. Generated files live in a GE session; the proxy fetches them through the (undocumented) `{session}:downloadFile` endpoint, which is why `url` mode returns a proxy URL (`/v1/media/{session_id}/{file_id}`) rather than a Google URL. Video jobs are kept in process memory — restart the proxy and they are gone.

---

## Prerequisites

- A GCP project with **Vertex AI API** and **Discovery Engine API** enabled
- A **Gemini Enterprise / Agentspace app** with an assistant that has `webGroundingType: GOOGLE_SEARCH` enabled
- `gcloud` CLI authenticated (`gcloud auth login` + `gcloud auth application-default login`)
- Python 3.11+

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — set GCP_PROJECT_ID and AGENT_SEARCH_ENGINE
# Tip: discover available engines with
python scripts/list_engines.py [PROJECT_ID]

# 3. Start the proxy
python app.py
# → http://localhost:8000/health

# 4. Run the full benchmark (8 questions × 4 channels, ~10 min)
python benchmark.py

# 5. Explore results
streamlit run dashboard.py

# 6. (Optional) Regenerate static charts for the report
python scripts/render_charts.py
```

### Calling the proxy directly

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "discovery-grounded-search",
    "messages": [{"role": "user", "content": "Who is the current UK Prime Minister?"}],
    "stream": true
  }'
```

```bash
# Image → PNG file
curl -s http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "discovery-image", "prompt": "A watercolor lighthouse at dawn"}' \
  | python3 -c "import sys,json,base64; open('out.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"

# Video → async job
curl -s -X POST http://localhost:8000/v1/videos \
  -H "Content-Type: application/json" \
  -d '{"model": "discovery-video", "prompt": "Aerial shot over a pine forest in morning fog"}'
# → {"id": "video_…", "status": "queued", ...}
curl -s http://localhost:8000/v1/videos/video_…            # poll until "status": "completed"
curl -s http://localhost:8000/v1/videos/video_…/content -o out.mp4
```

---

## Configuration (`.env`)

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | GCP project ID (auto-detected from ADC if omitted) |
| `GCP_PROJECT_NUMBER` | GCP project number (optional, auto-detected) |
| `VERTEX_LOCATION` | Vertex AI region, default `us-central1` |
| `DISCOVERY_LOCATION` | Discovery Engine location, default `global` |
| `GEMINI_MODEL` | Vertex model ID, default `gemini-2.5-flash` |
| `AGENT_SEARCH_ENGINE` | **Required for GE path** — Agentspace engine ID |
| `AGENT_SEARCH_ASSISTANT` | Assistant ID, default `default_assistant` |
| `AGENT_SEARCH_MODEL_VERSION` | GE `generationSpec.modelId` (leave empty to use app default) |
| `GCP_AUTH_SOURCE` | `gcloud` (CLI active account, default) or `adc` |
| `GCP_AUTH_ACCOUNT` | Optional gcloud account to use instead of the active one (multi-account machines) |
| `PROXY_PORT` | Proxy listen port, default `8000` |
| `CORS_ORIGINS` | Comma-separated allowed origins |

> ⚠️ **Never commit `.env`** — it is git-ignored. Only `.env.example` (with placeholders) is tracked.

---

## Key Findings (TL;DR)

| Dimension | Vertex + Search | GE Assist + Search |
|-----------|-----------------|---------------------|
| Answer quality (Judge, /5) | 4.46 | **4.54** (statistically tied) |
| TTFT | **6.30s** | 8.73s (+38%) |
| End-to-end latency | **7.53s** | 14.00s (+86%) |
| In-quota cost/query | $0.040 | **$0.006** (Standard seat, 6× cheaper) |
| Overage cost/query | — | $0.10 (2.5× **more expensive** than Vertex) |
| Multimodal output | Text only (separate APIs for Imagen/Veo/Code) | **Text + image + code-exec + file in one endpoint** |

**Bottom line:** Same Google Search, same answer quality. Vertex is **fast & elastic** — pick it for products and APIs. Gemini Enterprise is **slower but 6× cheaper within quota and natively multimodal** — pick it for internal employee tooling with predictable headcount.

Full methodology, per-question breakdown, and pricing model: **[docs/vertex-vs-ge-comparison.en.md](docs/vertex-vs-ge-comparison.en.md)**

---

## Repository Layout

```
.
├── app.py                  # FastAPI OpenAI-compatible proxy
├── benchmark.py            # 8-question benchmark + LLM-as-a-Judge
├── dashboard.py            # Streamlit arena + analytics
├── discovery_api/
│   ├── config.py           # Env-based settings (no hardcoded secrets)
│   ├── auth.py             # GCP token cache (gcloud CLI / ADC)
│   ├── schemas.py          # Pydantic models + model routing table
│   ├── sse.py              # SSE parser + OpenAI chunk factory
│   ├── citations.py        # Grounding citation extraction
│   ├── handlers.py         # Vertex AI backend
│   ├── agent_search.py     # Gemini Enterprise :streamAssist backend
│   ├── media.py            # GE image/video generation + session file download
│   ├── media_routes.py     # OpenAI-shaped /v1/images, /v1/videos, /v1/media routes
│   └── video_jobs.py       # In-memory async video job store
├── tests/                  # pytest (unit + route tests, GE calls mocked)
├── scripts/
│   ├── list_engines.py     # Discover Agentspace engines in a project
│   └── render_charts.py    # Export benchmark charts as PNG
├── docs/
│   ├── vertex-vs-ge-comparison.en.md
│   ├── vertex-vs-ge-comparison.md      # 中文版
│   └── images/
└── .env.example
```

---

## Contributing

This repo lives under the GTM org — contributions go through PRs:

1. Fork / branch from `main`
2. `cp .env.example .env` and fill in **your own** project — never commit real project IDs
3. Run `python -m pytest tests` and a smoke test before pushing
4. Open a PR; do **not** include `benchmark_results.json`, `.env`, or any file containing project IDs / tokens

---

## Disclaimer

This is a GTM enablement asset, not an official Google product. Pricing and quotas referenced in the report are point-in-time observations — always verify against the [official pricing page](https://cloud.google.com/vertex-ai/generative-ai/pricing) before making purchasing decisions.
