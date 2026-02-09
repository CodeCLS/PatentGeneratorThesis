# Graph Validator Chat – Performance: Culprits & Options

## Where the slowness comes from

### 1. **Next.js dev server (`npm run dev`)** – likely main culprit for “website feels slow”

- The app is started with **`npm run dev`** (see `server.py` → `run_nextjs()`).
- In dev mode Next.js compiles pages **on demand**: the first time you open the app or navigate to a new route, it compiles that page. That can take several seconds per page.
- There is no persistent build cache between restarts, so the first load after starting the server is always slow.

**Options:**

| Option | Effort | Effect |
|--------|--------|--------|
| **Use production build for daily use** | Low | Much faster page loads. Run `npm run build` then `npm run start` in `graph_validator_chat/nextjs`, and change `server.py` to start the app with `start` instead of `dev` when you want speed. |
| **Keep using `dev`** | None | Accept slower first load and first visit to each page; subsequent visits to the same page are faster. |

### 2. **Blocking `/api/questions/first`**

- When the chat UI loads, it calls **`/api/questions/first`**.
- If the validator has no questions yet, the handler calls **`generate_questions(validator)`**, which runs four question creators. Several of them call the **LLM** (e.g. entity merging, triple merging).
- That work runs **synchronously** in the request handler, so the HTTP request can hang for many seconds until question generation finishes.

**Options:**

| Option | Effort | Effect |
|--------|--------|--------|
| **Return “generating” immediately and poll** | Medium | Frontend shows “Loading…” and polls for questions; first question appears when ready. No long blocking request. |
| **Pre-generate questions in background** | Low | Rely on `run_initial_analysis()` to populate questions; ensure the UI doesn’t call `/api/questions/first` until `initial_analysis_complete` is true, and return a “still initializing” response until then so the UI can poll. |

### 3. **Validator startup (background, but can delay “ready” state)**

- **`initialize_validator()`** runs in a background thread but:
  - It loads **persisted state** from `persisted_validator.pkl` (can be slow if the file is large).
  - It builds and compiles the LangGraph workflow.
  - It starts **`run_initial_analysis()`**, which runs the full graph (including LLM calls). Until that finishes, the API may report “Validator initializing…” and the UI may wait or retry.
- So “website is slow” can also mean “the app stays in loading/initializing for a long time” because of this background work.

**Options:**

| Option | Effort | Effect |
|--------|--------|--------|
| **Lazy-load persisted state** | Medium | Only load the pickle when the first request needs validator state; show a clear “Starting…” message. |
| **Skip or shorten initial analysis** | Low | Make `run_initial_analysis()` optional or lighter (e.g. skip or reduce LLM steps) so the validator becomes “ready” faster; questions can be generated on demand. |

### 4. **`/api/graph/html`**

- Builds the full graph from triples and generates HTML (including writing a temp file). For **large graphs** this can take a noticeable time and blocks the request.

**Options:**

| Option | Effort | Effect |
|--------|--------|--------|
| **Cache graph HTML** | Medium | Cache the result by (triples/graph) fingerprint and reuse until data changes. |
| **Lazy-load graph tab** | Low | Only call `/api/graph/html` when the user opens the graph tab, not on initial app load. |

### 5. **Pipeline (first run)**

- **`PipelineManager`** is created lazily when you run the pipeline for the first time. Its **`__init__`** loads:
  - **`SentenceClassifier`** – loads a HuggingFace `AutoModelForSequenceClassification` and tokenizer from disk (often 10–30+ seconds).
  - Plus other components (e.g. FAISS merger, formatting, etc.).
- So the **first** “Run pipeline” after starting the server is slow; later runs reuse the same manager.

**Options:**

| Option | Effort | Effect |
|--------|--------|--------|
| **Eagerly create PipelineManager after API start** | Low | Start a background thread that calls `_get_pipeline_manager()` once the server is up, so the first user “Run pipeline” doesn’t pay the cost. |
| **Show “Loading pipeline…”** | Low | Frontend shows a clear message and possibly a progress indicator while the first pipeline request is in progress. |

### 6. **Heavy imports when starting the server**

- Importing **`graph_validator_chat.server`** (e.g. when the notebook runs `start_validator_chat()`) pulls in LangChain, graph tools, Neo4j, pipeline manager module, etc. That can add a few seconds to the **first** run of the cell that starts the website.

**Options:**

| Option | Effort | Effect |
|--------|--------|--------|
| **Lazy imports in server.py** | Medium | Import heavy modules only inside the request paths or background threads that need them (e.g. delay importing `PipelineManager` or LLM code until first use). |
| **Accept one-time cost** | None | Run the “start server” cell once and leave it running; no change. |

---

## Recommended order of actions

1. **Quick win (frontend):** Use a **production build** when you care about speed: in `graph_validator_chat/nextjs` run `npm run build`, then change the start command in `server.py` to use `npm run start` (and the correct port) instead of `npm run dev`. Keep `dev` for development.
2. **Quick win (API):** Avoid blocking the UI on question generation: have **`/api/questions/first`** return immediately with `generating: true` when there are no questions yet, and let the frontend poll; move `generate_questions()` into a background task and store the result when done.
3. **Optional:** Eagerly create **PipelineManager** in a background thread after the API server starts, so the first pipeline run doesn’t pay the full init cost.
4. **Optional:** Cache **`/api/graph/html`** per graph fingerprint so repeated loads are fast.

If you tell me which part feels slowest (first page load, chat loading, graph tab, or pipeline), we can implement the matching options first.
