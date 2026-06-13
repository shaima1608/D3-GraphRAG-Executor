# CSAI415 D3 — GraphRAG Executor, Evaluation & Safety

This update builds on the D2 retrieval stack and implements the Week 9 D3 requirements.

## What D3 adds

D3 adds a full GraphRAG executor:

1. Selects a relevant Neo4j subgraph with Cypher.
2. Expands from selected Paper nodes to supporting chunks in MongoDB.
3. Blends graph-expanded chunks with hybrid BM25 + dense Qdrant retrieval.
4. Produces a grounded answer using only retrieved evidence.
5. Returns citations with page ranges and chunk IDs.
6. Runs D3 evaluation and ablation.
7. Adds safety mitigation evidence.

## New D3 endpoints

| Endpoint | Purpose |
|---|---|
| `POST /ask` | Full GraphRAG executor with grounded answer and citations |
| `POST /graphrag` | Alias for `/ask` |
| `GET /ask-ui` | Demo UI for GraphRAG answers |
| `GET /d3/evaluate` | Faithfulness, answer relevance, citation coverage, latency |
| `GET /d3/ablation` | Vector-only vs hybrid-only vs graph-guided hybrid |
| `GET /d3/safety-demo` | Before/after safety evidence |
| `POST /cypher` | Safe read-only Neo4j Cypher query runner |

## Run setup

Start databases:

```powershell
docker compose up -d mongo qdrant neo4j
```

Set local environment variables when running from Windows terminal:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:MONGO_URI = "mongodb://localhost:27017"
$env:QDRANT_URL = "http://localhost:6333"
$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "password123"
$env:PDF_DIR = "data/pdfs"
$env:QUESTIONS_PATH = "data/question.xlsx"
```

Run the API:

```powershell
uvicorn app.main:app --reload
```

Open:

```text
http://localhost:8000
http://localhost:8000/ask-ui
http://localhost:8000/d3/evaluate
http://localhost:8000/d3/ablation
http://localhost:8000/d3/safety-demo
http://localhost:8000/swagger
```

## Run D3 evaluation

```powershell
python scripts/evaluate_d3.py
python scripts/run_ablation_d3.py
python scripts/safety_demo_d3.py
```

These save outputs under `reports/`.

## Safety mitigations

The project implements three safety controls:

1. Prompt-injection pattern detection for user queries and retrieved chunks.
2. Source pinning/provenance filtering, which removes evidence without chunk ID, paper ID, filename, and page number.
3. Read-only Cypher allowlist, which blocks destructive graph commands such as `DELETE`, `DETACH`, `CREATE`, `MERGE`, `SET`, `LOAD CSV`, and admin/tool calls.

## D3 ablation modes

The ablation compares:

- `vector_only`: Qdrant dense retrieval only.
- `hybrid_only`: BM25 + dense Qdrant retrieval.
- `graph_guided_hybrid`: Neo4j subgraph expansion + hybrid retrieval + safety filtering.

## Notes

The answer generator is extractive and grounded: it only composes answers from retrieved chunks. This avoids unsupported generation and makes citation checking easier for the D3 demo.
