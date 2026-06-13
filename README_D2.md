# CSAI415 D2 — Retrieval Stack & Graph Build

This package implements D2 on top of the completed D1 retrieval work. It adds a real ingestion pipeline, MongoDB metadata store, Qdrant dense vector store, Neo4j graph, FastAPI `/search`, metrics, citations, Docker Compose, and seed scripts. The package has been updated to include the uploaded D1 PDF corpus and `question.xlsx`.

## D2 checklist

- PDF → text → chunks with overlap and page provenance.
- MongoDB stores document metadata and chunk metadata.
- Qdrant stores dense chunk embeddings.
- BM25 lexical index is saved for hybrid retrieval.
- `/search` FastAPI endpoint combines BM25 and dense retrieval.
- Neo4j graph contains `Author`, `Paper`, `Topic`, and `Venue` nodes with `WROTE`, `ABOUT`, and `PUBLISHED_IN` edges.
- Five example Cypher queries are included in `src/graph.py`.
- Dataflow diagram is saved in `reports/d2_dataflow_diagram.mmd`.
- Metrics script calculates Recall@1, Recall@3, Recall@5, and p95 latency.
- Top-k results include filename/page citations.

## Included D1 dataset

This updated package already includes the uploaded D1 dataset:

```text
data/pdfs/              # 109 scientific PDF files
data/question.xlsx      # 150 D1 evaluation questions
data/pdf_manifest.csv   # list of included PDFs
data/raw_d1/            # reference D1 notebook
```

The Excel file has columns including `Question` and `Correct PDF`. The evaluation code automatically normalizes column names, so `Correct PDF` becomes `Correct_PDF`.

## Run locally with Docker

```bash
cp .env.example .env
docker compose up -d mongo qdrant neo4j
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/seed_stores.py
python scripts/seed_neo4j.py
python scripts/evaluate_search.py
uvicorn app.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

Test `/search` with:

```json
{"query": "What is retrieval augmented generation?", "top_k": 5, "alpha": 0.55}
```

## Docker API option

Because the PDFs and `question.xlsx` are already included, run:

```bash
cp .env.example .env
docker compose up --build
```

Then call:

```bash
curl -X POST http://localhost:8000/ingest
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query":"What is graph retrieval?","top_k":5}'
```

## Required outputs for submission

1. Push this package to GitHub.
2. Run the three seed/evaluation scripts.
3. Include `reports/d2_metrics_table.md` and `reports/d2_search_examples.csv` in the repo.
4. Add the Mermaid diagram from `reports/d2_dataflow_diagram.mmd` to the report.
5. Add screenshots of MongoDB/Qdrant/Neo4j/FastAPI if required by your instructor.

## Notes

The code uses `sentence-transformers/all-MiniLM-L6-v2` for speed. You can change `EMBEDDING_MODEL` in `.env` to `BAAI/bge-small-en-v1.5` if your environment can download and run it.
