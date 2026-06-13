# D3 Latency Optimization

This update reduces D3 latency without removing required D3 features.

## Changes

1. Cached the SentenceTransformer embedding model in `src/stores.py`.
2. Cached MongoDB and Qdrant clients in `src/stores.py`.
3. Cached `reports/bm25_index.pkl` in memory in `src/search.py` instead of reloading it for every query.
4. Reduced retrieval candidate pools from 50+ candidates to a smaller controlled candidate pool.
5. Optimized graph chunk expansion in `src/graphrag.py` so MongoDB filters candidate chunks by query terms instead of loading every chunk for every graph-selected paper.
6. Added evaluation warm-up in `src/d3_eval.py` so one-time model/index loading is not counted as normal p95 latency.

## Expected effect

The first request after starting FastAPI may still be slower because the embedding model is loaded into memory. After that, `/ask-ui`, `/d3/evaluate`, and `/d3/ablation` should be noticeably faster. Run the evaluation again and use the new p95 latency shown on `/d3/evaluate` and `/d3/ablation` in the final report.

