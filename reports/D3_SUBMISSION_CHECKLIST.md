# D3 Submission Checklist

| D3 requirement | Implemented file(s) | Evidence | Status |
|---|---|---|---|
| Choose subgraph by Cypher | `src/graphrag.py`, `src/graph.py` | `choose_subgraph()` selects Paper/Topic/Author neighborhoods | Complete |
| Expand to supporting chunks | `src/graphrag.py`, `src/search.py` | `expand_supporting_chunks()` fetches/ranks chunks from selected papers | Complete |
| Hybrid blend and rerank | `src/graphrag.py`, `src/search.py` | `blend_and_rerank()` combines graph and hybrid candidates | Complete |
| Answer with citations/page ranges | `src/graphrag.py`, `src/safety.py` | `generate_grounded_answer()` and `citation_for()` include page metadata | Complete |
| Gold Q/A evaluation | `data/question.xlsx`, `scripts/evaluate_d3.py` | `reports/d3_eval_results.md/json` | Complete |
| Faithfulness/relevance/p95 latency | `src/d3_eval.py` | `reports/d3_eval_results.md/json` | Complete |
| Safety mitigation + before/after | `src/safety.py`, `scripts/safety_demo_d3.py` | `reports/d3_safety_evidence.md/json` | Complete |
| Ablation | `scripts/run_ablation_d3.py` | `reports/d3_ablation_table.md/json` | Complete |
| API/UI | `app/main.py` | `/ask`, `/graphrag`, `/d3/evaluate`, `/d3/ablation`, `/d3/safety-demo` | Complete |


