# D3 Report — GraphRAG Executor, Evaluation, and Safety

## GraphRAG executor

D3 extends the D2 retrieval stack by adding a GraphRAG executor. The executor first checks the user query for prompt-injection or unsafe requests. If the query passes, the system selects a relevant Neo4j subgraph by matching query terms against Paper titles and Topic names. It then expands the selected graph neighborhood into supporting chunks from MongoDB, blends these graph-supported chunks with hybrid BM25 + Qdrant retrieval results, and reranks the combined evidence. The final answer is generated using only retrieved evidence, so every answer can be traced back to cited chunks with page ranges.

## Evaluation

The D3 evaluation script uses the existing question set and computes local RAGAS-style proxy metrics: faithfulness, answer relevance, citation coverage, gold-hit proxy, and p95 latency. Faithfulness measures how much of the generated answer is supported by retrieved evidence. Answer relevance measures query-answer token overlap. Citation coverage checks whether the generated answer includes at least one grounded citation. The evaluation is saved in `reports/d3_eval_results.md` and `reports/d3_eval_results.json`.

## Ablation

The ablation compares three retrieval modes: vector-only retrieval using Qdrant, hybrid retrieval using BM25 + Qdrant, and graph-guided hybrid retrieval using Neo4j subgraph expansion plus hybrid retrieval. The output is saved in `reports/d3_ablation_table.md` and `reports/d3_ablation_results.json`. This comparison demonstrates the quality and latency trade-off between vector retrieval, normal hybrid retrieval, and graph-guided retrieval.

## Safety

The D3 safety layer implements three mitigations. First, user queries and retrieved chunks are checked for prompt-injection patterns such as requests to reveal system prompts, ignore instructions, expose passwords, or delete data. Second, source pinning removes evidence that does not include provenance metadata such as paper ID, filename, chunk ID, and page number. Third, the Cypher endpoint uses a read-only allowlist and blocks dangerous operations such as `DELETE`, `DETACH`, `CREATE`, `MERGE`, `SET`, `LOAD CSV`, and admin/tool calls. Before/after safety evidence is saved in `reports/d3_safety_evidence.md`.

## Conclusion

The D3 implementation satisfies the GraphRAG, evaluation, ablation, and safety requirements. It provides a working `/ask` endpoint, a friendly API tester, evaluation scripts, safety evidence, and graph-guided retrieval that builds directly on the D2 MongoDB, Qdrant, Neo4j, and FastAPI stack.
After initial testing, the D3 system was optimized to reduce latency and improve answer relevance. The final version caches retrieval components, reduces unnecessary candidate expansion, keeps grounded citation logic, and improves the GraphRAG interface for clearer evidence display. The final evaluation and ablation reports were regenerated after these changes.
