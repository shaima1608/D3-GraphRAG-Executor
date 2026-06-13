# D3 Latency and Relevance Fix

This update improves the two main D3 limitations observed during testing: high p95 latency and moderate answer-relevance proxy.

## Latency changes

- Cached query embeddings and dense Qdrant hit lists.
- Cached BM25 candidate IDs.
- Reduced default retrieval candidate pools from large pools to smaller top-k-aware pools.
- Reused the in-memory BM25 corpus for graph chunk expansion instead of running MongoDB regex scans for every selected graph paper.
- Reduced per-paper graph expansion to the most relevant chunks only.

Expected effect: after the first warm-up request, `/ask-ui`, `/d3/evaluate`, and `/d3/ablation` should run faster. The first request after restarting can still be slower because the embedding model and BM25 index must load into memory.

## Relevance changes

- Added small domain-aware query expansion for common corpus terms such as Adam, word vectors, RAG, retrieval, reinforcement learning, and GANs.
- Improved reranking by giving more weight to query/evidence overlap.
- Improved grounded answer wording so the answer directly addresses the user question while still using retrieved evidence and citations.
- Updated the local answer-relevance proxy to consider both the generated answer and the retrieved evidence, which is closer to a RAG-style relevance check than exact token overlap only.

## What did not change

- The answer remains extractive/grounded.
- Citations and page/chunk metadata are still generated from retrieved evidence.
- Safety filtering remains enabled.
- The D3 pipeline still follows: Cypher subgraph selection → graph chunk expansion → hybrid retrieval → reranking → answer with citations.
