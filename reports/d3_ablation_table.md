# D3 Ablation: Vector-only vs Hybrid-only vs Graph-guided Hybrid

| Mode | Faithfulness | Relevance | Citation Coverage | Gold Source Hit | p95 Latency (s) |
|---|---:|---:|---:|---:|---:|
| vector_only | 0.8748 | 0.8174 | 1.0 | 0.8667 | 0.2142 |
| hybrid_only | 0.8843 | 0.8817 | 1.0 | 0.9333 | 0.5945 |
| graph_guided_hybrid | 0.8833 | 0.9015 | 1.0 | 0.9333 | 0.8436 |

The graph-guided hybrid mode is the D3 executor: Cypher subgraph selection, supporting chunk expansion, hybrid blending, reranking, citations, and safety filtering.
