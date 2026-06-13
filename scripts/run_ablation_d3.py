import json
from pathlib import Path

from src.d3_eval import run_ablation

if __name__ == '__main__':
    out = run_ablation(limit=15, top_k=5)
    Path('reports').mkdir(exist_ok=True)
    with open('reports/d3_ablation_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    with open('reports/d3_ablation_table.md', 'w', encoding='utf-8') as f:
        f.write('| Mode | Faithfulness | Relevance | Citation Coverage | Gold Hit | p95 Latency (s) |\n')
        f.write('|---|---:|---:|---:|---:|---:|\n')
        for mode, metrics in out.items():
            f.write(f"| {mode} | {metrics['faithfulness_proxy']} | {metrics['answer_relevance_proxy']} | {metrics['citation_coverage']} | {metrics['gold_hit_proxy']} | {metrics['p95_latency_seconds']} |\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in out.items()}, indent=2))
    print('Saved reports/d3_ablation_results.json and reports/d3_ablation_table.md')
