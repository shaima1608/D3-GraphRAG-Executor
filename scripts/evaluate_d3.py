import json
from pathlib import Path

from src.d3_eval import run_d3_evaluation

if __name__ == '__main__':
    out = run_d3_evaluation(limit=20, top_k=5)
    Path('reports').mkdir(exist_ok=True)
    with open('reports/d3_eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    with open('reports/d3_eval_results.md', 'w', encoding='utf-8') as f:
        f.write('# D3 Evaluation Results\n\n')
        f.write(f"- Queries: {out['num_queries']}\n")
        f.write(f"- Faithfulness proxy: {out['faithfulness_proxy']}\n")
        f.write(f"- Answer relevance proxy: {out['answer_relevance_proxy']}\n")
        f.write(f"- Citation coverage: {out['citation_coverage']}\n")
        f.write(f"- Gold hit proxy: {out['gold_hit_proxy']}\n")
        f.write(f"- p95 latency seconds: {out['p95_latency_seconds']}\n")
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=2))
    print('Saved reports/d3_eval_results.json and reports/d3_eval_results.md')
