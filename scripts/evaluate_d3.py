import json
from pathlib import Path

import pandas as pd

from src.d3_eval import run_d3_evaluation


if __name__ == "__main__":
    out = run_d3_evaluation(limit=20, top_k=5)
    Path("reports").mkdir(exist_ok=True)

    with open("reports/d3_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    rows = pd.DataFrame(out.get("rows", []))
    if not rows.empty:
        rows.to_csv("reports/d3_eval_rows.csv", index=False)

    with open("reports/d3_eval_results.md", "w", encoding="utf-8") as f:
        f.write("# D3 Evaluation Results\n\n")
        f.write("These are local RAGAS-style proxy metrics that run without external API keys.\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---:|\n")
        f.write(f"| Queries | {out['num_queries']} |\n")
        f.write(f"| Faithfulness proxy | {out['faithfulness_proxy']} |\n")
        f.write(f"| Answer relevance proxy | {out['answer_relevance_proxy']} |\n")
        f.write(f"| Citation coverage | {out['citation_coverage']} |\n")
        f.write(f"| Gold source hit | {out['gold_source_hit']} |\n")
        f.write(f"| p95 latency seconds | {out['p95_latency_seconds']} |\n\n")
        f.write("## Metric definitions\n\n")
        f.write("- **Faithfulness proxy:** fraction of answer tokens supported by retrieved evidence.\n")
        f.write("- **Answer relevance proxy:** coverage of important query terms in answer and cited evidence.\n")
        f.write("- **Citation coverage:** fraction of answers with at least one citation.\n")
        f.write("- **Gold source hit:** whether the retrieved evidence includes the gold `Correct_PDF` source.\n")
        f.write("- **p95 latency:** 95th percentile runtime after warm-up.\n\n")
        if not rows.empty:
            f.write("## Per-question evidence check\n\n")
            cols = ["query", "correct_pdf", "gold_source_hit", "faithfulness_proxy", "answer_relevance_proxy", "latency_seconds"]
            f.write(rows[cols].to_markdown(index=False))
            f.write("\n")

    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    print("Saved reports/d3_eval_results.json, reports/d3_eval_results.md, and reports/d3_eval_rows.csv")
