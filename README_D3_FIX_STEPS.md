# D3 fixes to apply before submission

This patch fixes the main grading risk found in the D3 project review: `gold_hit_proxy` stayed `0` because the evaluator did not read the course spreadsheet column `Correct_PDF`.

## Files to replace

Copy these files into your repository and replace the existing versions:

```text
src/d3_eval.py
scripts/evaluate_d3.py
scripts/run_ablation_d3.py
```

## PowerShell commands

Run these inside your project folder:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:MONGO_URI = "mongodb://localhost:27017"
$env:QDRANT_URL = "http://localhost:6333"
$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "password123"
$env:PDF_DIR = "data/pdfs"
$env:QUESTIONS_PATH = "data/question.xlsx"

docker compose up -d mongo qdrant neo4j
python scripts/evaluate_d3.py
python scripts/run_ablation_d3.py
python scripts/safety_demo_d3.py
```

## What should improve

- `gold_hit_proxy` / `gold_source_hit` should no longer stay `0` if the correct PDF appears in the retrieved evidence.
- The report now explains the local RAGAS-style proxy metrics clearly.
- Latency is measured after a warm-up query, so the first model/index loading time is not unfairly counted.
- The ablation table now uses the clearer name `Gold Source Hit`.

## Add this sentence to the README/report

> The D3 evaluation uses local RAGAS-style proxy metrics because the project is designed to be reproducible without external API keys. Faithfulness is measured by answer-token support in retrieved evidence, answer relevance is measured by query-term coverage in the answer/evidence, and gold source hit checks whether the retrieved citations include the expected `Correct_PDF` from the gold set.

## GitHub commit commands

```powershell
git add src/d3_eval.py scripts/evaluate_d3.py scripts/run_ablation_d3.py reports/d3_eval_results.* reports/d3_eval_rows.csv reports/d3_ablation_*.json reports/d3_ablation_table.md
git commit -m "Fix D3 gold source evaluation and report metrics"
git push origin main
```

For the group-member commit requirement, ask each member to make one real small commit, such as adding their task ownership section, demo screenshot, or README note.
