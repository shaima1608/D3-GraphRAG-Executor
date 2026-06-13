from pathlib import Path

from src.config import settings
from src.metrics import evaluate_search, load_eval_questions

if __name__ == '__main__':
    df = load_eval_questions(settings.questions_path)
    result = evaluate_search(df)
    Path('reports').mkdir(exist_ok=True)
    result['records'].to_csv('reports/d2_search_examples.csv', index=False)
    with open('reports/d2_metrics_table.md', 'w', encoding='utf-8') as f:
        f.write('| Metric | Value |\n|---|---:|\n')
        for k, v in result['metrics'].items():
            f.write(f'| {k} | {v:.4f} |\n' if isinstance(v, float) else f'| {k} | {v} |\n')
    print(result['metrics'])
    print('Saved reports/d2_metrics_table.md and reports/d2_search_examples.csv')
