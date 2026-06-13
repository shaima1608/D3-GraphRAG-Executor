import time
from typing import Dict, List

import numpy as np
import pandas as pd

from .search import hybrid_search


def load_eval_questions(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine='openpyxl')
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    required = {'Question', 'Correct_PDF'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f'Excel file must contain columns: {required}. Found: {list(df.columns)}')
    return df


def evaluate_search(questions_df: pd.DataFrame, top_k_values=(1, 3, 5)) -> Dict:
    records = []
    latencies = []
    for _, row in questions_df.iterrows():
        q = row['Question']
        correct = str(row['Correct_PDF'])
        result = hybrid_search(q, top_k=max(top_k_values))
        latencies.append(result['latency_ms'] / 1000.0)
        returned = [r['filename'] for r in result['results']]
        rec = {'Question': q, 'Correct_PDF': correct, 'Top_Results': returned, 'Latency_ms': result['latency_ms']}
        for k in top_k_values:
            rec[f'Recall@{k}'] = 1 if correct in returned[:k] else 0
        rec['Top_Citations'] = [r['citation'] for r in result['results'][:5]]
        records.append(rec)
    out = pd.DataFrame(records)
    metrics = {f'Recall@{k}': float(out[f'Recall@{k}'].mean()) for k in top_k_values}
    metrics['p95_latency_seconds'] = float(np.percentile(latencies, 95)) if latencies else 0.0
    metrics['num_queries'] = int(len(out))
    return {'metrics': metrics, 'records': out}
