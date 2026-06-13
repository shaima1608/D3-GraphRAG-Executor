import pickle
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .config import settings
from .stores import get_mongo, get_qdrant, load_embedding_model

INDEX_PATH = Path('reports/bm25_index.pkl')

# Smaller pools reduce D3 p95 latency while preserving enough candidates for top-k.
DEFAULT_CANDIDATE_POOL = 8
MAX_CANDIDATE_POOL = 20


def _clean_doc(doc: Dict) -> Dict:
    if doc is None:
        return {}
    out = dict(doc)
    out.pop('_id', None)
    return out


def load_chunks_from_mongo() -> List[Dict]:
    db = get_mongo()
    return [_clean_doc(c) for c in db[settings.mongo_chunks_collection].find({}, {'_id': 0})]


def build_bm25_index(chunks: List[Dict] | None = None, output_path: str = str(INDEX_PATH)):
    if chunks is None:
        chunks = load_chunks_from_mongo()
    tokenized = [(c.get('text') or '').lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    data = {'bm25': bm25, 'chunks': chunks}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    load_bm25_index.cache_clear()
    _chunk_by_id.cache_clear()
    _chunks_by_paper.cache_clear()
    return output_path


@lru_cache(maxsize=1)
def load_bm25_index(path: str = str(INDEX_PATH)):
    p = Path(path)
    if not p.exists():
        build_bm25_index(output_path=str(p))
    with open(p, 'rb') as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _chunk_by_id(path: str = str(INDEX_PATH)) -> Dict[str, Dict]:
    data = load_bm25_index(path)
    return {c['chunk_id']: c for c in data['chunks'] if c.get('chunk_id')}


@lru_cache(maxsize=1)
def _chunks_by_paper(path: str = str(INDEX_PATH)) -> Dict[str, List[Dict]]:
    data = load_bm25_index(path)
    out: Dict[str, List[Dict]] = {}
    for c in data['chunks']:
        pid = c.get('paper_id')
        if pid:
            out.setdefault(pid, []).append(c)
    return out


def get_chunks_for_paper(paper_id: str, path: str = str(INDEX_PATH)) -> List[Dict]:
    return _chunks_by_paper(path).get(paper_id, [])


def normalize_scores(score_dict: Dict[str, float]) -> Dict[str, float]:
    if not score_dict:
        return {}
    vals = np.array(list(score_dict.values()), dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return {k: 1.0 for k in score_dict}
    return {k: (v - lo) / (hi - lo) for k, v in score_dict.items()}


def _candidate_limit(top_k: int) -> int:
    return min(MAX_CANDIDATE_POOL, max(DEFAULT_CANDIDATE_POOL, int(top_k) * 2))


@lru_cache(maxsize=256)
def _cached_query_vector(query: str) -> Tuple[float, ...]:
    model = load_embedding_model()
    vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    return tuple(float(x) for x in vec)


@lru_cache(maxsize=256)
def _cached_dense_hits(query: str, limit: int) -> Tuple[Tuple[str, float], ...]:
    qvec = list(_cached_query_vector(query))
    qdrant = get_qdrant()
    hits = qdrant.search(
        collection_name=settings.qdrant_collection,
        query_vector=qvec,
        limit=limit,
        with_payload=True,
    )
    pairs = []
    for h in hits:
        payload = h.payload or {}
        cid = payload.get('chunk_id')
        if cid:
            pairs.append((cid, float(h.score)))
    return tuple(pairs)


@lru_cache(maxsize=256)
def _cached_bm25_ids(query: str, limit: int, bm25_path: str = str(INDEX_PATH)) -> Tuple[Tuple[str, float], ...]:
    data = load_bm25_index(bm25_path)
    bm25, chunks = data['bm25'], data['chunks']
    raw = np.array(bm25.get_scores(query.lower().split()), dtype=float)
    candidate_idx = raw.argsort()[-limit:][::-1]
    pairs = []
    for i in candidate_idx:
        if int(i) < len(chunks) and chunks[int(i)].get('chunk_id'):
            pairs.append((chunks[int(i)]['chunk_id'], float(raw[int(i)])))
    return tuple(pairs)


def _result_from_chunk(ch: Dict, score: float, bm25_score: float, dense_score: float) -> Dict:
    return {
        'score': float(score),
        'chunk_id': ch.get('chunk_id'),
        'paper_id': ch.get('paper_id'),
        'filename': ch.get('filename'),
        'title': ch.get('title'),
        'page_start': ch.get('page_start'),
        'page_end': ch.get('page_end'),
        'citation': ch.get('citation'),
        'text': (ch.get('text') or ''),
        'bm25_score': float(bm25_score),
        'dense_score': float(dense_score),
    }


def hybrid_search(query: str, top_k: int = 5, alpha: float = None, bm25_path: str = str(INDEX_PATH)) -> Dict:
    alpha = settings.hybrid_alpha if alpha is None else float(alpha)
    started = time.time()
    limit = _candidate_limit(top_k)

    bm25_scores = normalize_scores(dict(_cached_bm25_ids(query, limit, bm25_path)))
    dense_scores = normalize_scores(dict(_cached_dense_hits(query, limit)))

    chunk_by_id = _chunk_by_id(bm25_path)
    all_ids = set(bm25_scores) | set(dense_scores)
    fused = []
    for cid in all_ids:
        score = alpha * dense_scores.get(cid, 0.0) + (1.0 - alpha) * bm25_scores.get(cid, 0.0)
        ch = chunk_by_id.get(cid, {})
        fused.append(_result_from_chunk(ch, score, bm25_scores.get(cid, 0.0), dense_scores.get(cid, 0.0)))
    fused.sort(key=lambda x: x['score'], reverse=True)
    return {'query': query, 'latency_ms': round((time.time() - started) * 1000, 2), 'results': fused[:top_k]}


def mongo_lookup(chunk_id: str) -> Dict:
    ch = _chunk_by_id().get(chunk_id)
    if ch:
        return ch
    db = get_mongo()
    return db[settings.mongo_chunks_collection].find_one({'chunk_id': chunk_id}, {'_id': 0}) or {}


def vector_only_search(query: str, top_k: int = 5) -> Dict:
    started = time.time()
    chunk_by_id = _chunk_by_id()
    results = []
    for cid, score in _cached_dense_hits(query, top_k):
        ch = chunk_by_id.get(cid, {})
        results.append(_result_from_chunk(ch, score, 0.0, score))
    return {'query': query, 'latency_ms': round((time.time() - started) * 1000, 2), 'results': results}


def bm25_only_search(query: str, top_k: int = 5, bm25_path: str = str(INDEX_PATH)) -> Dict:
    started = time.time()
    chunk_by_id = _chunk_by_id(bm25_path)
    raw_pairs = list(_cached_bm25_ids(query, top_k, bm25_path))
    max_score = raw_pairs[0][1] if raw_pairs else 1.0
    results = []
    for cid, raw in raw_pairs:
        score = float(raw / max_score) if max_score else 0.0
        ch = chunk_by_id.get(cid, {})
        results.append(_result_from_chunk(ch, score, score, 0.0))
    return {'query': query, 'latency_ms': round((time.time() - started) * 1000, 2), 'results': results}
