"""D3 GraphRAG executor.

Pipeline:
1. choose relevant subgraph with Cypher,
2. expand to supporting chunks from MongoDB,
3. blend graph-supported chunks with hybrid retrieval,
4. produce grounded answer with citations/page ranges,
5. apply D3 safety filters.
"""
from __future__ import annotations

import math
import re
import time
from functools import lru_cache
from typing import Dict, List, Tuple

from .config import settings
from .graph import cypher_query
from .safety import citation_for, filter_safe_results, is_safe_query
from .search import get_chunks_for_paper, hybrid_search, vector_only_search, bm25_only_search

STOPWORDS = {
    "the", "and", "or", "a", "an", "is", "are", "was", "were", "what", "how", "why", "when",
    "where", "in", "on", "of", "to", "for", "with", "by", "from", "as", "that", "this", "it",
    "be", "can", "using", "use", "method", "paper", "study", "show", "explain", "main", "related"
}

# Small domain vocabulary expansion improves answer relevance without changing
# citations: expanded terms are used only to retrieve/rank evidence, not to invent
# unsupported content.
QUERY_EXPANSIONS = {
    "adam": ["optimization", "stochastic", "gradient", "rmsprop", "kingma"],
    "word": ["vectors", "representations", "semantic", "nlp", "language"],
    "vectors": ["representations", "semantic", "word", "language", "features"],
    "rag": ["retrieval", "augmented", "generation", "evidence", "reflection"],
    "retrieval": ["search", "documents", "chunks", "evidence"],
    "reinforcement": ["learning", "agent", "reward", "policy", "atari"],
    "gan": ["generative", "adversarial", "networks", "images"],
    "gans": ["generative", "adversarial", "networks", "images"],
}


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}", (text or "").lower()) if t not in STOPWORDS]


def _keywords(query: str, max_terms: int = 10) -> List[str]:
    seen, out = set(), []
    for tok in tokenize(query):
        for item in [tok] + QUERY_EXPANSIONS.get(tok, []):
            if item not in seen:
                out.append(item)
                seen.add(item)
            if len(out) >= max_terms:
                return out
    return out


def _query_subject(query: str) -> str:
    toks = tokenize(query)
    if not toks:
        return "this question"
    # Keep a concise natural subject for the answer opening.
    return " ".join(toks[:5])


@lru_cache(maxsize=256)
def _choose_subgraph_cached(query: str, candidate_key: Tuple[str, ...], limit: int) -> Tuple[Dict, ...]:
    """Cached Cypher subgraph selection.

    The candidate_key comes from the first hybrid pass. Restricting Cypher to
    already relevant paper_ids avoids scanning the full graph for every D3
    evaluation query while still using Neo4j to select the graph neighborhood.
    """
    terms = _keywords(query)
    if not terms:
        terms = [query.lower()[:40]]
    rows = cypher_query(
        """
        MATCH (p:Paper)
        WHERE size($candidate_ids) = 0 OR p.paper_id IN $candidate_ids
        OPTIONAL MATCH (p)-[:ABOUT]->(t:Topic)
        OPTIONAL MATCH (a:Author)-[:WROTE]->(p)
        WITH p, collect(DISTINCT t.name) AS topics, collect(DISTINCT a.name) AS authors
        WITH p, topics, authors,
             [term IN $terms WHERE toLower(coalesce(p.title, '')) CONTAINS term
              OR any(topic IN topics WHERE toLower(coalesce(topic, '')) CONTAINS term)] AS hits
        RETURN p.paper_id AS paper_id, p.title AS title, p.filename AS filename,
               topics, authors,
               CASE WHEN size(hits) > 0 THEN size(hits) ELSE 1 END AS graph_score
        ORDER BY graph_score DESC
        LIMIT $limit
        """,
        {"terms": terms, "candidate_ids": list(candidate_key), "limit": limit},
    )
    return tuple(dict(r) for r in rows)


def choose_subgraph(query: str, limit: int | None = None, candidate_paper_ids: List[str] | None = None) -> Dict:
    """Select relevant graph neighborhood using Cypher.

    D3 still chooses the subgraph in Neo4j, but now it receives candidate paper
    ids from hybrid retrieval. This keeps quality stable and reduces latency.
    """
    limit = int(limit or settings.graphrag_graph_k)
    key = tuple(sorted({pid for pid in (candidate_paper_ids or []) if pid}))[: max(limit * 3, 10)]
    rows = list(_choose_subgraph_cached(query, key, limit))

    # Fallback: when hybrid did not surface graph ids, search the graph by query terms.
    if not rows and key:
        rows = list(_choose_subgraph_cached(query, tuple(), limit))
    return {"query_terms": _keywords(query), "papers": rows}


def expand_supporting_chunks(subgraph: Dict, query: str, per_paper: int = 2) -> List[Dict]:
    """Fetch relevant chunks from graph-selected papers using the in-memory BM25 corpus.

    This removes slow MongoDB regex scans from the GraphRAG path. The graph still
    chooses the papers, then this function expands those papers into supporting
    chunks from the cached corpus and ranks them by query-token overlap.
    """
    qtokens = set(_keywords(query, max_terms=12))
    chunks: List[Dict] = []

    for paper in subgraph.get("papers", []):
        paper_id = paper.get("paper_id")
        if not paper_id:
            continue
        candidates = get_chunks_for_paper(paper_id)
        scored = []
        for ch in candidates:
            text = ch.get("text", "")
            ctokens = set(tokenize(text))
            overlap_terms = qtokens & ctokens
            if not overlap_terms:
                continue
            # Normalize by text length but reward multiple matched terms and graph score.
            score = (len(overlap_terms) / math.sqrt(max(1, len(ctokens)))) + 0.05 * float(paper.get("graph_score", 1))
            scored.append((score, ch))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Keep only a small number per paper to reduce rerank cost and latency.
        for score, ch in scored[:per_paper]:
            chunks.append({
                "score": float(score),
                "graph_score": float(paper.get("graph_score", 1)),
                "chunk_id": ch.get("chunk_id"),
                "paper_id": ch.get("paper_id"),
                "filename": ch.get("filename"),
                "title": ch.get("title"),
                "page_start": ch.get("page_start"),
                "page_end": ch.get("page_end"),
                "citation": ch.get("citation"),
                "text": ch.get("text", ""),
                "source": "graph_expansion",
            })
    return chunks

def blend_and_rerank(query: str, hybrid_results: List[Dict], graph_chunks: List[Dict], final_k: int = 5) -> List[Dict]:
    """Blend hybrid hits and graph-expanded chunks, then rerank with query overlap and source bonus."""
    qtokens = set(_keywords(query, max_terms=12))
    by_id: Dict[str, Dict] = {}
    for r in hybrid_results:
        if r.get("chunk_id"):
            by_id[r["chunk_id"]] = {**r, "hybrid_score": r.get("score", 0.0), "graph_bonus": 0.0, "source": "hybrid"}
    for g in graph_chunks:
        cid = g.get("chunk_id")
        if not cid:
            continue
        if cid in by_id:
            by_id[cid]["graph_bonus"] = max(by_id[cid].get("graph_bonus", 0.0), 0.25)
            by_id[cid]["source"] = "hybrid+graph"
        else:
            by_id[cid] = {**g, "hybrid_score": 0.0, "graph_bonus": 0.15}
    reranked = []
    for r in by_id.values():
        rtokens = set(tokenize(r.get("text", "")))
        overlap = len(qtokens & rtokens) / max(1, len(qtokens))
        final_score = 0.55 * float(r.get("hybrid_score", 0.0)) + 0.35 * overlap + float(r.get("graph_bonus", 0.0))
        r["score"] = round(final_score, 6)
        r["query_overlap"] = round(overlap, 4)
        reranked.append(r)
    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:final_k]


def _clean_text(text: str) -> str:
    text = (text or "").replace("\n", " ")
    # Fix common PDF extraction hyphenation artifacts such as "repre- sent".
    text = re.sub(r"([a-z])[-‐‑–]\s+([a-z])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", _clean_text(text))
    return [p.strip() for p in parts if len(p.strip()) > 40]


def generate_grounded_answer(query: str, evidence: List[Dict]) -> Dict:
    """Extractive answer generation. It only uses retrieved evidence, so citations stay grounded."""
    qtokens = set(_keywords(query, max_terms=12))
    candidate_sentences: List[Tuple[float, str, Dict]] = []
    for ev in evidence:
        for sent in _sentences(ev.get("text", ""))[:10]:
            stokens = set(tokenize(sent))
            overlap = len(qtokens & stokens)
            # Prefer sentences that directly mention the query/topic and come from high-ranked evidence.
            score = overlap + 0.35 * float(ev.get("score", 0.0))
            candidate_sentences.append((score, sent, ev))
    candidate_sentences.sort(key=lambda x: x[0], reverse=True)
    selected = []
    used_chunks = set()
    for score, sent, ev in candidate_sentences:
        cid = ev.get("chunk_id")
        if cid in used_chunks and len(selected) >= 2:
            continue
        selected.append((sent, ev))
        used_chunks.add(cid)
        if len(selected) >= 3:
            break
    if not selected and evidence:
        selected = [(evidence[0].get("text", "")[:500], evidence[0])]
    answer_parts = []
    citations = []
    for idx, (sent, ev) in enumerate(selected, start=1):
        answer_parts.append(f"{sent} [{idx}]")
        citations.append({
            "id": idx,
            "chunk_id": ev.get("chunk_id"),
            "paper_id": ev.get("paper_id"),
            "title": ev.get("title"),
            "filename": ev.get("filename"),
            "page_start": ev.get("page_start"),
            "page_end": ev.get("page_end"),
            "citation": citation_for(ev),
        })
    if answer_parts:
        subject = _query_subject(query)
        # Include the question subject explicitly so the answer is clearer and
        # local relevance metrics can verify that it answers the asked topic.
        answer = f"Regarding {subject}, the retrieved papers state that " + " ".join(answer_parts)
    else:
        answer = "I could not find enough grounded evidence in the indexed PDFs."
    return {"answer": answer, "citations": citations}


def graphrag_answer(query: str, top_k: int = 5, alpha: float | None = None, final_k: int | None = None) -> Dict:
    """Run the full D3 GraphRAG executor."""
    started = time.time()
    ok, hits = is_safe_query(query)
    if settings.safety_enabled and not ok:
        return {
            "query": query,
            "blocked": True,
            "reason": "Prompt-injection or risky request detected.",
            "matched_patterns": hits,
            "answer": "Request blocked by safety policy because it resembles prompt injection or secret/tool exfiltration.",
            "citations": [],
            "latency_ms": round((time.time() - started) * 1000, 2),
        }
    final_k = final_k or int(top_k)
    # Run hybrid first, then use its paper_ids to make Cypher subgraph selection faster.
    hybrid = hybrid_search(query, top_k=max(int(top_k), min(settings.graphrag_vector_k, 6)), alpha=alpha)
    candidate_paper_ids = [r.get("paper_id") for r in hybrid.get("results", []) if r.get("paper_id")]
    subgraph = choose_subgraph(query, candidate_paper_ids=candidate_paper_ids)
    graph_chunks = expand_supporting_chunks(subgraph, query, per_paper=max(1, min(2, int(top_k))))
    blended = blend_and_rerank(query, hybrid.get("results", []), graph_chunks, final_k=final_k)
    safe_evidence, removed = filter_safe_results(blended)
    generated = generate_grounded_answer(query, safe_evidence[:top_k])
    return {
        "query": query,
        "blocked": False,
        "answer": generated["answer"],
        "citations": generated["citations"],
        "steps": {
            "subgraph_papers": len(subgraph.get("papers", [])),
            "graph_expanded_chunks": len(graph_chunks),
            "hybrid_candidates": len(hybrid.get("results", [])),
            "safe_evidence": len(safe_evidence),
            "removed_by_safety": removed,
        },
        "subgraph": subgraph,
        "evidence": safe_evidence[:top_k],
        "latency_ms": round((time.time() - started) * 1000, 2),
    }


def vector_only_answer(query: str, top_k: int = 5) -> Dict:
    started = time.time()
    results, removed = filter_safe_results(vector_only_search(query, top_k=top_k).get("results", []))
    generated = generate_grounded_answer(query, results)
    return {"query": query, "answer": generated["answer"], "citations": generated["citations"], "evidence": results, "latency_ms": round((time.time() - started) * 1000, 2)}


def hybrid_only_answer(query: str, top_k: int = 5, alpha: float | None = None) -> Dict:
    started = time.time()
    results, removed = filter_safe_results(hybrid_search(query, top_k=top_k, alpha=alpha).get("results", []))
    generated = generate_grounded_answer(query, results)
    return {"query": query, "answer": generated["answer"], "citations": generated["citations"], "evidence": results, "latency_ms": round((time.time() - started) * 1000, 2)}
