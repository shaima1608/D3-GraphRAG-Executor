"""D3 safety controls for GraphRAG.

This module implements lightweight safety mitigations required by D3:
1. source pinning / provenance filtering,
2. prompt-injection filtering for query and retrieved chunks,
3. risky tool-call and write-Cypher denial.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s+prompt",
    r"developer\s+message",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"reveal\s+(the\s+)?prompt",
    r"delete\s+the\s+database",
    r"drop\s+collection",
    r"exfiltrate",
    r"send\s+.*secret",
    r"api[_ -]?key",
    r"password",
    r"token",
]

DANGEROUS_CYPHER = [
    "CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP", "LOAD CSV",
    "CALL DBMS", "APOC.LOAD", "APOC.PERIODIC", "APOC.CYBER", "APOC.EXPORT",
]

ALLOWED_CYPHER_STARTS = ("MATCH", "WITH", "RETURN", "CALL DB.LABELS", "CALL DB.RELATIONSHIPTYPES")


def detect_prompt_injection(text: str) -> List[str]:
    """Return matched prompt-injection patterns."""
    text = text or ""
    hits = []
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(pattern)
    return hits


def is_safe_query(query: str) -> Tuple[bool, List[str]]:
    hits = detect_prompt_injection(query)
    return (len(hits) == 0, hits)


def is_safe_cypher(query: str) -> Tuple[bool, str]:
    """Allow read-only Cypher queries and deny write/admin/file operations."""
    q = (query or "").strip()
    upper = re.sub(r"\s+", " ", q.upper())
    if not q:
        return False, "Empty Cypher query."
    if not upper.startswith(ALLOWED_CYPHER_STARTS):
        return False, "Only read-only MATCH/WITH/RETURN or db.labels/db.relationshipTypes queries are allowed."
    for keyword in DANGEROUS_CYPHER:
        if keyword in upper:
            return False, f"Blocked risky Cypher operation: {keyword}."
    return True, "allowed"


def has_required_provenance(result: Dict) -> bool:
    """A result is pinned only if it contains enough citation metadata."""
    return bool(result.get("chunk_id") and result.get("paper_id") and result.get("filename") and result.get("page_start") is not None)


def filter_safe_results(results: Iterable[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Drop chunks with missing provenance or prompt-injection-looking text."""
    safe, removed = [], []
    for r in results:
        reasons = []
        if not has_required_provenance(r):
            reasons.append("missing_provenance")
        hits = detect_prompt_injection(r.get("text", ""))
        if hits:
            reasons.append("prompt_injection_text")
        if reasons:
            removed.append({"chunk_id": r.get("chunk_id"), "reasons": reasons})
        else:
            safe.append(r)
    return safe, removed


def citation_for(result: Dict) -> str:
    title = result.get("title") or result.get("filename") or "Unknown source"
    filename = result.get("filename") or "unknown file"
    page_start = result.get("page_start")
    page_end = result.get("page_end") or page_start
    chunk_id = result.get("chunk_id")
    if page_start == page_end:
        page = f"p. {page_start}"
    else:
        page = f"pp. {page_start}-{page_end}"
    return f"{title} ({filename}, {page}, chunk {chunk_id})"
