"""
Generic search utility functions.

No OpenSearch client, no index-specific dependencies — safe to import from any layer.
"""
from __future__ import annotations

from typing import Any, Dict, List


def truthy_list(v: Any) -> List[str]:
    """Return a list of non-empty stripped strings extracted from *v* (str or list)."""
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def truncate_chars(s: str, *, max_chars: int) -> str:
    t = (s or "").strip()
    return t if len(t) <= max_chars else t[:max_chars]


def chunks_from_query_parts(
    parts: List[str],
    *,
    max_chars: int,
    max_phrases: int,
    max_chunks: int,
) -> List[str]:
    """
    Pack deduplicated phrases into at most *max_chunks* strings, each under *max_chars*,
    for issuing as independent BM25 calls (avoids Lucene maxClauseCount explosions).
    """
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if cur:
            chunks.append(" ".join(cur))
            cur = []
            cur_len = 0

    for p in parts:
        if len(chunks) >= max_chunks:
            break
        piece = str(p or "").strip()
        if not piece:
            continue

        if len(piece) > max_chars:
            flush()
            for i in range(0, len(piece), max_chars):
                if len(chunks) >= max_chunks:
                    break
                chunks.append(piece[i : i + max_chars])
            continue

        add_len = len(piece) + (1 if cur else 0)
        if (cur and cur_len + add_len > max_chars) or (cur and len(cur) >= max_phrases):
            flush()

        if len(chunks) >= max_chunks:
            break

        cur.append(piece)
        cur_len += add_len

    flush()
    return [c for c in chunks if c.strip()]


def rrf_fuse_ranked_lists(
    ranked_lists: List[List[str]],
    *,
    k: int = 60,
    top_n: int = 200,
) -> List[str]:
    """
    Reciprocal Rank Fusion (RRF):
      score(d) = sum_i  1 / (k + rank_i(d)),  rank is 1-based.

    Merges multiple ranked lists of doc IDs into a single re-ranked list.
    """
    scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for idx, doc_id in enumerate(lst or []):
            if not doc_id:
                continue
            rank = idx + 1
            scores[doc_id] = float(scores.get(doc_id, 0.0)) + 1.0 / (float(k) + float(rank))
    out = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return out[:int(top_n)]


def reciprocal_rank_fuse(
    ranked_lists: List[List[str]],
    weights: List[float],
    *,
    rank_constant: int = 60,
    top_n: int = 200,
) -> List[tuple[str, float]]:
    """
    Weighted RRF: score(d) += weight_i / (k + rank_i(d)), rank is 1-based.
    Returns (doc_id, fused_score) sorted by score descending.
    """
    if len(ranked_lists) != len(weights):
        raise ValueError("ranked_lists and weights must have the same length")
    scores: Dict[str, float] = {}
    k = float(rank_constant)
    for lst, w in zip(ranked_lists, weights):
        if float(w) <= 0.0:
            continue
        wf = float(w)
        for idx, doc_id in enumerate(lst or []):
            if not doc_id:
                continue
            rank = float(idx + 1)
            scores[str(doc_id)] = scores.get(str(doc_id), 0.0) + wf / (k + rank)
    out = sorted(scores.items(), key=lambda x: -x[1])
    return out[: int(top_n)]
