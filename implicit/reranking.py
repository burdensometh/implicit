"""Category-aware re-ranking of model candidates with per-category quotas.

The model candidate list for a request can be dominated by a single category.
This module re-ranks those candidates so that every category keeps a guaranteed
share of the slots, while still preferring the highest scored items inside each
category.

The re-ranking is a deterministic, dependency-free pipeline so that a consumer
can copy it and reproduce every selection exactly:

1. ``filter_purchased`` drops items the user has already bought (and de-duplicates
   repeated item ids, keeping the first occurrence).
2. ``order_candidates`` establishes the *model order*: score descending, with
   ties broken by the original input position, so equal-score items keep their
   model-provided order.
3. ``apply_category_caps`` walks the model order once and admits an item while
   its category quota is not exhausted; items blocked by a quota are parked.
4. When fewer than ``n`` items were admitted, the parked items are back-filled in
   model order. If the pool is still too small (for example after the purchased
   filter), a shorter list is returned rather than raising.

Items without a category form their own group (sentinel ``UNCATEGORIZED`` /
``None``); they never consume the quota of a real category.

Every returned record carries the untouched model score, the 1-based model rank
(after purchased filtering) and the 1-based final rank, which is everything a
consumer needs to recompute the selection.
"""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

# Category sentinel for items that have no category. ``None`` may also be used
# directly as a key of ``category_caps`` to cap this group.
UNCATEGORIZED = None


class RerankingError(ValueError):
    """Raised when the re-ranking inputs cannot be interpreted."""


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def _as_python(value: Any) -> Any:
    """Convert numpy (and similar array-scalar) values to plain Python types."""
    if value is None:
        return None
    converter = getattr(value, "item", None)
    if callable(converter):
        try:
            return converter()
        except TypeError:
            return value
    return value


def _is_iterable(value: Any) -> bool:
    if isinstance(value, (str, bytes, Mapping)):
        return False
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _is_parallel_arrays(candidates: Any) -> bool:
    """Detect the ``(item_ids, scores)`` parallel-array input form.

    Parallel arrays are passed as a 2-tuple of iterables; a list is always a
    sequence of candidate records (a list of exactly two records must not be
    mistaken for arrays). A single ``(item_id, score)`` pair is also excluded
    because its halves are scalars.
    """
    if not isinstance(candidates, tuple) or len(candidates) != 2:
        return False
    ids, scores = candidates
    return _is_iterable(ids) and _is_iterable(scores)


def _normalize_candidates(
    candidates: Any, categories: Optional[Mapping[Any, Any]] = None
) -> List[dict]:
    """Normalize every supported input shape into internal candidate dicts.

    Accepted shapes:

    * parallel arrays: ``([item_id, ...], [score, ...])`` with an optional
      ``categories`` mapping of item id -> category;
    * mappings with keys ``item_id``/``id``, ``score`` and optional
      ``category``/``cat`` (records produced by :func:`rerank` are accepted
      unchanged, which makes recomputation straightforward);
    * tuples/lists of ``(item_id, score)`` or ``(item_id, score, category)``.
    """
    if candidates is None:
        raise RerankingError("candidates must not be None")

    if _is_parallel_arrays(candidates):
        ids, scores = candidates
        raw_candidates = list(zip(ids, scores))
    else:
        if isinstance(candidates, (str, bytes, Mapping)):
            raise RerankingError(
                "candidates must be a sequence of candidate records or an "
                "(item_ids, scores) pair"
            )
        raw_candidates = list(candidates)

    normalized: List[dict] = []
    for position, raw in enumerate(raw_candidates):
        tie_rank = None
        if isinstance(raw, Mapping):
            if "item_id" in raw:
                item_id = raw["item_id"]
            elif "id" in raw:
                item_id = raw["id"]
            else:
                raise RerankingError(f"candidate at position {position} has no item_id")
            score = raw.get("score", raw.get("model_score"))
            category = raw.get("category", raw.get("cat", UNCATEGORIZED))
            tie_rank = raw.get("model_rank")
        elif isinstance(raw, (tuple, list)):
            if len(raw) == 2:
                item_id, score = raw
                category = UNCATEGORIZED
            elif len(raw) == 3:
                item_id, score, category = raw
            else:
                raise RerankingError(
                    f"candidate at position {position} must have 2 or 3 fields"
                )
        else:
            raise RerankingError(
                f"unsupported candidate type at position {position}: {type(raw)!r}"
            )

        item_id = _as_python(item_id)
        score = _as_python(score)
        category = _as_python(category)
        if category is UNCATEGORIZED and categories is not None:
            category = _as_python(categories.get(item_id))
        if tie_rank is not None:
            tie_rank = _as_python(tie_rank)

        if score is None:
            raise RerankingError(f"candidate {item_id!r} has no score")
        try:
            score = float(score)
        except (TypeError, ValueError) as exc:
            raise RerankingError(
                f"candidate {item_id!r} has a non-numeric score: {score!r}"
            ) from exc

        normalized.append(
            {
                "item_id": item_id,
                "score": score,
                "category": category,
                "source_index": position,
                "tie_rank": tie_rank,
            }
        )
    return normalized


# ---------------------------------------------------------------------------
# Pipeline stages (public so each step is independently auditable)
# ---------------------------------------------------------------------------


def filter_purchased(
    candidates: Sequence[Mapping[str, Any]],
    already_purchased: Optional[Iterable[Any]] = None,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Remove purchased items and duplicate item ids.

    Returns ``(eligible, removed_purchased, removed_duplicate)``. The relative
    order of eligible candidates is preserved.
    """
    purchased = {_as_python(item_id) for item_id in (already_purchased or ())}
    eligible: List[dict] = []
    removed_purchased: List[dict] = []
    removed_duplicate: List[dict] = []
    seen = set()
    for candidate in candidates:
        item_id = candidate["item_id"]
        if item_id in purchased:
            removed_purchased.append(candidate)
        elif item_id in seen:
            removed_duplicate.append(candidate)
        else:
            seen.add(item_id)
            eligible.append(candidate)
    return eligible, removed_purchased, removed_duplicate


def order_candidates(candidates: Sequence[Mapping[str, Any]]) -> List[dict]:
    """Return candidates in model order: score descending, ties keep input order.

    NaN scores are valid floats but cannot be ordered; they sink to the end so
    they can never displace a real candidate.

    A record that already carries a ``model_rank`` (i.e. output of a previous
    :func:`rerank` call) uses it as the tie-breaker, so recomputation is exact.
    """

    def sort_key(candidate: Mapping[str, Any]):
        score = candidate["score"]
        # ascending sort: the most negative key wins; NaN maps to +infinity.
        score_key = math.inf if math.isnan(score) else -score
        tie_rank = candidate.get("tie_rank")
        source_index = candidate.get("source_index", 0)
        return score_key, tie_rank if tie_rank is not None else source_index

    return sorted(candidates, key=sort_key)


def _resolve_cap(
    category: Any,
    category_caps: Mapping[Any, int],
    default_cap: Optional[int],
    uncategorized_cap: Optional[int],
) -> Optional[int]:
    if category is UNCATEGORIZED:
        if uncategorized_cap is not None:
            return uncategorized_cap
        if UNCATEGORIZED in category_caps:
            return category_caps[UNCATEGORIZED]
        return default_cap
    if category in category_caps:
        return category_caps[category]
    return default_cap


def _validate_caps(
    category_caps: Mapping[Any, int],
    default_cap: Optional[int],
    uncategorized_cap: Optional[int],
) -> None:
    if not isinstance(category_caps, Mapping):
        raise RerankingError("category_caps must be a mapping of category -> cap")
    for cap in list(category_caps.values()) + [default_cap, uncategorized_cap]:
        invalid = cap is not None and (
            not isinstance(cap, int) or isinstance(cap, bool) or cap < 0
        )
        if invalid:
            raise RerankingError(f"category caps must be non-negative integers, got {cap!r}")


def apply_category_caps(
    ordered_candidates: Sequence[Mapping[str, Any]],
    category_caps: Mapping[Any, int],
    n: Optional[int] = None,
    default_cap: Optional[int] = None,
    uncategorized_cap: Optional[int] = None,
    backfill: bool = True,
) -> Tuple[List[dict], List[dict], dict, int]:
    """Greedily apply per-category caps in model order.

    Parameters
    ----------
    ordered_candidates:
        Candidates already sorted by :func:`order_candidates`.
    category_caps:
        Mapping of category to the maximum number of slots it may occupy during
        the greedy pass. The ``None`` key caps the uncategorized group.
    n:
        Target number of selections. When ``None`` the caps are strict: only the
        greedy pass runs and no back-filling happens.
    default_cap:
        Cap for categories absent from ``category_caps`` (default: unlimited).
    uncategorized_cap:
        Cap for items without a category (default: unlimited, or the ``None``
        entry of ``category_caps`` when present).
    backfill:
        When ``True`` (default) and fewer than ``n`` items were admitted, parked
        items are admitted in model order until ``n`` slots are filled or the
        pool is exhausted.

    Returns
    -------
    tuple
        ``(selected, parked, category_counts, num_backfilled)``.
    """
    _validate_caps(category_caps, default_cap, uncategorized_cap)
    if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 0):
        raise RerankingError(f"n must be a non-negative integer, got {n!r}")

    target = n if n is not None else len(ordered_candidates)
    selected: List[dict] = []
    parked: List[dict] = []
    category_counts: dict = {}

    for candidate in ordered_candidates:
        if len(selected) >= target:
            parked.append(candidate)
            continue
        category = candidate["category"]
        cap = _resolve_cap(category, category_caps, default_cap, uncategorized_cap)
        used = category_counts.get(category, 0)
        if cap is not None and used >= cap:
            parked.append(candidate)
            continue
        selected.append(candidate)
        category_counts[category] = used + 1

    num_backfilled = 0
    if n is not None and backfill and len(selected) < n:
        for candidate in parked:
            if len(selected) >= n:
                break
            selected.append(candidate)
            num_backfilled += 1
            category = candidate["category"]
            category_counts[category] = category_counts.get(category, 0) + 1

    return selected, parked, category_counts, num_backfilled


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _build_records(
    selected: Sequence[Mapping[str, Any]], ordered: Sequence[Mapping[str, Any]]
):
    # Records fed back from a previous run carry their original model_rank; keep
    # it so a recomputation reproduces ranks even after purchased filtering.
    model_rank_by_index = {
        id(candidate): (
            candidate["tie_rank"]
            if candidate.get("tie_rank") is not None
            else rank
        )
        for rank, candidate in enumerate(ordered, start=1)
    }
    records = []
    for final_rank, candidate in enumerate(selected, start=1):
        records.append(
            {
                "item_id": candidate["item_id"],
                "score": candidate["score"],
                "category": candidate["category"],
                "model_rank": model_rank_by_index[id(candidate)],
                "rank": final_rank,
            }
        )
    return records


def rerank(
    candidates: Any,
    category_caps: Mapping[Any, int],
    n: Optional[int] = None,
    *,
    already_purchased: Optional[Iterable[Any]] = None,
    categories: Optional[Mapping[Any, Any]] = None,
    default_cap: Optional[int] = None,
    uncategorized_cap: Optional[int] = None,
    backfill: bool = True,
) -> List[dict]:
    """Re-rank model candidates under per-category caps.

    See the module docstring for the full pipeline.

    Parameters
    ----------
    candidates:
        Model candidates in any of the forms accepted by
        :func:`_normalize_candidates` (mapping records, ``(id, score[, category])``
        tuples, or parallel ``(ids, scores)`` arrays).
    category_caps:
        Maximum greedy-pass slots per category. Use the ``None`` key (or
        ``uncategorized_cap``) to cap items without a category.
    n:
        Number of slots to fill. When the eligible pool cannot fill every slot
        the result is simply shorter. When ``None``, caps are applied strictly
        without back-filling.
    already_purchased:
        Item ids to filter out before re-ranking.
    categories:
        Optional item id -> category mapping for inputs that lack categories.
    default_cap:
        Cap for categories not listed in ``category_caps`` (default unlimited).
    uncategorized_cap:
        Cap for the uncategorized group (default unlimited).
    backfill:
        Whether parked items may fill leftover slots.

    Returns
    -------
    list of dict
        One record per selected item, ordered by final rank (1-based). Each
        record contains ``item_id``, the untouched ``score``, ``category``,
        ``model_rank`` (rank after purchased filtering, before caps) and
        ``rank`` (the final position).
    """
    normalized = _normalize_candidates(candidates, categories)
    eligible, _purchased, _duplicates = filter_purchased(normalized, already_purchased)
    ordered = order_candidates(eligible)
    selected, _parked, _counts, _backfilled = apply_category_caps(
        ordered,
        category_caps,
        n=n,
        default_cap=default_cap,
        uncategorized_cap=uncategorized_cap,
        backfill=backfill,
    )
    return _build_records(selected, ordered)


def rerank_with_report(
    candidates: Any,
    category_caps: Mapping[Any, int],
    n: Optional[int] = None,
    *,
    already_purchased: Optional[Iterable[Any]] = None,
    categories: Optional[Mapping[Any, Any]] = None,
    default_cap: Optional[int] = None,
    uncategorized_cap: Optional[int] = None,
    backfill: bool = True,
) -> dict:
    """Same re-ranking as :func:`rerank`, plus an auditable processing report."""
    normalized = _normalize_candidates(candidates, categories)
    num_candidates = len(normalized)
    eligible, removed_purchased, removed_duplicate = filter_purchased(
        normalized, already_purchased
    )
    ordered = order_candidates(eligible)
    selected, parked, category_counts, num_backfilled = apply_category_caps(
        ordered,
        category_caps,
        n=n,
        default_cap=default_cap,
        uncategorized_cap=uncategorized_cap,
        backfill=backfill,
    )
    records = _build_records(selected, ordered)
    shortfall = max(0, n - len(records)) if n is not None else 0
    return {
        "selections": records,
        "num_candidates": num_candidates,
        "num_eligible": len(eligible),
        "num_filtered_purchased": len(removed_purchased),
        "num_filtered_duplicate": len(removed_duplicate),
        "num_parked_by_cap": len(parked),
        "num_backfilled": num_backfilled,
        "shortfall": shortfall,
        "category_counts": category_counts,
    }


def selection_arrays(records: Sequence[Mapping[str, Any]]) -> Tuple[Tuple, Tuple, Tuple]:
    """Split selection records into parallel ``(item_ids, scores, ranks)`` tuples."""
    return (
        tuple(record["item_id"] for record in records),
        tuple(record["score"] for record in records),
        tuple(record["rank"] for record in records),
    )


def rerank_to_json(records: Sequence[Mapping[str, Any]]) -> str:
    """Serialize selection records to JSON for a consumer service."""
    return json.dumps(list(records), ensure_ascii=False, sort_keys=True)


# Descriptive aliases for callers that prefer an explicit function name.
rerank_candidates = rerank
category_aware_rerank = rerank
