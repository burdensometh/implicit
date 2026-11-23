"""Session-only recommendations for anonymous (cold-start) users.

This module serves users that have no persisted user vector in the model: for
instance an anonymous visitor that has only interacted with items during the
current browser session. Nothing here mutates a fitted model - the fitted
``item_factors`` / ``user_factors`` are read-only, and any user representation
that is needed is calculated on the fly and returned to the caller instead of
being stored on the model.

Three kinds of sessions are distinguished:

``SessionRecommendation.REGIME_INFERRED``
    The session contains feedback for at least one item the model was fit on.
    A temporary user factor is inferred from that feedback with
    :meth:`MatrixFactorizationBase.recalculate_user` and used to rank the
    catalogue. The inferred factor is *not* written back to the model.

``SessionRecommendation.REGIME_NO_PREFERENCE``
    The session carries no explicit feedback at all (an empty session, or one
    where every feedback entry has a zero weight). There is nothing to infer a
    vector from, so the configured popularity fallback is used.

``SessionRecommendation.REGIME_UNMODELED_ONLY``
    Every feedback item is outside the range of items the model knows about
    (for example items added to the catalogue after the model was fit). A
    meaningful temporary vector cannot be built, so the popularity fallback is
    used.

In both fallback regimes the caller-supplied popularity ordering is used, which
lets the application decide which "crowd" a cold-start user is assigned to
(global most-popular, popularity within a region or segment, promoted items,
...). Items the user has already given feedback on - including items the model
has never seen - are always filtered out of the result.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class SessionRecommendation:
    """Result of a session-only recommendation request.

    Attributes
    ----------
    ids : ndarray
        The recommended item ids, ordered by descending score.
    scores : ndarray
        The score for every id in ``ids``. For inferred sessions these are the
        dot products of the temporary user factor with the item factors; for
        fallback sessions they are the supplied popularity values.
    regime : str
        One of :attr:`REGIME_INFERRED`, :attr:`REGIME_NO_PREFERENCE` and
        :attr:`REGIME_UNMODELED_ONLY` - describing how the result was produced.
    user_factors : ndarray, optional
        The temporary user factor that was inferred from the session. This is
        ``None`` for the two fallback regimes and is never persisted on the
        model.
    unmodeled_items : ndarray
        Feedback item ids that fall outside the model's catalogue and therefore
        could not contribute to an inferred factor. They are still excluded
        from the recommendations.
    """

    REGIME_INFERRED = "inferred"
    REGIME_NO_PREFERENCE = "no_preference"
    REGIME_UNMODELED_ONLY = "unmodeled_only"

    ids: np.ndarray
    scores: np.ndarray
    regime: str
    user_factors: np.ndarray
    unmodeled_items: np.ndarray


def item_popularity(user_items):
    """Counts interactions per item as a popularity signal.

    The returned array can be passed as ``popularity`` to
    :func:`recommend_for_session` to back the cold-start fallback with the
    global interaction popularity of the training data.

    Parameters
    ----------
    user_items : csr_matrix
        The (users, items) confidence matrix the model was fit on.

    Returns
    -------
    ndarray
        Array of total interaction weight per item, with one entry per column
        of ``user_items``.
    """
    if not isinstance(user_items, csr_matrix):
        user_items = csr_matrix(user_items)
    popularity = np.asarray(user_items.sum(axis=0)).reshape(-1)
    return popularity.astype(np.float64, copy=False)


def recommend_for_session(
    model,
    session_items,
    popularity,
    N=10,
    session_weights=None,
    filter_items=None,
    items=None,
):
    """Recommends items for an anonymous user using only current-session data.

    The fitted ``model`` is never modified: a user vector for the session is at
    most calculated on the fly and returned on the result object. When the
    session does not support inferring a vector, the caller-supplied
    ``popularity`` ordering is used as the cold-start fallback.

    Parameters
    ----------
    model : RecommenderBase
        A fitted recommendation model. Temporary vectors are only supported by
        models implementing ``recalculate_user`` (the CPU and GPU ALS models
        do); calling with modeled session feedback on any other model raises
        ``NotImplementedError`` so the caller can route those users itself.
    session_items : array_like or csr_matrix
        Feedback gathered during the session, as either item ids (optionally
        paired with ``session_weights``) or a single-row CSR matrix of shape
        ``(1, session_catalogue_size)``. The session matrix is allowed to be
        wider than the model catalogue: columns past the model's item range are
        treated as unmodeled items rather than raising an index error.
    popularity : array_like
        Fallback ordering, with one popularity score per item in the *model's*
        catalogue (``model.item_factors.shape[0]`` entries). Higher is more
        popular. This determines the cold-start crowd a user is assigned to.
    N : int, optional
        Maximum number of recommendations to return.
    session_weights : array_like, optional
        Confidence weights matching ``session_items`` when item ids are passed.
        Defaults to all ones. Zero-weight entries do not count as feedback.
    filter_items : array_like, optional
        Extra item ids to exclude from the output (in addition to the session
        feedback items).
    items : array_like, optional
        When set, only these item ids are ranked. Cannot be combined with
        ``filter_items``.

    Returns
    -------
    SessionRecommendation
        The recommendations together with the regime that produced them and any
        inferred temporary user factor.
    """
    if items is not None and filter_items is not None:
        raise ValueError("Can't set both items and filter_items")

    item_count = _model_item_count(model)
    popularity = _check_popularity(popularity, item_count)

    feedback_ids, feedback_weights = _parse_session_feedback(
        session_items, session_weights
    )

    # feedback with a positive/negative weight is an actual signal; zero weights
    # (and an empty session) carry no preference
    meaningful = feedback_weights != 0
    feedback_ids = feedback_ids[meaningful]
    feedback_weights = feedback_weights[meaningful]

    in_range = feedback_ids < item_count
    modeled_ids = feedback_ids[in_range]
    modeled_weights = feedback_weights[in_range]
    unmodeled_items = feedback_ids[~in_range]

    # everything the user has already reacted to, plus explicit exclusions, must
    # never come back - including items the model itself does not know about
    excluded = set(feedback_ids.tolist())
    if filter_items is not None:
        excluded.update(int(i) for i in np.asarray(filter_items).reshape(-1))

    candidates = _resolve_candidates(items, item_count, excluded)

    # regime 1: nothing the user actually told us - assign to the fallback crowd
    if feedback_ids.size == 0:
        return _popularity_fallback(
            popularity, candidates, N,
            SessionRecommendation.REGIME_NO_PREFERENCE, unmodeled_items,
        )

    # regime 2: feedback exists but every item is unknown to the model
    if modeled_ids.size == 0:
        return _popularity_fallback(
            popularity, candidates, N,
            SessionRecommendation.REGIME_UNMODELED_ONLY, unmodeled_items,
        )

    # regime 3: at least one modeled item - infer a temporary user factor, but
    # only on models that can express one without being mutated
    if not _supports_recalculate_user(model):
        raise NotImplementedError(
            "model does not support recalculate_user, so no temporary user "
            "factor can be inferred from the session; provide a popularity-based "
            "recommender or an ALS model instead"
        )

    session_row = csr_matrix(
        (
            modeled_weights.astype(np.float32, copy=False),
            modeled_ids.astype(np.int32, copy=False),
            np.array([0, modeled_ids.size], dtype=np.int32),
        ),
        shape=(1, item_count),
    )
    # read-only inference: recalculate_user returns a factor without storing it
    user_factor = model.recalculate_user(0, session_row)

    # a zero vector (e.g. regularized away) carries no preference either
    if not np.any(user_factor):
        return _popularity_fallback(
            popularity, candidates, N,
            SessionRecommendation.REGIME_NO_PREFERENCE, unmodeled_items,
        )

    ids, scores = _top_n(
        model.item_factors.dot(user_factor), candidates, N
    )
    return SessionRecommendation(
        ids=ids,
        scores=scores,
        regime=SessionRecommendation.REGIME_INFERRED,
        user_factors=user_factor,
        unmodeled_items=unmodeled_items,
    )


def _model_item_count(model):
    item_factors = getattr(model, "item_factors", None)
    if item_factors is not None:
        return item_factors.shape[0]
    # item-item nearest-neighbour models store the catalogue in the
    # square similarity matrix instead of item factors
    similarity = getattr(model, "similarity", None)
    if similarity is not None:
        return similarity.shape[0]
    raise ValueError("model has not been fit yet")


def _check_popularity(popularity, item_count):
    popularity = np.asarray(popularity, dtype=np.float64).reshape(-1)
    if popularity.shape[0] != item_count:
        raise ValueError(
            f"popularity must have one entry per model item ({item_count}), "
            f"got {popularity.shape[0]}"
        )
    return popularity


def _parse_session_feedback(session_items, session_weights):
    if isinstance(session_items, csr_matrix):
        row = session_items
        if row.ndim != 2 or row.shape[0] != 1:
            raise ValueError("a session must be a single CSR row")
        row = row.tocsr()
        ids = row.indices.astype(np.int64, copy=True)
        weights = row.data.astype(np.float64, copy=True)
    else:
        ids = np.asarray(session_items, dtype=np.int64).reshape(-1)
        if session_weights is None:
            weights = np.ones(ids.size, dtype=np.float64)
        else:
            weights = np.asarray(session_weights, dtype=np.float64).reshape(-1)
        if ids.shape != weights.shape:
            raise ValueError("session_items and session_weights must have the same length")

    if ids.size and ids.min() < 0:
        raise ValueError("session item ids must be non-negative")

    order = np.argsort(ids)
    return ids[order], weights[order]


def _supports_recalculate_user(model):
    # True only when recalculate_user is a real implementation rather than the
    # stub on the matrix-factorization base classes that raises
    # NotImplementedError. Walk the MRO manually so this works for both the CPU
    # and GPU ALS classes (which inherit from different base classes).
    for klass in type(model).__mro__:
        candidate = klass.__dict__.get("recalculate_user")
        if candidate is None:
            continue
        # the first class in the MRO that defines the method: a base-class
        # definition means the model itself never overrode the stub
        return "matrix_factorization_base" not in klass.__module__
    return False


def _resolve_candidates(items, item_count, excluded):
    """Returns the modeled item ids that may be recommended."""
    if items is None:
        candidates = np.arange(item_count, dtype=np.int64)
    else:
        candidates = np.asarray(items, dtype=np.int64).reshape(-1)
        if candidates.size and (candidates.min() < 0 or candidates.max() >= item_count):
            raise IndexError("Some itemids in the items parameter are not in the model")

    if not excluded:
        return candidates
    excluded_array = np.fromiter(excluded, dtype=np.int64, count=len(excluded))
    return candidates[~np.isin(candidates, excluded_array)]


def _popularity_fallback(popularity, candidates, N, regime, unmodeled_items):
    ids, scores = _top_n(popularity, candidates, N)
    return SessionRecommendation(
        ids=ids,
        scores=scores,
        regime=regime,
        user_factors=None,
        unmodeled_items=unmodeled_items,
    )


def _top_n(score_by_item, candidates, N):
    """Ranks ``candidates`` by the precomputed per-item score array."""
    if candidates.size == 0 or N <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=score_by_item.dtype)

    candidate_scores = score_by_item[candidates]
    n = min(N, candidates.size)

    # rank globally by descending score, with ascending item id as the
    # deterministic tie-breaker. The whole candidate set is sorted (rather than
    # argpartition-then-locally-sort) so a tie that straddles the Nth position
    # is resolved by the same id rule as the rest of the ranking.
    order = np.lexsort((candidates, -candidate_scores))[:n]
    return candidates[order], candidate_scores[order]
