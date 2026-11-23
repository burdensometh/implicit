"""Time based (backtest) splitting for recommendation evaluation.

The classic :func:`implicit.evaluation.train_test_split` helper randomly holds
out entries from a sparse user/item matrix.  That protocol lets a model see
interactions that happened *after* the interactions it is evaluated on, which
leaks future information into the past and makes offline metrics look better
than the model would have looked at the time the decision had to be made.

The helpers in this module instead split an *event log* on a single global
cutoff timestamp:

* every positive event at or before the cutoff becomes the training
  interaction matrix;
* only **positive** feedback that happened *after* the cutoff **and** was not
  already present in the user's history at the cutoff is kept as a validation
  target.  This avoids rewarding a model for "predicting" an item that the
  user had already interacted with.

The user and item axes are fixed from the complete event log (or from caller
supplied ``user_ids``/``item_ids``), so the training and validation matrices
always share the same shape and give the same integer id to a given
account/item -- including accounts and items that only appear after the
cutoff.

Accounts without enough history before the cutoff cannot be evaluated fairly.
They are reported (both through the returned :class:`TimeBasedSplitReport` and
the module logger) and their validation targets are dropped, so the resulting
validation matrix only contains targets the model could actually have been
expected to retrieve.

Example
-------
::

    from implicit.backtest import time_based_split, evaluate_time_based_split
    from implicit.als import AlternatingLeastSquares

    train, validation, report = time_based_split(events, cutoff, return_report=True)
    report.summary()
    # report.insufficient_history_users lists accounts with too little history

    model = AlternatingLeastSquares()
    model.fit(train)
    # validation can be passed straight into the real evaluation calls
    metrics = evaluate_time_based_split(model, train, validation, K=10)
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Optional

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

log = logging.getLogger("implicit")

__all__ = [
    "TimeBasedSplitReport",
    "time_based_split",
    "evaluate_time_based_split",
]


@dataclass
class TimeBasedSplitReport:
    """Diagnostics describing a :func:`time_based_split` run.

    Attributes
    ----------
    cutoff : object
        The (possibly normalised) cutoff passed to :func:`time_based_split`.
    num_users, num_items : int
        Sizes of the fixed user/item axes shared by both output matrices.
    num_events : int
        Number of events in the input log.
    num_train_events : int
        Number of positive events at or before the cutoff.
    num_train_pairs : int
        Number of distinct ``(user, item)`` pairs in the training matrix.
    num_validation_candidates : int
        Number of positive events after the cutoff, before any filtering.
    num_seen_targets_dropped : int
        Candidate events removed because the user had already interacted with
        the item at the cutoff.
    num_unknown_item_dropped : int
        Candidate events removed because the item had no positive training
        interaction before the cutoff (and so could not be recommended).
    num_insufficient_history_dropped : int
        Candidate events removed because the user did not have at least
        ``min_history`` distinct items in the training matrix.
        The three "dropped" counts are disjoint: each candidate event is
        attributed to at most one of them.
    num_validation_targets : int
        Number of distinct ``(user, item)`` pairs in the validation matrix.
    insufficient_history_users : ndarray
        Ids (on the fixed user axis) of accounts with fewer than
        ``min_history`` distinct items before the cutoff.
    users_without_validation : ndarray
        Ids of accounts that have no validation targets after filtering.
    num_new_items : int
        Number of items on the fixed item axis with no positive training
        interaction.
    min_history : int
        The minimum training history required for an account to be evaluated.
    """

    cutoff: Any
    num_users: int
    num_items: int
    num_events: int
    num_train_events: int
    num_train_pairs: int
    num_validation_candidates: int
    num_seen_targets_dropped: int
    num_insufficient_history_dropped: int
    num_unknown_item_dropped: int
    num_validation_targets: int
    insufficient_history_users: np.ndarray
    users_without_validation: np.ndarray
    num_new_items: int
    min_history: int

    def summary(self) -> str:
        """Returns a human readable summary of the split."""
        lines = [
            f"time based split at cutoff {self.cutoff!r}",
            f"  users={self.num_users} items={self.num_items} events={self.num_events}",
            f"  train: {self.num_train_events} events, {self.num_train_pairs} user/item pairs",
            (
                f"  validation: {self.num_validation_targets} targets from "
                f"{self.num_validation_candidates} post-cutoff candidates"
            ),
            f"  dropped {self.num_seen_targets_dropped} already seen targets",
            f"  dropped {self.num_unknown_item_dropped} targets for items with no training history",
            (
                f"  dropped {self.num_insufficient_history_dropped} targets from "
                f"{len(self.insufficient_history_users)} accounts with < {self.min_history} "
                "training items"
            ),
            f"  {self.num_new_items} items have no training interactions",
        ]
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        """Returns the report as a JSON-serialisable dictionary."""
        return {
            "cutoff": _json_value(self.cutoff),
            "num_users": int(self.num_users),
            "num_items": int(self.num_items),
            "num_events": int(self.num_events),
            "num_train_events": int(self.num_train_events),
            "num_train_pairs": int(self.num_train_pairs),
            "num_validation_candidates": int(self.num_validation_candidates),
            "num_seen_targets_dropped": int(self.num_seen_targets_dropped),
            "num_insufficient_history_dropped": int(self.num_insufficient_history_dropped),
            "num_unknown_item_dropped": int(self.num_unknown_item_dropped),
            "num_validation_targets": int(self.num_validation_targets),
            "insufficient_history_users": [_json_value(x) for x in self.insufficient_history_users],
            "users_without_validation": [_json_value(x) for x in self.users_without_validation],
            "num_new_items": int(self.num_new_items),
            "min_history": int(self.min_history),
        }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _optional_spec(name, available, default):
    """Resolves an optional column spec.

    ``name`` may be a column name, an array-like of values, or ``None``.  When a
    column is missing the default name falls back to ``None`` (meaning "no such
    column"), but an explicitly requested missing column is an error.
    """
    if name is None:
        return None
    if not isinstance(name, str):
        # caller supplied the values directly, no column lookup required
        return name
    if name in available:
        return name
    if name == default:
        return None
    raise ValueError(f"events is missing requested column {name!r}")


def _column_getter(events, user, item, timestamp, value, positive):
    """Returns ``(get, num_events, value_spec, positive_spec)`` for the log."""
    required = (user, item, timestamp)

    if isinstance(events, dict):
        available = set(events)
        missing = [c for c in required if c not in available]
        if missing:
            raise ValueError(f"events dictionary is missing required columns: {missing}")
        num_events = len(np.asarray(events[user]))
        value_spec = _optional_spec(value, available, "value")
        positive_spec = _optional_spec(positive, available, "positive")
        return (lambda name: np.asarray(events[name])), num_events, value_spec, positive_spec

    if hasattr(events, "columns") and hasattr(events, "__getitem__"):
        available = set(events.columns)
        missing = [c for c in required if c not in available]
        if missing:
            raise ValueError(f"events is missing required columns: {missing}")
        value_spec = _optional_spec(value, available, "value")
        positive_spec = _optional_spec(positive, available, "positive")
        return (lambda name: np.asarray(events[name])), len(events), value_spec, positive_spec

    array = np.asarray(events)
    if array.dtype.names:
        available = set(array.dtype.names)
        missing = [c for c in required if c not in available]
        if missing:
            raise ValueError(f"events is missing required fields: {missing}")
        value_spec = _optional_spec(value, available, "value")
        positive_spec = _optional_spec(positive, available, "positive")
        return (lambda name: np.asarray(array[name])), len(array), value_spec, positive_spec

    if array.ndim == 2:
        # positional layout: (user, item, timestamp[, value[, positive]])
        ncols = array.shape[1]
        if ncols < 3:
            raise ValueError(
                f"2d events array needs at least 3 columns (user, item, timestamp), got {ncols}"
            )
        mapping = {user: array[:, 0], item: array[:, 1], timestamp: array[:, 2]}
        value_spec = value if (value is not None and not isinstance(value, str)) else None
        positive_spec = (
            positive if (positive is not None and not isinstance(positive, str)) else None
        )
        if ncols >= 4:
            if value_spec is None:
                value_spec = value or "value"
                mapping[value_spec] = array[:, 3]
        elif value is not None:
            raise ValueError(
                "events has 3 columns but a value column was requested; pass a 4th column"
            )
        if ncols >= 5:
            if positive_spec is None:
                positive_spec = positive or "positive"
                mapping[positive_spec] = array[:, 4]
        elif positive is not None:
            raise ValueError(
                "events has fewer than 5 columns but a positive column was requested; "
                "pass a 5th column"
            )
        return (lambda name: mapping[name]), array.shape[0], value_spec, positive_spec

    raise TypeError(
        "events must be a pandas DataFrame, a dict of columns, a structured numpy "
        "array, or a 2d numpy array of (user, item, timestamp, ...)"
    )


def _resolve(spec, get, length, default=None):
    if spec is None:
        return default
    if isinstance(spec, str):
        values = get(spec)
    else:
        values = np.asarray(spec)
    if len(values) != length:
        raise ValueError(f"column of length {len(values)} does not match {length} events")
    return values


def _coerce_timestamps(timestamps, cutoff):
    """Normalises timestamps/cutoff so they can be compared with <= and >."""
    arr = np.asarray(timestamps)

    if arr.dtype.kind == "M":
        if not isinstance(cutoff, np.datetime64):
            cutoff = np.datetime64(cutoff)
        return arr, cutoff

    # strings (or objects holding dates): parse both sides to datetime64
    if arr.dtype.kind in ("U", "S") or (
        arr.dtype.kind == "O" and isinstance(cutoff, (datetime, date, np.datetime64))
    ):
        if not isinstance(cutoff, np.datetime64):
            cutoff = np.datetime64(cutoff)
        try:
            arr = arr.astype("datetime64[ns]")
        except (TypeError, ValueError):
            arr = np.array([np.datetime64(x) for x in arr], dtype="datetime64[ns]")
        return arr, cutoff

    if isinstance(cutoff, (datetime, date)) and arr.dtype.kind != "M":
        # numeric timestamps with a datetime cutoff: assume unix seconds
        arr = arr.astype("int64")
        cutoff = int(np.datetime64(cutoff).astype("datetime64[s]").astype("int64"))
        return arr, cutoff

    return arr, cutoff


def _sorted_unique(values):
    """Returns the sorted unique values, tolerating ids that can't be ordered."""
    try:
        return np.unique(values)
    except TypeError:
        seen = {}
        for value in values.tolist():
            seen.setdefault(value, None)
        # mixed types can't be sorted deterministically; preserve first-seen order
        return np.array(list(seen.keys()), dtype=object)


class _IdAxis:
    """Maps arbitrary account/item ids onto a fixed, contiguous integer axis."""

    def __init__(self, raw_values, provided, name):
        if provided is None:
            ids = _sorted_unique(np.asarray(raw_values))
        else:
            ids = np.asarray(provided)
            if ids.ndim != 1:
                raise ValueError(f"{name}_ids must be one dimensional")
            if len(_sorted_unique(ids)) != len(ids):
                raise ValueError(f"{name}_ids contains duplicate ids")

        if len(ids) == 0:
            raise ValueError(f"{name} axis is empty")

        self.ids = np.asarray(ids)
        self.name = name
        self._search = None
        self._restore = None
        try:
            order = np.argsort(self.ids, kind="stable")
        except TypeError:
            order = None
        if order is not None:
            self._search = self.ids[order]
            # position in self.ids of each position in the sorted array
            self._restore = order

    def encode(self, values):
        values = np.asarray(values)
        if self._search is not None:
            pos = np.searchsorted(self._search, values)
            clipped = np.minimum(pos, len(self._search) - 1)
            valid = pos < len(self._search)
            valid[valid] = self._search[clipped[valid]] == values[valid]
            encoded = self._restore[clipped]
        else:  # mixed/unorderable object ids: fall back to a dict lookup
            lookup = {value: i for i, value in enumerate(self.ids)}
            encoded = np.empty(len(values), dtype=np.int64)
            valid = np.zeros(len(values), dtype=bool)
            for i, value in enumerate(values):
                try:
                    encoded[i] = lookup[value]
                    valid[i] = True
                except (KeyError, TypeError):
                    pass

        if not valid.all():
            missing = np.asarray(values)[~valid]
            raise ValueError(
                f"events contain {len(missing)} {self.name} id(s) that are not present in "
                f"{self.name}_ids, for example: {missing[:5].tolist()}"
            )
        return encoded.astype(np.int64, copy=False)

    def decode(self, indices):
        return self.ids[np.asarray(indices)]


def _pair_keys(user_idx, item_idx, num_items):
    return user_idx.astype(np.int64) * np.int64(num_items) + item_idx.astype(np.int64)


def _contains(sorted_keys, query_keys):
    if len(query_keys) == 0:
        return np.zeros(0, dtype=bool)
    if len(sorted_keys) == 0:
        return np.zeros(len(query_keys), dtype=bool)
    pos = np.searchsorted(sorted_keys, query_keys)
    clipped = np.minimum(pos, len(sorted_keys) - 1)
    found = pos < len(sorted_keys)
    found[found] = sorted_keys[clipped[found]] == query_keys[found]
    return found


def _to_float(values):
    return np.asarray(values, dtype=np.float64)


def time_based_split(
    events,
    cutoff,
    user_ids=None,
    item_ids=None,
    *,
    user: str = "user",
    item: str = "item",
    timestamp: str = "timestamp",
    value: Optional[str] = "value",
    positive: Optional[str] = "positive",
    min_history: int = 1,
    return_report: bool = False,
):
    """Splits a timestamped event log into a training and a validation matrix.

    Unlike :func:`implicit.evaluation.train_test_split` this is a *temporal*
    split: every training event happens at or before a single global ``cutoff``
    and every validation target happens strictly after it.  A target is only
    kept when it is positive feedback for an item the user had not interacted
    with before the cutoff.

    Parameters
    ----------
    events : pandas.DataFrame, dict, structured ndarray or 2d ndarray
        The event log.  ``user``/``item``/``timestamp`` select the columns
        holding the account, item and event time.  A 2d ndarray is interpreted
        positionally as ``(user, item, timestamp[, value[, positive]])``.
    cutoff : object
        Global split point.  Events with ``timestamp <= cutoff`` are used for
        training; events strictly after it are validation candidates.
    user_ids, item_ids : array_like, optional
        Fixed axes to use for the output matrices.  When omitted the axes are
        the sorted unique ids found in the full event log (so accounts and
        items that appear only after the cutoff are still represented).  When
        supplied, every event id must be present and ordering is preserved.
    user, item, timestamp : str, optional
        Column names.
    value : str or array_like or None, optional
        Positive interaction weight stored in the matrices.  ``None`` means a
        weight of 1 for every event.
    positive : str or array_like or None, optional
        Boolean mask (or the name of a column) marking eligible positive
        feedback.  Non-positive events are excluded from both matrices.
        ``None`` treats every event as positive.
    min_history : int, optional
        Minimum number of distinct items an account needs in the training
        matrix to be eligible for evaluation.  Accounts below this threshold
        (including accounts with no history at the cutoff) are reported and
        their validation targets are dropped.
    return_report : bool, optional
        When True, return ``(train, validation, report)`` instead of
        ``(train, validation)``.

    Returns
    -------
    train : csr_matrix
        Binary/weighted user x item matrix built from events at or before the
        cutoff.
    validation : csr_matrix
        User x item matrix with the same shape holding post-cutoff positive
        targets.  A target is kept only when the user had at least
        ``min_history`` items at the cutoff, the ``(user, item)`` pair was
        unseen at the cutoff, and the item itself had at least one positive
        training interaction (otherwise the model could never recommend it).
    report : TimeBasedSplitReport
        Only returned when ``return_report`` is True.
    """
    if min_history < 0:
        raise ValueError("min_history must be >= 0")

    get, num_events, value_spec, positive_spec = _column_getter(
        events, user, item, timestamp, value, positive
    )
    if num_events == 0:
        raise ValueError("events is empty")

    raw_users = get(user)
    raw_items = get(item)
    if len(raw_users) != num_events or len(raw_items) != num_events:
        raise ValueError("user/item/timestamp columns must all have the same length")

    timestamps, cutoff = _coerce_timestamps(get(timestamp), cutoff)

    weights = _resolve(value_spec, get, num_events, default=np.ones(num_events))
    positives = _resolve(positive_spec, get, num_events, default=np.ones(num_events, dtype=bool))
    positives = np.asarray(positives, dtype=bool)
    weights = _to_float(weights)

    user_axis = _IdAxis(raw_users, user_ids, "user")
    item_axis = _IdAxis(raw_items, item_ids, "item")
    num_users, num_items = len(user_axis.ids), len(item_axis.ids)

    user_idx = user_axis.encode(raw_users)
    item_idx = item_axis.encode(raw_items)

    in_train = np.asarray(timestamps <= cutoff) & positives
    after = np.asarray(timestamps > cutoff) & positives

    # --- training matrix -------------------------------------------------
    train_users = user_idx[in_train]
    train_items = item_idx[in_train]
    train_values = weights[in_train]
    train_pairs = _pair_keys(train_users, train_items, num_items)
    unique_train_pairs = np.unique(train_pairs)
    train_mat = coo_matrix(
        (train_values, (train_users, train_items)),
        shape=(num_users, num_items),
        dtype=np.float64,
    ).tocsr()

    # number of distinct items each account/item has in its training history
    pair_users = (unique_train_pairs // np.int64(num_items)).astype(np.int64)
    pair_items = (unique_train_pairs % np.int64(num_items)).astype(np.int64)
    history = np.bincount(pair_users, minlength=num_users)
    item_history = np.bincount(pair_items, minlength=num_items)

    # --- validation matrix -----------------------------------------------
    cand_users = user_idx[after]
    cand_items = item_idx[after]
    cand_values = weights[after]
    num_candidates = len(cand_users)
    cand_pairs = _pair_keys(cand_users, cand_items, num_items)

    # drop targets whose (user, item) pair was already seen at the cutoff
    seen = _contains(unique_train_pairs, cand_pairs)
    novelty_keep = ~seen

    # drop targets for accounts without enough history to be evaluated
    history_keep = history[cand_users] >= min_history

    # drop targets for items that had no positive interaction before the cutoff:
    # the model has no way of surfacing these, so they aren't a fair target.
    item_keep = item_history[cand_items] > 0

    keep = novelty_keep & history_keep & item_keep
    final_users = cand_users[keep]
    final_items = cand_items[keep]
    final_values = cand_values[keep]

    val_mat = coo_matrix(
        (final_values, (final_users, final_items)),
        shape=(num_users, num_items),
        dtype=np.float64,
    ).tocsr()

    insufficient_users = user_axis.decode(np.flatnonzero(history < min_history))
    unique_final_pairs = _pair_keys(final_users, final_items, num_items)
    target_users = np.unique(unique_final_pairs // np.int64(num_items)) if len(final_users) else []
    users_without_validation = user_axis.decode(
        np.setdiff1d(np.arange(num_users), np.asarray(target_users, dtype=np.int64))
    )
    new_items = np.setdiff1d(np.arange(num_items), item_idx[in_train])

    # Report the drops in a fixed order so the counts are disjoint: seen pairs
    # first, then unknown items, then the remaining targets whose account is
    # below min_history.
    seen_dropped = int(seen.sum())
    unknown_item_dropped = int(np.sum(novelty_keep & ~item_keep))
    insufficient_dropped = int(np.sum(novelty_keep & item_keep & ~history_keep))

    report = TimeBasedSplitReport(
        cutoff=_json_value(cutoff),
        num_users=num_users,
        num_items=num_items,
        num_events=num_events,
        num_train_events=int(in_train.sum()),
        num_train_pairs=int(len(unique_train_pairs)),
        num_validation_candidates=int(num_candidates),
        num_seen_targets_dropped=seen_dropped,
        num_insufficient_history_dropped=insufficient_dropped,
        num_unknown_item_dropped=unknown_item_dropped,
        num_validation_targets=int(val_mat.nnz),
        insufficient_history_users=insufficient_users,
        users_without_validation=users_without_validation,
        num_new_items=int(len(new_items)),
        min_history=int(min_history),
    )

    if len(insufficient_users):
        level = logging.WARNING if report.num_insufficient_history_dropped else logging.INFO
        log.log(
            level,
            "time_based_split: %d account(s) have fewer than %d training item(s); "
            "%d post-cutoff target(s) were dropped from those accounts. Examples: %s",
            len(insufficient_users),
            min_history,
            report.num_insufficient_history_dropped,
            insufficient_users[:10].tolist(),
        )
    log.debug(
        "time_based_split: %d train events, %d validation targets, cutoff=%r",
        report.num_train_events,
        report.num_validation_targets,
        report.cutoff,
    )

    if return_report:
        return train_mat, val_mat, report
    return train_mat, val_mat


def evaluate_time_based_split(
    model: Any,
    train: csr_matrix,
    validation: csr_matrix,
    K: int = 10,
    show_progress: bool = True,
    num_threads: int = 1,
):
    """Evaluates a fitted model on a :func:`time_based_split` validation set.

    This is a thin wrapper around :func:`implicit.evaluation.ranking_metrics_at_k`
    so that a temporal split can be plugged into the regular evaluation calls
    without any further reshaping.  ``model`` must already be fitted on
    ``train``; the validation matrix is used as the held out target set.

    Parameters
    ----------
    model : RecommenderBase
        A fitted recommendation model.
    train : csr_matrix
        The training matrix returned by :func:`time_based_split`.  It is used
        both to describe each user's known history and to filter already liked
        items.
    validation : csr_matrix
        The validation matrix returned by :func:`time_based_split`.
    K : int, optional
        Number of items to test on.
    show_progress : bool, optional
        Whether to show a progress bar.
    num_threads : int, optional
        Number of threads passed to the evaluation code.

    Returns
    -------
    dict[str, float]
        ``precision``, ``map``, ``ndcg`` and ``auc`` at ``K``.
    """
    if train.shape != validation.shape:
        raise ValueError(
            f"train and validation must share the fixed axis, got {train.shape} and "
            f"{validation.shape}"
        )
    if validation.nnz == 0:
        raise ValueError(
            "validation has no targets; lower min_history, move the cutoff, or pass a "
            "longer event log before evaluating"
        )

    # imported lazily so the splitting helpers can be used without the compiled
    # evaluation extension being importable first.
    from .evaluation import ranking_metrics_at_k

    return ranking_metrics_at_k(
        model,
        train,
        validation,
        K=K,
        show_progress=show_progress,
        num_threads=num_threads,
    )
