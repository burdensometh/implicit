"""Deterministic conversion of raw interaction events into an implicit feedback matrix.

Recommendation models such as :class:`implicit.als.AlternatingLeastSquares` are trained on a
sparse ``(users, items)`` confidence matrix. Real event streams are not in that shape: the same
event can be delivered more than once (for example when a receiver replays a batch), different
event types carry very different intent, and events are worth less as they get older relative to
the point in time the model is trained for.

This module implements a reproducible *event -> feedback matrix* path:

1. Events are deduplicated by **event identity** (the ``event_id``), regardless of the order the
   events are supplied in. Duplicates whose payloads are identical are counted as replays and
   collapsed. Duplicates whose payloads disagree are *conflicts* and resolved by an explicit,
   deterministic policy.
2. Each surviving event is converted into a signed confidence value from its **behaviour type**
   (``event_weights``) and its **age relative to the training cutoff** (per-type or global
   ``half_life`` decay). Events newer than the cutoff are handled explicitly instead of leaking
   future information into training.
3. Confidences for the same ``(user, item)`` pair are combined with an explicit **cancellation
   policy**, which defines what happens when a real purchase and a rejection signal disagree.
4. The result is a canonical CSR matrix, with ids remapped to contiguous integer indices, that can
   be passed straight to :meth:`implicit.als.AlternatingLeastSquares.fit`.

The generated matrix is deterministic: two runs over the same event *set* produce the same matrix
and the same statistics even if the input order differs. Sums use :func:`math.fsum`, events are
sorted in a canonical order before aggregation, and ids that are not integers are mapped to indices
by sorting their values (never by hashing or insertion order).

Example
-------
>>> from datetime import datetime, timedelta
>>> from implicit.feedback import FeedbackConfig, build_feedback_matrix
>>> cutoff = datetime(2024, 1, 8)
>>> events = [
...     {"user": "u1", "item": "i1", "type": "view", "event_id": "e1",
...      "timestamp": cutoff - timedelta(days=3)},
...     {"user": "u1", "item": "i1", "type": "purchase", "event_id": "e2",
...      "timestamp": cutoff - timedelta(hours=1)},
...     # receiver replay of e1 - collapsed
...     {"user": "u1", "item": "i1", "type": "view", "event_id": "e1",
...      "timestamp": cutoff - timedelta(days=3)},
... ]
>>> config = FeedbackConfig(
...     training_cutoff=cutoff,
...     event_weights={"view": 1.0, "purchase": 5.0, "reject": -4.0},
...     half_life={"view": timedelta(days=2), "purchase": timedelta(days=30),
...                "reject": timedelta(days=30)},
... )
>>> feedback = build_feedback_matrix(events, config)
>>> feedback.matrix.shape, feedback.stats.duplicates_removed
((1, 1), 1)

The returned :class:`FeedbackMatrix` keeps ``user_ids``/``item_ids`` and lookup dictionaries, so
recommendations can be mapped back to the original ids.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

__all__ = [
    "FeedbackConfig",
    "FeedbackMatrix",
    "FeedbackStats",
    "build_feedback_matrix",
    "fit_feedback",
]


_EPOCH_NAIVE = datetime(1970, 1, 1)

_IDENTITY_FIELDS = ("user", "item", "type", "event_id", "timestamp")
_FUTURE_POLICIES = ("drop", "clamp")
_CONFLICT_POLICIES = ("latest", "earliest")
_CANCELLATION_POLICIES = ("net", "latest_wins", "rejection_wins")
_UNKNOWN_POLICIES = ("error", "ignore")
_IDENTITY_POLICIES = ("event_id", "auto")


@dataclass
class FeedbackStats:
    """Counts collected while building a feedback matrix.

    Attributes
    ----------
    events_received : int
        Number of raw event records supplied.
    events_after_dedup : int
        Number of distinct event identities kept after replay/conflict resolution.
    duplicates_removed : int
        Records dropped because their event identity was already seen.
    conflicts : int
        Number of event identities whose duplicate records disagreed on their payload.
    unknown_type_dropped : int
        Events ignored because their type had no configured weight.
    future_dropped : int
        Events ignored because they happened after the training cutoff.
    too_old_dropped : int
        Events ignored because they were older than ``max_age``.
    weak_dropped : int
        User/item pairs dropped because their combined confidence was zero or below
        ``min_confidence``.
    positive_pairs : int
        User/item pairs kept with a positive confidence.
    negative_pairs : int
        User/item pairs kept with a negative confidence (explicit rejection).
    """

    events_received: int = 0
    events_after_dedup: int = 0
    duplicates_removed: int = 0
    conflicts: int = 0
    unknown_type_dropped: int = 0
    future_dropped: int = 0
    too_old_dropped: int = 0
    weak_dropped: int = 0
    positive_pairs: int = 0
    negative_pairs: int = 0


@dataclass(frozen=True)
class FeedbackMatrix:
    """A canonical user/item confidence matrix built from events.

    Attributes
    ----------
    matrix : csr_matrix
        Sparse ``(len(user_ids), len(item_ids))`` matrix of signed confidences. Positive values
        are likes, negative values are explicit dislikes (both are understood by
        :class:`implicit.als.AlternatingLeastSquares`).
    user_ids : ndarray
        Original user ids, ordered by matrix row.
    item_ids : ndarray
        Original item ids, ordered by matrix column.
    user_index : Mapping
        Mapping from original user id to matrix row.
    item_index : Mapping
        Mapping from original item id to matrix column.
    stats : FeedbackStats
        Counts describing how the events were reduced.
    """

    matrix: csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_index: Mapping[Any, int]
    item_index: Mapping[Any, int]
    stats: FeedbackStats

    @property
    def shape(self) -> Tuple[int, int]:
        return self.matrix.shape

    def user_row(self, user_id: Any) -> csr_matrix:
        """Returns the single-row confidence matrix for one original user id."""
        return self.matrix[self.user_index[user_id]]

    def fit(self, model: Any, show_progress: bool = True) -> Any:
        """Fits a model (such as ALS) on this feedback matrix and returns it.

        The model's factors are indexed by the row/column positions of ``matrix``; use
        ``user_index``/``item_index`` to translate original ids.
        """
        model.fit(self.matrix, show_progress=show_progress)
        return model


@dataclass(frozen=True)
class FeedbackConfig:
    """Configuration for :func:`build_feedback_matrix`.

    Parameters
    ----------
    training_cutoff : float or datetime.datetime
        Point in time the model is trained for. Only events at or before this instant are used by
        default, and event age is measured as ``training_cutoff - timestamp``. When a
        ``datetime`` is supplied, event timestamps must be ``datetime`` objects too; when a number
        is supplied, timestamps must be numbers in the same unit.
    event_weights : Mapping
        Mapping of behaviour type to weight. A positive weight is positive feedback, a negative
        weight is a rejection signal, and the magnitude is the confidence of a single event at age
        zero. Types not present here are rejected (or ignored, see ``unknown_event_policy``).
    half_life : float, datetime.timedelta or Mapping, optional
        Exponential decay half-life. ``None`` disables decay. A single value applies to every
        type; a mapping must provide a half-life for every type in ``event_weights`` and lets
        different behaviours age differently (for example old browsing may decay much faster than
        a purchase). The unit is the unit of the timestamps: with a ``datetime`` cutoff, pass a
        ``timedelta`` (or a number of seconds); with a numeric cutoff, pass a number in the same
        unit.
    max_age : float or datetime.timedelta, optional
        If set, events older than this are discarded after deduplication. Uses the same unit
        rules as ``half_life``.
    future_policy : {'drop', 'clamp'}, optional
        How to treat events with a timestamp after ``training_cutoff``. ``'drop'`` (default)
        discards them, which prevents future leakage. ``'clamp'`` moves them to the cutoff and
        gives them full confidence.
    conflict_policy : {'latest', 'earliest'}, optional
        How to choose between duplicate records that share an event identity but disagree on their
        payload. ``'latest'`` (default) keeps the record with the greatest timestamp (ties broken
        by a canonical key), ``'earliest'`` keeps the smallest.
    cancellation_policy : {'net', 'latest_wins', 'rejection_wins'}, optional
        How positive and negative signals for the same ``(user, item)`` pair combine:

        * ``'net'`` (default): signed sum; a rejection offsets a purchase. A negative total is
          kept as a dislike, and a zero total is dropped.
        * ``'latest_wins'``: the most recent event decides the sign; the kept confidence is the
          summed magnitude of the events that agree with that sign.
        * ``'rejection_wins'``: any rejection makes the pair negative (magnitude is the summed
          rejection), modelling a purchase that was later returned.

    min_confidence : float, optional
        Pairs whose combined absolute confidence is below this value are discarded. The default
        of ``0.0`` only discards exact zeros (including exact purchase/rejection cancellation).
    unknown_event_policy : {'error', 'ignore'}, optional
        Whether an event type missing from ``event_weights`` raises (default) or is skipped.
    event_identity : {'event_id', 'auto'}, optional
        ``'event_id'`` (default) requires every event to carry an event id and deduplicates on it.
        ``'auto'`` falls back to ``(user, item, type, timestamp)`` when the event id is missing,
        which also collapses repeated identical events that carry no id.
    dtype : numpy dtype, optional
        dtype of the generated matrix. Defaults to ``numpy.float32``, matching ALS.
    """

    training_cutoff: Union[float, datetime]
    event_weights: Mapping[Any, float]
    half_life: Optional[Union[float, Mapping[Any, float]]] = None
    max_age: Optional[float] = None
    future_policy: str = "drop"
    conflict_policy: str = "latest"
    cancellation_policy: str = "net"
    min_confidence: float = 0.0
    unknown_event_policy: str = "error"
    event_identity: str = "event_id"
    dtype: Any = np.float32

    def __post_init__(self) -> None:
        if not isinstance(self.training_cutoff, (int, float, np.integer, np.floating, datetime)):
            raise TypeError(
                "training_cutoff must be a number or a datetime, got "
                f"{type(self.training_cutoff).__name__}"
            )
        if isinstance(self.training_cutoff, (int, float, np.integer, np.floating)) and not (
            math.isfinite(float(self.training_cutoff))
        ):
            raise ValueError("training_cutoff must be finite")

        if not isinstance(self.event_weights, Mapping) or not self.event_weights:
            raise ValueError("event_weights must be a non-empty mapping of type -> weight")
        weights: Dict[Any, float] = {}
        for event_type, weight in self.event_weights.items():
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                raise TypeError(f"weight for event type {event_type!r} must be numeric") from None
            if not math.isfinite(weight):
                raise ValueError(f"weight for event type {event_type!r} must be finite")
            weights[event_type] = weight
        if not any(weight > 0 for weight in weights.values()):
            raise ValueError("event_weights must contain at least one positive weight")

        cutoff_is_datetime = isinstance(self.training_cutoff, datetime)
        half_life = self.half_life
        if half_life is not None:
            if isinstance(half_life, Mapping):
                missing = [t for t in weights if t not in half_life]
                if missing:
                    raise ValueError(
                        "half_life mapping is missing entries for event types: "
                        f"{[_safe_repr(t) for t in missing]}"
                    )
                extra = [t for t in half_life if t not in weights]
                if extra:
                    raise ValueError(
                        "half_life mapping contains types with no weight: "
                        f"{[_safe_repr(t) for t in extra]}"
                    )
                normalized = {}
                for event_type, value in half_life.items():
                    normalized[event_type] = _duration_to_units(
                        value, f"half_life[{_safe_repr(event_type)}]", cutoff_is_datetime
                    )
            else:
                normalized = _duration_to_units(half_life, "half_life", cutoff_is_datetime)
            object.__setattr__(self, "half_life", normalized)

        if self.max_age is not None:
            object.__setattr__(
                self,
                "max_age",
                _duration_to_units(self.max_age, "max_age", cutoff_is_datetime),
            )
        object.__setattr__(
            self,
            "min_confidence",
            _positive_float(self.min_confidence, "min_confidence", allow_none=True, zero_ok=True),
        )

        _check_choice(self.future_policy, _FUTURE_POLICIES, "future_policy")
        _check_choice(self.conflict_policy, _CONFLICT_POLICIES, "conflict_policy")
        _check_choice(self.cancellation_policy, _CANCELLATION_POLICIES, "cancellation_policy")
        _check_choice(self.unknown_event_policy, _UNKNOWN_POLICIES, "unknown_event_policy")
        _check_choice(self.event_identity, _IDENTITY_POLICIES, "event_identity")

        # normalize the mapping so later mutation of the caller's dict cannot change results
        object.__setattr__(self, "event_weights", weights)
        try:
            dtype = np.dtype(self.dtype)
        except TypeError:
            raise TypeError(f"invalid dtype {self.dtype!r}") from None
        if dtype.kind != "f":
            raise ValueError(f"dtype must be a floating point type, got {dtype}")
        object.__setattr__(self, "dtype", dtype)


def build_feedback_matrix(
    events: Any = None,
    config: Optional[FeedbackConfig] = None,
    *,
    users: Optional[Sequence[Any]] = None,
    items: Optional[Sequence[Any]] = None,
    event_types: Optional[Sequence[Any]] = None,
    event_ids: Optional[Sequence[Any]] = None,
    timestamps: Optional[Sequence[Any]] = None,
    user_ids: Optional[Sequence[Any]] = None,
    item_ids: Optional[Sequence[Any]] = None,
) -> FeedbackMatrix:
    """Builds a canonical implicit-feedback matrix from interaction events.

    Parameters
    ----------
    events : sequence of mappings, mapping of columns, or None
        Events as a sequence of mappings with keys ``user``, ``item``, ``type``, ``event_id`` and
        ``timestamp``, or as a mapping of those names to equal-length column sequences. Pass
        ``None`` to supply the same data through the keyword column arguments instead.
    config : FeedbackConfig
        Configuration controlling deduplication, decay and cancellation.
    users, items, event_types, event_ids, timestamps : sequence, optional
        Column-oriented alternative to ``events``.
    user_ids, item_ids : sequence, optional
        Explicit vocabularies for the matrix rows/columns. When omitted, the sorted unique ids
        seen in the events are used. Supplying them keeps indices stable across batches, and an
        event whose id is not present raises ``KeyError``.

    Returns
    -------
    FeedbackMatrix
        The matrix, original id vocabularies and reduction statistics.

    Notes
    -----
    The pipeline runs in a fixed order: normalise -> deduplicate by event identity -> resolve
    conflicts -> drop future/old/unknown events -> apply behaviour weight and time decay ->
    combine per ``(user, item)`` pair by the cancellation policy -> threshold -> build CSR.
    """
    if config is None:
        raise TypeError("config is required (FeedbackConfig)")
    if not isinstance(config, FeedbackConfig):
        raise TypeError(f"config must be a FeedbackConfig, got {type(config).__name__}")

    records = _read_records(
        events, users, items, event_types, event_ids, timestamps, config
    )
    if not records:
        raise ValueError("no events supplied")

    stats = FeedbackStats(events_received=len(records))

    # ---- 1. deduplicate by event identity, independently of input order ----------------
    groups: Dict[Any, List[_Record]] = {}
    for record in records:
        groups.setdefault(record.identity, []).append(record)

    deduped: List[_Record] = []
    for identity in sorted(groups, key=_identity_sort_key):
        group = groups[identity]
        if len(group) > 1:
            stats.duplicates_removed += len(group) - 1
            if len({record.payload for record in group}) > 1:
                stats.conflicts += 1
        deduped.append(_resolve_conflict(group, config.conflict_policy))

    stats.events_after_dedup = len(deduped)

    # canonical order makes the float sums order independent
    deduped.sort(key=_record_sort_key)

    # ---- 2. apply age/type weighting, group per pair ----------------------------------
    cutoff = _to_units(config.training_cutoff)

    contributions: Dict[Tuple[Any, Any], List[_Contribution]] = {}
    for record in deduped:
        weight = config.event_weights.get(record.event_type)
        if weight is None:
            if config.unknown_event_policy == "error":
                raise ValueError(
                    f"event {_safe_repr(record.event_id)} has unknown type "
                    f"{_safe_repr(record.event_type)}; either add it to event_weights or set "
                    "unknown_event_policy='ignore'"
                )
            stats.unknown_type_dropped += 1
            continue

        timestamp = record.timestamp
        if timestamp > cutoff:
            if config.future_policy == "drop":
                stats.future_dropped += 1
                continue
            timestamp = cutoff
        age = cutoff - timestamp
        if config.max_age is not None and age > config.max_age:
            stats.too_old_dropped += 1
            continue

        decay = _decay(age, config.half_life, record.event_type)
        contributions.setdefault((record.user, record.item), []).append(
            _Contribution(
                timestamp=timestamp,
                event_type=record.event_type,
                event_id=record.event_id,
                contribution=weight * decay,
                weight=weight,
            )
        )

    # ---- 3. combine positive/negative evidence with the cancellation policy ----------
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    kept_pairs: List[Tuple[Any, Any]] = []
    for pair in sorted(contributions, key=lambda p: (_id_sort_key(p[0]), _id_sort_key(p[1]))):
        value = _combine(contributions[pair], config.cancellation_policy)
        if value == 0.0 or abs(value) < config.min_confidence:
            stats.weak_dropped += 1
            continue
        kept_pairs.append(pair)
        data.append(value)
        if value > 0:
            stats.positive_pairs += 1
        else:
            stats.negative_pairs += 1

    # the vocabulary is built from the pairs that are actually kept, so dropped events do not
    # inflate the matrix dimensions
    user_index, user_vocab = _make_vocabulary(
        user_ids, [pair[0] for pair in kept_pairs], "user_ids"
    )
    item_index, item_vocab = _make_vocabulary(
        item_ids, [pair[1] for pair in kept_pairs], "item_ids"
    )
    for user, item in kept_pairs:
        rows.append(user_index[user])
        cols.append(item_index[item])

    matrix = csr_matrix(
        (np.asarray(data, dtype=config.dtype), (np.asarray(rows, dtype=np.int64),
                                               np.asarray(cols, dtype=np.int64))),
        shape=(len(user_vocab), len(item_vocab)),
        dtype=config.dtype,
    )
    matrix.sum_duplicates()
    matrix.sort_indices()

    return FeedbackMatrix(
        matrix=matrix,
        user_ids=user_vocab,
        item_ids=item_vocab,
        user_index=user_index,
        item_index=item_index,
        stats=stats,
    )


def fit_feedback(
    model: Any,
    events: Any = None,
    config: Optional[FeedbackConfig] = None,
    show_progress: bool = True,
    **columns: Any,
) -> FeedbackMatrix:
    """Builds a feedback matrix from events and fits ``model`` on it.

    ``model`` can be any :class:`implicit.RecommenderBase` model, including
    :class:`implicit.als.AlternatingLeastSquares`. The model is fitted with
    ``model.fit(feedback.matrix)``. The returned :class:`FeedbackMatrix` carries the id mappings
    needed to interpret the fitted factors and recommendations.

    Parameters
    ----------
    model : RecommenderBase
        The model to fit.
    events : sequence of mappings, mapping of columns, or FeedbackMatrix
        Raw events (see :func:`build_feedback_matrix`). A pre-built :class:`FeedbackMatrix` may
        be passed instead, in which case ``config`` is not required.
    config : FeedbackConfig, optional
        Required when ``events`` are raw events.
    show_progress : bool, optional
        Passed through to ``model.fit``.
    **columns
        Column-oriented event data, forwarded to :func:`build_feedback_matrix`.

    Returns
    -------
    FeedbackMatrix
        The matrix that was fitted.
    """
    if isinstance(events, FeedbackMatrix):
        feedback = events
    else:
        feedback = build_feedback_matrix(events, config, **columns)
    feedback.fit(model, show_progress=show_progress)
    return feedback


# ---------------------------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Record:
    identity: Any
    payload: Tuple[Any, Any, Any, float]
    user: Any
    item: Any
    event_type: Any
    event_id: Any
    timestamp: float


@dataclass(frozen=True)
class _Contribution:
    timestamp: float
    event_type: Any
    event_id: Any
    contribution: float
    weight: float


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:  # pragma: no cover - defensive against weird __repr__  # noqa: BLE001
        return f"<{type(value).__name__}>"


def _positive_float(
    value: Any, name: str, allow_none: bool, zero_ok: bool = False
) -> Optional[float]:
    if value is None and allow_none:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be numeric") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (value == 0 and not zero_ok):
        raise ValueError(f"{name} must be {'non-negative' if zero_ok else 'positive'}")
    return value


def _check_choice(value: str, allowed: Sequence[str], name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}, got {_safe_repr(value)}")


def _duration_to_units(value: Any, name: str, cutoff_is_datetime: bool) -> float:
    """Validates a positive time interval and expresses it in the cutoff's time unit.

    When the cutoff is a ``datetime`` ages are measured in seconds, so a ``timedelta`` is the
    natural way to express a half-life. Seconds as a number are accepted there too; when the
    cutoff is numeric, ``timedelta`` values are rejected since the unit is unknown.
    """
    if isinstance(value, timedelta):
        if not cutoff_is_datetime:
            raise TypeError(
                f"{name} is a timedelta but training_cutoff is numeric; pass the interval as a "
                "number in the same unit as the timestamps"
            )
        seconds = value.total_seconds()
        if seconds <= 0 or not math.isfinite(seconds):
            raise ValueError(f"{name} must be a positive, finite duration")
        return seconds
    return _positive_float(value, name, allow_none=False)


def _id_sort_key(value: Any) -> Tuple[int, float, str]:
    """A total, deterministic order over ids of mixed types."""
    if isinstance(value, (bool, np.bool_)):
        return (0, float(value), "")
    if isinstance(value, (int, np.integer)):
        return (1, float(value), "")
    if isinstance(value, (float, np.floating)):
        return (2, float(value), "")
    if isinstance(value, str):
        return (3, 0.0, value)
    return (4, 0.0, f"{type(value).__name__}:{_safe_repr(value)}")


def _identity_sort_key(identity: Any) -> Tuple[str, str]:
    return (type(identity).__name__, _safe_repr(identity))


def _type_key(value: Any) -> Tuple[str, str]:
    return (type(value).__name__, _safe_repr(value))


def _record_sort_key(record: _Record) -> Tuple[Any, ...]:
    return (
        _id_sort_key(record.user),
        _id_sort_key(record.item),
        record.timestamp,
        _type_key(record.event_type),
        _type_key(record.event_id),
    )


def _to_units(value: Any) -> float:
    """Converts a datetime to seconds since the epoch without relying on local timezone."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return (value - _EPOCH_NAIVE).total_seconds()
        return (value - datetime(1970, 1, 1, tzinfo=value.tzinfo)).total_seconds()
    return float(value)


def _read_records(
    events: Any,
    users: Any,
    items: Any,
    event_types: Any,
    event_ids: Any,
    timestamps: Any,
    config: FeedbackConfig,
) -> List[_Record]:
    columns = (users, items, event_types, event_ids, timestamps)
    supplied_columns = [column is not None for column in columns]
    if events is not None and any(supplied_columns):
        raise ValueError("pass either 'events' or the keyword column arguments, not both")
    if events is None and not all(supplied_columns):
        raise ValueError(
            "no events supplied: pass 'events' (sequence of mappings or mapping of columns) or "
            "all of users, items, event_types, event_ids, timestamps"
        )

    if events is not None:
        if isinstance(events, Mapping):
            missing = [key for key in _IDENTITY_FIELDS if key not in events]
            if missing:
                raise ValueError(f"columnar events are missing keys: {missing}")
            columns = (
                events["user"],
                events["item"],
                events["type"],
                events["event_id"],
                events["timestamp"],
            )
        else:
            records = [
                _normalize_record(record, config, index)
                for index, record in enumerate(events)
            ]
            return records

    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise ValueError(f"event columns must have equal lengths, got {sorted(lengths)}")
    return [
        _normalize_record(
            {
                "user": columns[0][index],
                "item": columns[1][index],
                "type": columns[2][index],
                "event_id": columns[3][index],
                "timestamp": columns[4][index],
            },
            config,
            index,
        )
        for index in range(lengths.pop())
    ]


def _normalize_record(record: Any, config: FeedbackConfig, index: int) -> _Record:
    if not isinstance(record, Mapping):
        raise TypeError(f"event {index} must be a mapping, got {type(record).__name__}")
    missing = [key for key in _IDENTITY_FIELDS if key not in record]
    if missing:
        raise ValueError(f"event {index} is missing fields {missing}")

    user = record["user"]
    item = record["item"]
    event_type = record["type"]
    event_id = record["event_id"]
    raw_timestamp = record["timestamp"]
    if user is None or item is None:
        raise ValueError(f"event {index} has a null user or item")

    timestamp = _normalize_timestamp(raw_timestamp, config, index)

    if event_id is None:
        if config.event_identity == "event_id":
            raise ValueError(
                f"event {index} is missing event_id; set event_identity='auto' to fall back to "
                "the (user, item, type, timestamp) tuple"
            )
        identity: Any = ("auto", user, item, event_type, timestamp)
    else:
        try:
            hash(event_id)
        except TypeError:
            raise TypeError(f"event {index} has an unhashable event_id") from None
        identity = ("event_id", event_id)

    return _Record(
        identity=identity,
        payload=(user, item, event_type, timestamp),
        user=user,
        item=item,
        event_type=event_type,
        event_id=event_id,
        timestamp=timestamp,
    )


def _normalize_timestamp(value: Any, config: FeedbackConfig, index: int) -> float:
    cutoff_is_datetime = isinstance(config.training_cutoff, datetime)
    if cutoff_is_datetime:
        if not isinstance(value, datetime):
            raise TypeError(
                f"event {index} timestamp must be a datetime because training_cutoff is one, "
                f"got {type(value).__name__}"
            )
        if (value.tzinfo is None) != (config.training_cutoff.tzinfo is None):
            raise TypeError(
                f"event {index} timestamp and training_cutoff must both be timezone-aware or "
                "both be naive"
            )
        return _to_units(value)
    if isinstance(value, datetime):
        raise TypeError(
            f"event {index} timestamp is a datetime but training_cutoff is numeric; use one "
            "representation for both"
        )
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"event {index} timestamp must be numeric, got bool")
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"event {index} timestamp must be numeric") from None
    if not math.isfinite(timestamp):
        raise ValueError(f"event {index} timestamp must be finite")
    return timestamp


def _resolve_conflict(group: List[_Record], policy: str) -> _Record:
    if len(group) == 1:
        return group[0]
    # canonical key breaks ties deterministically, so input order never matters
    key = lambda record: (record.timestamp, _type_key(record.event_type),
                          _type_key(record.event_id), _id_sort_key(record.user),
                          _id_sort_key(record.item))
    if policy == "latest":
        return max(group, key=key)
    return min(group, key=key)


def _decay(age: float, half_life: Any, event_type: Any) -> float:
    if half_life is None:
        return 1.0
    if isinstance(half_life, Mapping):
        half_life = half_life[event_type]
    # age is >= 0 here (future events were dropped or clamped to the cutoff)
    return 0.5 ** (age / half_life)


def _combine(contributions: List[_Contribution], policy: str) -> float:
    if policy == "net":
        return math.fsum(item.contribution for item in contributions)

    if policy == "rejection_wins":
        negative = [item for item in contributions if item.weight < 0]
        if negative:
            return -math.fsum(abs(item.contribution) for item in negative)
        return math.fsum(item.contribution for item in contributions if item.weight > 0)

    # latest_wins: the newest event picks the sign, events agreeing with it are summed
    latest = max(
        contributions,
        key=lambda item: (item.timestamp, _type_key(item.event_type), _type_key(item.event_id)),
    )
    if latest.weight == 0:
        return 0.0
    latest_positive = latest.weight > 0
    return math.fsum(
        abs(item.contribution)
        for item in contributions
        if item.weight != 0 and (item.weight > 0) == latest_positive
    ) * (1.0 if latest_positive else -1.0)


def _make_vocabulary(
    explicit: Optional[Sequence[Any]], values: Iterable[Any], name: str
) -> Tuple[Dict[Any, int], np.ndarray]:
    if explicit is not None:
        vocabulary = list(explicit)
        lookup: Dict[Any, int] = {}
        for index, value in enumerate(vocabulary):
            if value in lookup:
                raise ValueError(f"{name} contains duplicate id {_safe_repr(value)}")
            lookup[value] = index
        unknown = sorted({value for value in values if value not in lookup}, key=_id_sort_key)
        if unknown:
            raise KeyError(
                f"events reference {name} ids not present in the explicit vocabulary: "
                f"{[_safe_repr(value) for value in unknown]}"
            )
        return lookup, np.asarray(vocabulary)

    vocabulary = sorted(set(values), key=_id_sort_key)
    lookup = {value: index for index, value in enumerate(vocabulary)}
    return lookup, np.asarray(vocabulary)
