"""Build session-based co-occurrence inputs for the item-item similarity models.

Item-item similarity models in :mod:`implicit.nearest_neighbours` treat every row of their
input matrix as one "basket" of items consumed together.  When the raw data is a log of
browsing events rather than a user-item matrix, feeding all of a member's events in as one row
conflates months of unrelated visits: two products browsed weeks apart are counted as
co-occurring, and the similarity model will happily recommend one on the page of the other.

The helpers here turn timestamped events into *sessions* first:

* events for each member are ordered by time,
* a new session starts whenever the member has been idle for longer than ``idle_timeout``,
* repeated events for the same item inside one session are merged into a single entry,

and then build the sparse ``(sessions x items)`` matrix consumed by the existing ``Cosine``,
``TF-IDF`` and ``BM25`` recommenders.  Because each row is one session (not one member), items
that only appear in different sessions of the same member never share a row and therefore never
become similar merely because the same member looked at both.

``ItemItemRecommender.fit_sessions`` wires these helpers into the normal model API::

    model = implicit.nearest_neighbours.TFIDFRecommender()
    model.fit_sessions(events, idle_timeout=30 * 60)
    ids, scores = model.similar_items(itemid)

Events can be supplied as an iterable of ``(user, item, timestamp)`` triples, a 2D numpy array
with three columns, or a pandas ``DataFrame`` (pass ``columns=("user", "item", "timestamp")``
to choose the relevant columns explicitly).  Timestamps may be numbers or anything that can be
converted to seconds (``datetime`` objects, ``numpy.datetime64`` values, ``pandas.Timestamp``).
"""

from __future__ import annotations

import datetime as _datetime
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

#: Default idle time, in seconds, used to split sessions (30 minutes).
DEFAULT_IDLE_TIMEOUT = 30 * 60

_EPOCH = _datetime.datetime(1970, 1, 1)

#: Common column names used when detecting the relevant columns of a DataFrame.
_USER_COLUMNS = ("user", "user_id", "userid", "member", "member_id", "customer", "customer_id")
_ITEM_COLUMNS = (
    "item",
    "item_id",
    "itemid",
    "product",
    "product_id",
    "article",
    "article_id",
    "artist",
    "artist_id",
    "movie",
    "movie_id",
)
_TIMESTAMP_COLUMNS = (
    "timestamp",
    "time",
    "ts",
    "event_time",
    "event_timestamp",
    "datetime",
    "date",
    "created_at",
)


class Event(NamedTuple):
    """A single timestamped interaction between a user and an item."""

    user: object
    item: object
    timestamp: object


class Session(list):
    """One session of a user's activity.

    A :class:`Session` *is* a list of item ids (duplicates merged, in the order items were first
    viewed), so it compares equal to a plain ``list``/``tuple`` of those items::

        split_sessions(events, idle_timeout=1800) == [["headphones", "laptop"], ["blender"]]

    The originating user and the raw :class:`Event` objects are also available as attributes
    ``user`` and ``events`` respectively.
    """

    def __init__(self, user, events=(), items=None):
        self.user = user
        self.events = tuple(events)
        if items is None:
            items = _dedupe_in_order(event.item for event in self.events)
        super().__init__(items)

    @property
    def items(self):
        """The distinct item ids of the session, in first-view order."""
        return list(self)

    def __eq__(self, other):
        if isinstance(other, Session):
            return self.user == other.user and list.__eq__(self, other)
        if isinstance(other, (list, tuple)):
            return list(self) == list(other)
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    __hash__ = None

    def __repr__(self):
        return f"Session(user={self.user!r}, items={list(self)!r})"


def _to_seconds(value):
    """Converts a single timestamp to a float number of seconds."""
    if isinstance(value, np.datetime64):
        return float(value.astype("datetime64[s]").astype(np.int64))
    if isinstance(value, _datetime.datetime):
        if value.tzinfo is not None:
            return value.timestamp()
        # Treat naive datetimes as UTC so results don't depend on the local timezone.
        return (value - _EPOCH).total_seconds()
    if isinstance(value, _datetime.date):
        return (value - _EPOCH.date()).total_seconds()
    if isinstance(value, _datetime.timedelta):
        return value.total_seconds()
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("timestamps must not be booleans")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        # Accept ISO-8601 strings (e.g. "2020-01-01T12:00:00" or "2020-01-01 12:00:00").
        try:
            parsed = np.datetime64(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "timestamps given as strings must be parseable as ISO-8601 datetimes; "
                f"got {value!r}"
            ) from exc
        if np.isnat(parsed):
            raise ValueError(f"could not parse timestamp {value!r}")
        return _to_seconds(parsed)
    raise TypeError(
        "timestamps must be numeric, datetime-like, or convertible to seconds; "
        f"got {type(value).__name__}"
    )


def _timestamps_to_seconds(timestamps):
    array = np.asarray(timestamps)
    if array.dtype.kind == "M":
        seconds = array.astype("datetime64[s]").astype(np.float64)
    elif array.dtype.kind in "iuf":
        seconds = array.astype(np.float64)
    else:
        seconds = np.empty(len(array), dtype=np.float64)
        for index, value in enumerate(array):
            seconds[index] = _to_seconds(value)

    if np.isnan(seconds).any():
        raise ValueError("timestamps must not contain NaN values")
    return seconds


def _detect_dataframe_columns(frame):
    lower = {str(name).lower(): name for name in frame.columns}
    chosen = []
    for candidates, kind in (
        (_USER_COLUMNS, "user"),
        (_ITEM_COLUMNS, "item"),
        (_TIMESTAMP_COLUMNS, "timestamp"),
    ):
        match = next((lower[name] for name in candidates if name in lower), None)
        if match is None:
            raise ValueError(
                f"could not detect a {kind} column in events; "
                "pass columns=(user, item, timestamp) explicitly"
            )
        chosen.append(match)
    return tuple(chosen)


def _parse_events(events, columns=None):
    """Normalizes the accepted event containers into parallel user/item/timestamp lists."""
    if events is None:
        return [], [], []

    # pandas (or any DataFrame-like object) - duck typed to keep pandas optional.
    if hasattr(events, "columns") and hasattr(events, "__getitem__"):
        if columns is None:
            columns = _detect_dataframe_columns(events)
        elif isinstance(columns, str) or len(columns) != 3:
            raise ValueError("columns must be a sequence of (user, item, timestamp) names")
        try:
            users = list(events[columns[0]])
            items = list(events[columns[1]])
            timestamps = list(events[columns[2]])
        except KeyError as exc:
            raise ValueError(f"missing column {exc} in events") from exc
        return users, items, timestamps

    # A mapping of column name -> values is treated like a DataFrame.
    if isinstance(events, Mapping):
        if columns is None:
            columns = ("user", "item", "timestamp")
        elif isinstance(columns, str) or len(columns) != 3:
            raise ValueError("columns must be a sequence of (user, item, timestamp) names")
        try:
            values = [events[columns[0]], events[columns[1]], events[columns[2]]]
        except KeyError as exc:
            raise ValueError(f"missing key {exc} in events") from exc
        scalars = all(
            isinstance(value, (str, bytes)) or not hasattr(value, "__iter__") for value in values
        )
        if scalars:
            # A single event passed as a mapping, e.g. {"user": ..., "item": ..., ...}
            return [values[0]], [values[1]], [values[2]]
        return list(values[0]), list(values[1]), list(values[2])

    if isinstance(events, np.ndarray):
        if events.dtype.names is not None:
            # structured array with named fields
            if columns is None:
                names = events.dtype.names
                if not {"user", "item", "timestamp"}.issubset(names):
                    raise ValueError(
                        "events structured arrays need user/item/timestamp fields; "
                        "pass columns=(user, item, timestamp) explicitly"
                    )
                columns = ("user", "item", "timestamp")
            users = list(events[columns[0]])
            items = list(events[columns[1]])
            timestamps = list(events[columns[2]])
            return users, items, timestamps
        if events.ndim != 2 or events.shape[1] != 3:
            raise ValueError("events arrays must have shape (num_events, 3)")
        users = list(events[:, 0])
        items = list(events[:, 1])
        timestamps = list(events[:, 2])
        return users, items, timestamps

    if isinstance(events, (str, bytes)):
        raise TypeError("events must be an iterable of (user, item, timestamp) records")

    users, items, timestamps = [], [], []
    for index, event in enumerate(events):
        if isinstance(event, Event):
            user, item, timestamp = event
        elif isinstance(event, Mapping):
            try:
                user = event["user"]
                item = event["item"]
                timestamp = event["timestamp"]
            except KeyError as exc:
                raise ValueError(f"event {index} is missing key {exc}") from exc
        else:
            try:
                user, item, timestamp = event
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "events must be (user, item, timestamp) records; "
                    f"event {index} has type {type(event).__name__}"
                ) from exc
        users.append(user)
        items.append(item)
        timestamps.append(timestamp)
    return users, items, timestamps


def _validate_idle_timeout(idle_timeout):
    if idle_timeout is None:
        raise ValueError("idle_timeout must be a number of seconds")
    if isinstance(idle_timeout, _datetime.timedelta):
        idle_timeout = idle_timeout.total_seconds()
    try:
        idle_timeout = float(idle_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("idle_timeout must be a number of seconds") from exc
    if np.isnan(idle_timeout) or idle_timeout < 0:
        raise ValueError("idle_timeout must be a non-negative number of seconds")
    return idle_timeout


def _assign_sessions(users, timestamps, idle_timeout):
    """Assigns each event to a session.

    Returns ``(session_of_event, session_users, order)`` where ``session_of_event`` is aligned
    with the input events, ``session_users`` holds the user of each session in session order,
    and ``order`` is the index order that sorts events by (user, timestamp).
    """
    count = len(users)
    if count == 0:
        return np.empty(0, dtype=np.int64), [], np.empty(0, dtype=np.int64)

    # Assign compact user codes in order of first appearance; this keeps the output stable and
    # avoids requiring user ids to be sortable.
    user_codes = np.empty(count, dtype=np.int64)
    code_by_user = {}
    for index, user in enumerate(users):
        try:
            code = code_by_user.get(user)
        except TypeError as exc:
            raise TypeError("user ids must be hashable") from exc
        if code is None:
            code = len(code_by_user)
            code_by_user[user] = code
        user_codes[index] = code

    seconds = _timestamps_to_seconds(timestamps)
    # Sort by user first, then timestamp, then original position (lexsort is stable).
    order = np.lexsort((np.arange(count), seconds, user_codes))

    sorted_users = user_codes[order]
    sorted_seconds = seconds[order]
    # A new session starts when the user changes, or the idle gap exceeds the timeout.
    new_session = np.empty(count, dtype=bool)
    new_session[0] = True
    if count > 1:
        user_changed = sorted_users[1:] != sorted_users[:-1]
        idle_gap = sorted_seconds[1:] - sorted_seconds[:-1] > idle_timeout
        new_session[1:] = user_changed | idle_gap

    session_of_sorted_event = np.cumsum(new_session) - 1
    session_of_event = np.empty(count, dtype=np.int64)
    session_of_event[order] = session_of_sorted_event

    starts = np.flatnonzero(new_session)
    session_users = [users[order[start]] for start in starts]
    return session_of_event, session_users, order


def split_sessions(events, idle_timeout=DEFAULT_IDLE_TIMEOUT, columns=None):
    """Splits timestamped events into sessions.

    Events are grouped by user and ordered by timestamp.  A new session begins when the gap
    between consecutive events of the same user exceeds ``idle_timeout`` seconds.  A gap equal
    to ``idle_timeout`` stays in the same session.

    Parameters
    ----------
    events : iterable of (user, item, timestamp), numpy.ndarray or pandas.DataFrame
        The events to split.  See the module documentation for accepted formats.
    idle_timeout : float, optional
        Maximum idle time, in seconds, that keeps two consecutive events in the same session.
        Defaults to 30 minutes.
    columns : tuple of str, optional
        Column names to use when ``events`` is a DataFrame.

    Returns
    -------
    list of Session
        Sessions in a deterministic order (users in order of first appearance, then by time).
        Each session is a list of its distinct item ids and also exposes ``user`` and ``events``
        attributes.
    """
    idle_timeout = _validate_idle_timeout(idle_timeout)
    users, items, timestamps = _parse_events(events, columns=columns)
    session_of_event, session_users, order = _assign_sessions(users, timestamps, idle_timeout)

    grouped = [[] for _ in session_users]
    for event_index in order:
        session_index = session_of_event[event_index]
        grouped[session_index].append(
            Event(users[event_index], items[event_index], timestamps[event_index])
        )
    return [
        Session(user=session_users[index], events=tuple(grouped[index]))
        for index in range(len(session_users))
    ]


#: Alias of :func:`split_sessions`.
segment_sessions = split_sessions
#: Alias of :func:`split_sessions`.
group_sessions = split_sessions
#: Alias of :func:`split_sessions`.
group_into_sessions = split_sessions
#: Alias of :func:`split_sessions`.
sessions_from_events = split_sessions


def _dedupe_in_order(items):
    seen = {}
    unique = []
    for item in items:
        try:
            if item not in seen:
                seen[item] = True
                unique.append(item)
        except TypeError as exc:
            raise TypeError("item ids must be hashable") from exc
    return unique


def build_sessions(events, idle_timeout=DEFAULT_IDLE_TIMEOUT, columns=None, dedupe=True):
    """Returns sessions as lists of item ids.

    This is a lightweight view over :func:`split_sessions`; the default output is therefore
    identical to :func:`split_sessions`.  With ``dedupe=False`` repeated views of the same item
    inside a session are preserved as separate entries.
    """
    if dedupe:
        return split_sessions(events, idle_timeout=idle_timeout, columns=columns)
    sessions = split_sessions(events, idle_timeout=idle_timeout, columns=columns)
    return [
        Session(user=session.user, events=session.events, items=[e.item for e in session.events])
        for session in sessions
    ]


#: Alias of :func:`build_sessions`.
sessionize = build_sessions


def _unique_in_order(values):
    unique = {}
    for value in values:
        try:
            unique.setdefault(value, len(unique))
        except TypeError as exc:
            raise TypeError("item ids must be hashable") from exc
    return list(unique)


def _build_item_index(all_items, item_ids, num_items):
    """Builds the column mapping for the item dimension."""
    if item_ids is not None:
        item_ids = np.asarray(list(item_ids), dtype=object)
        label_to_column = {}
        for column, label in enumerate(item_ids):
            try:
                label_to_column[label] = column
            except TypeError as exc:
                raise TypeError("item ids must be hashable") from exc
        if num_items is not None and int(num_items) != len(item_ids):
            raise ValueError("num_items must match the length of item_ids when both are given")
        return label_to_column, len(item_ids), item_ids

    if num_items is not None:
        num_items = int(num_items)
        if num_items < 0:
            raise ValueError("num_items must be non-negative")

    # Fast path: non-negative integer item ids used directly as column indices.
    integer_ids = True
    max_item = -1
    for item in all_items:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            integer_ids = False
            break
        if item < 0:
            integer_ids = False
            break
        max_item = max(max_item, int(item))

    if integer_ids:
        if num_items is None:
            num_items = max_item + 1
        elif max_item >= num_items:
            raise ValueError(f"item id {max_item} is outside the given num_items={num_items}")
        return None, num_items, np.arange(num_items, dtype=np.int64)

    # Labels: compact columns in order of first appearance.
    labels = _unique_in_order(all_items)
    if num_items is not None:
        if num_items < len(labels):
            raise ValueError("num_items is smaller than the number of distinct items")
        labels = labels + [None] * (num_items - len(labels))
    label_to_column = {label: index for index, label in enumerate(labels)}
    return label_to_column, len(labels), np.asarray(labels, dtype=object)


def matrix_from_sessions(
    sessions,
    item_ids=None,
    num_items=None,
    duplicate="binary",
    min_session_items=1,
    return_item_ids=False,
):
    """Builds a sparse ``(sessions x items)`` matrix from already-sessionized item lists.

    Each element of ``sessions`` is an iterable of item ids viewed in one session.  Duplicate
    items within a session are merged (see ``duplicate``), and items from different sessions are
    never placed in the same row, so they can never be counted as co-occurring.

    Parameters
    ----------
    sessions : iterable of iterables
        One iterable of item ids per session.
    item_ids : sequence, optional
        Complete set of item labels, fixing the item columns.
    num_items : int, optional
        Number of item columns.  Inferred when omitted.
    duplicate : {"binary", "sum"}, optional
        ``"binary"`` (default) records each distinct item once per session; ``"sum"`` records
        the number of times it was viewed.  Either way duplicate events are merged into a single
        matrix entry rather than left as repeated entries.
    min_session_items : int, optional
        Drop sessions with fewer than this many distinct items.
    return_item_ids : bool, optional
        Return ``(matrix, item_ids)`` instead of just the matrix.

    Returns
    -------
    scipy.sparse.csr_matrix or (scipy.sparse.csr_matrix, numpy.ndarray)
    """
    if duplicate not in ("binary", "sum"):
        raise ValueError("duplicate must be 'binary' or 'sum'")
    if int(min_session_items) < 1:
        raise ValueError("min_session_items must be at least 1")
    min_session_items = int(min_session_items)

    cleaned = [list(session_items) for session_items in sessions]
    all_items = []
    for session_items in cleaned:
        all_items.extend(session_items)
    label_to_column, column_count, resolved_item_ids = _build_item_index(
        all_items, item_ids, num_items
    )

    def column_of(item):
        if label_to_column is None:
            return item
        try:
            return label_to_column[item]
        except KeyError as exc:
            raise ValueError(f"unknown item id {item!r}; not present in item_ids") from exc

    rows, cols, values = [], [], []
    kept = 0
    for session_items in cleaned:
        # Repeated views merge into one entry per (session, item). With duplicate="sum" the
        # entry counts the repeats; with "binary" it is always 1.
        counts = {}
        for item in session_items:
            column = column_of(item)
            if duplicate == "sum":
                counts[column] = counts.get(column, 0.0) + 1.0
            else:
                counts[column] = 1.0
        if len(counts) < min_session_items:
            continue
        for column, value in counts.items():
            rows.append(kept)
            cols.append(column)
            values.append(value)
        kept += 1

    matrix = coo_matrix(
        (np.asarray(values, dtype=np.float64), (rows, cols)),
        shape=(kept, column_count),
    ).tocsr()
    # coo_matrix already sums duplicate (row, column) pairs, but sorting makes the structure
    # canonical and deterministic.
    matrix.sum_duplicates()
    matrix.sort_indices()

    if return_item_ids:
        return matrix, resolved_item_ids
    return matrix


def build_session_matrix(
    events,
    idle_timeout=DEFAULT_IDLE_TIMEOUT,
    columns=None,
    item_ids=None,
    num_items=None,
    duplicate="binary",
    min_session_items=1,
    return_item_ids=False,
):
    """Builds a sparse ``(sessions x items)`` co-occurrence input from events.

    Each row is one session produced by :func:`split_sessions`; each column is an item.  The
    value of a cell is the weight of that item inside the session.  Since a row never spans two
    sessions, items from different sessions of the same user are never counted as co-occurring.

    Parameters
    ----------
    events : iterable of (user, item, timestamp), numpy.ndarray or pandas.DataFrame
        Timestamped events.
    idle_timeout : float, optional
        Idle time in seconds used to split sessions.  Defaults to 30 minutes.
    columns : tuple of str, optional
        Column names to use when ``events`` is a DataFrame.
    item_ids : sequence, optional
        The complete set of item labels.  Columns follow this order and every item seen in the
        events must be present in it.  When omitted, non-negative integer items are used as
        column indices directly, and other labels are mapped to columns in order of first
        appearance.
    num_items : int, optional
        Number of item columns.  Defaults to ``max(item) + 1`` for integer item ids, or the
        number of distinct labels otherwise.
    duplicate : {"binary", "sum"}, optional
        How repeated events for the same item in one session are merged.  ``"binary"``
        (default) records each distinct item once per session; ``"sum"`` records how many times
        it was viewed.
    min_session_items : int, optional
        Sessions containing fewer than this many *distinct* items are dropped.  Useful for
        filtering out single-item sessions, which add no co-occurrence signal.  Defaults to 1,
        which keeps all non-empty sessions.
    return_item_ids : bool, optional
        When True, returns ``(matrix, item_ids)``.  ``item_ids`` maps column indices to labels
        (``np.arange`` for integer ids used directly).

    Returns
    -------
    scipy.sparse.csr_matrix or (scipy.sparse.csr_matrix, numpy.ndarray)
        The session-item matrix, optionally with the column labels.
    """
    sessions = split_sessions(events, idle_timeout=idle_timeout, columns=columns)
    return matrix_from_sessions(
        # Use the raw event items rather than the deduplicated ``Session`` view so that
        # duplicate="sum" can count repeated views.
        ([event.item for event in session.events] for session in sessions),
        item_ids=item_ids,
        num_items=num_items,
        duplicate=duplicate,
        min_session_items=min_session_items,
        return_item_ids=return_item_ids,
    )


#: Alias of :func:`build_session_matrix`.
build_cooccurrence_matrix = build_session_matrix
#: Alias of :func:`build_session_matrix`.
build_item_matrix = build_session_matrix
#: Alias of :func:`build_session_matrix`.
build_cooccurrence_input = build_session_matrix
#: Alias of :func:`build_session_matrix`.
session_item_matrix = build_session_matrix


def cooccurrence_counts(
    events,
    idle_timeout=DEFAULT_IDLE_TIMEOUT,
    columns=None,
    item_ids=None,
    num_items=None,
    min_session_items=1,
):
    """Returns a sparse ``(items x items)`` matrix of in-session co-occurrence counts.

    Each pair of distinct items that appears together in at least one session increments the
    corresponding cell once per session.  Items browsed in different sessions of the same user
    never co-occur.

    Parameters
    ----------
    events : iterable of (user, item, timestamp), numpy.ndarray or pandas.DataFrame
        Timestamped events.
    idle_timeout : float, optional
        Idle time in seconds used to split sessions.
    columns : tuple of str, optional
        Column names to use when ``events`` is a DataFrame.
    item_ids : sequence, optional
        Complete set of item labels, fixing the item dimension.
    num_items : int, optional
        Number of items.  Inferred when omitted.
    min_session_items : int, optional
        Ignore sessions with fewer than this many distinct items.

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric matrix where ``result[i, j]`` counts the sessions containing both items.
    """
    matrix = build_session_matrix(
        events,
        idle_timeout=idle_timeout,
        columns=columns,
        item_ids=item_ids,
        num_items=num_items,
        duplicate="binary",
        min_session_items=min_session_items,
    )
    # (sessions x items) -> (items x sessions) x (sessions x items) = co-occurrence counts.
    cooccurrence = (matrix.T @ matrix).tocsr()
    cooccurrence.setdiag(0)
    cooccurrence.eliminate_zeros()
    cooccurrence.sort_indices()
    return cooccurrence


#: Alias of :func:`cooccurrence_counts`.
session_cooccurrence = cooccurrence_counts


def sessions_from_arrays(users, items, timestamps, idle_timeout=DEFAULT_IDLE_TIMEOUT):
    """Splits events given as three parallel arrays into sessions.

    This is a thin convenience wrapper around :func:`split_sessions` for callers that already
    hold separate ``users``, ``items`` and ``timestamps`` arrays.
    """
    users = list(users)
    items = list(items)
    timestamps = list(timestamps)
    if not (len(users) == len(items) == len(timestamps)):
        raise ValueError("users, items and timestamps must have the same length")
    events = list(zip(users, items, timestamps))
    return split_sessions(events, idle_timeout=idle_timeout)


def matrix_from_arrays(
    users,
    items,
    timestamps,
    idle_timeout=DEFAULT_IDLE_TIMEOUT,
    item_ids=None,
    num_items=None,
    duplicate="binary",
    min_session_items=1,
    return_item_ids=False,
):
    """Builds the session-item matrix from three parallel arrays.

    Equivalent to calling :func:`build_session_matrix` on ``zip(users, items, timestamps)``.
    """
    users = list(users)
    items = list(items)
    timestamps = list(timestamps)
    if not (len(users) == len(items) == len(timestamps)):
        raise ValueError("users, items and timestamps must have the same length")
    events = list(zip(users, items, timestamps))
    return build_session_matrix(
        events,
        idle_timeout=idle_timeout,
        item_ids=item_ids,
        num_items=num_items,
        duplicate=duplicate,
        min_session_items=min_session_items,
        return_item_ids=return_item_ids,
    )



class Sessionizer:
    """Splits events into sessions and builds session-item matrices with fixed settings.

    A small convenience wrapper for reuse when the same options are applied to several event
    batches::

        sessionizer = Sessionizer(idle_timeout=30 * 60)
        sessions = sessionizer.split_sessions(events)
        matrix = sessionizer.build_matrix(events)

    Parameters
    ----------
    idle_timeout : float, optional
        Idle time in seconds that separates sessions.  Defaults to 30 minutes.
    columns : tuple of str, optional
        Column names to use for DataFrame inputs.
    item_ids : sequence, optional
        Complete set of item labels, fixing the item dimension.
    num_items : int, optional
        Number of item columns.
    duplicate : {"binary", "sum"}, optional
        How to merge repeated item views inside a session.  Defaults to ``"binary"``.
    min_session_items : int, optional
        Drop sessions with fewer than this many distinct items.  Defaults to 1.
    """

    def __init__(
        self,
        idle_timeout=DEFAULT_IDLE_TIMEOUT,
        columns=None,
        item_ids=None,
        num_items=None,
        duplicate="binary",
        min_session_items=1,
    ):
        self.idle_timeout = _validate_idle_timeout(idle_timeout)
        self.columns = columns
        self.item_ids = item_ids
        self.num_items = num_items
        self.duplicate = duplicate
        self.min_session_items = min_session_items

    def split_sessions(self, events):
        """Splits ``events`` into sessions using this object's settings."""
        return split_sessions(events, idle_timeout=self.idle_timeout, columns=self.columns)

    def build_sessions(self, events, dedupe=True):
        """Returns sessions as lists of item ids using this object's settings."""
        return build_sessions(
            events, idle_timeout=self.idle_timeout, columns=self.columns, dedupe=dedupe
        )

    def build_matrix(self, events, return_item_ids=False):
        """Builds the sparse ``(sessions x items)`` matrix using this object's settings."""
        return build_session_matrix(
            events,
            idle_timeout=self.idle_timeout,
            columns=self.columns,
            item_ids=self.item_ids,
            num_items=self.num_items,
            duplicate=self.duplicate,
            min_session_items=self.min_session_items,
            return_item_ids=return_item_ids,
        )

    def cooccurrence(self, events):
        """Returns the ``(items x items)`` in-session co-occurrence counts."""
        return cooccurrence_counts(
            events,
            idle_timeout=self.idle_timeout,
            columns=self.columns,
            item_ids=self.item_ids,
            num_items=self.num_items,
            min_session_items=self.min_session_items,
        )

    def fit(self, model, events, show_progress=True, **kwargs):
        """Fits an item-item model through :meth:`ItemItemRecommender.fit_sessions`."""
        return model.fit_sessions(
            events,
            idle_timeout=self.idle_timeout,
            columns=self.columns,
            item_ids=self.item_ids,
            num_items=self.num_items,
            duplicate=self.duplicate,
            min_session_items=self.min_session_items,
            show_progress=show_progress,
            **kwargs,
        )

    def __repr__(self):
        return (
            f"Sessionizer(idle_timeout={self.idle_timeout!r}, num_items={self.num_items!r}, "
            f"duplicate={self.duplicate!r})"
        )


__all__ = [
    "DEFAULT_IDLE_TIMEOUT",
    "Event",
    "Session",
    "Sessionizer",
    "build_cooccurrence_input",
    "build_cooccurrence_matrix",
    "build_item_matrix",
    "build_session_matrix",
    "build_sessions",
    "cooccurrence_counts",
    "group_into_sessions",
    "group_sessions",
    "matrix_from_arrays",
    "matrix_from_sessions",
    "segment_sessions",
    "session_cooccurrence",
    "session_item_matrix",
    "sessionize",
    "sessions_from_arrays",
    "sessions_from_events",
    "split_sessions",
]
