"""Stable mappings between business keys and integer indices.

The recommendation models in this library operate on integer row/column positions inside
sparse matrices (and inside the ``user_factors`` / ``item_factors`` arrays).  Those
positions are meaningless outside of the process that built the matrix: if a product
catalog is re-sorted, or new SKUs are inserted in the middle of it, the position of every
existing item shifts and incremental updates silently write new behaviour into old rows.

:class:`IdMap` solves this by keeping an explicit, *append only* mapping between arbitrary
hashable business keys (SKUs, user ids, UUIDs, ...) and contiguous integer indices:

* keys already present in the map always keep their original index;
* previously unseen keys are appended, in first-seen order;
* feeding an existing key again is idempotent and resolves to its original index.

The map can be stored on a model as ``user_id_map`` / ``item_id_map``, is persisted when
the model is saved, and is used by the ``*_by_key`` update methods, the interaction matrix
builder and the recommend methods.
"""

import json

import numpy as np
from scipy.sparse import coo_matrix

__all__ = [
    "IdMap",
    "KeyMap",
    "build_interaction_matrix",
    "build_interactions",
    "build_user_items",
    "dedupe_keys",
    "map_indices_to_keys",
    "resolve_indices",
]


def _normalize_key(key):
    """Convert numpy scalars to plain python objects so keys stay hashable/serializable."""
    if isinstance(key, np.generic):
        return key.item()
    return key


def _is_integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


# JSON round-trips these exactly; tuples collapse to lists and objects are ambiguous.
_SERIALIZABLE_KEY_TYPES = (str, int, float, bool, type(None))


class IdMap:
    """An append-only, stable mapping between business keys and integer indices.

    Parameters
    ----------
    keys : iterable, optional
        Initial keys. Keys are assigned indices ``0..len(keys)-1`` in the order given.
        A key that is repeated in this iterable keeps the index of its first occurrence.
    index_dtype : numpy dtype, optional
        The integer dtype of the returned index arrays. Defaults to ``np.int64``.

    Examples
    --------
    >>> id_map = IdMap(["sku-a", "sku-b"])
    >>> id_map.add_many(["sku-b", "sku-c"])  # existing key keeps its index, new key appended
    array([1, 2])
    >>> id_map.keys
    ['sku-a', 'sku-b', 'sku-c']

    Notes
    -----
    Keys must be hashable. Repeated keys inside a single :meth:`add_many` call are *not*
    an error: they all resolve to the same index and no duplicate index is allocated.
    Use :func:`dedupe_keys` when a caller needs duplicate input keys to be rejected or
    collapsed explicitly.
    """

    def __init__(self, keys=None, index_dtype=np.int64):
        self.index_dtype = np.dtype(index_dtype)
        if self.index_dtype.kind not in "iu":
            raise ValueError(f"index_dtype must be an integer dtype, got {self.index_dtype}")
        self._keys = []
        self._key_to_index = {}
        if keys is not None:
            self.add_many(keys)

    # -- insertion ---------------------------------------------------------------------
    def add(self, key):
        """Add a single key if needed and return its (python int) index."""
        return self.add_many([key])[0].item()

    def add_many(self, keys):
        """Map ``keys`` to indices, appending any previously unseen keys.

        Existing keys keep their original index. New keys are appended in first-seen
        order. Repeated keys within ``keys`` resolve to the same index.

        Returns
        -------
        numpy.ndarray
            Array of indices with dtype :attr:`index_dtype`, one per input key.
        """
        keys = [_normalize_key(key) for key in keys]
        indices = np.empty(len(keys), dtype=self.index_dtype)
        for position, key in enumerate(keys):
            try:
                index = self._key_to_index.get(key)
            except TypeError as exc:  # unhashable key
                raise TypeError(f"business keys must be hashable, got {key!r}") from exc

            if index is None:
                index = len(self._keys)
                self._key_to_index[key] = index
                self._keys.append(key)
            indices[position] = index
        return indices

    # -- lookups -----------------------------------------------------------------------
    def index(self, key):
        """Return the index for ``key``, raising :class:`KeyError` if it is unknown."""
        key = _normalize_key(key)
        try:
            return self._key_to_index[key]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"unknown business key {key!r}") from exc

    def indices(self, keys):
        """Return the indices for ``keys``, raising :class:`KeyError` for unknown keys."""
        keys = [_normalize_key(key) for key in keys]
        missing = [key for key in keys if key not in self._key_to_index]
        if missing:
            raise KeyError(f"unknown business key(s) {missing!r}")
        return np.asarray([self._key_to_index[key] for key in keys], dtype=self.index_dtype)

    def key(self, index):
        """Return the business key stored at ``index``."""
        index = int(index)
        if index < 0 or index >= len(self._keys):
            raise IndexError(f"index {index} is out of range for an IdMap of size {len(self)}")
        return self._keys[index]

    def keys_of(self, indices):
        """Return a list of business keys for ``indices``."""
        return [self.key(index) for index in indices]

    @property
    def keys(self):
        """A copy of the keys in index order."""
        return list(self._keys)

    def is_positional(self):
        """True when the keys are exactly ``0..len-1`` (a positional placeholder map)."""
        return all(
            isinstance(key, (int, np.integer)) and not isinstance(key, bool) and key == index
            for index, key in enumerate(self._keys)
        )

    def copy(self):
        """Return a copy of this map (keys are shared, indices are independent)."""
        ret = IdMap(index_dtype=self.index_dtype)
        ret._keys = list(self._keys)
        ret._key_to_index = dict(self._key_to_index)
        return ret

    # -- persistence -------------------------------------------------------------------
    def to_json(self):
        """Serialize the map to a JSON string (usable inside ``np.savez`` without pickle)."""
        # JSON silently converts tuples to lists (and would happily pick apart other
        # objects), so reject anything that wouldn't round-trip to an equal key.
        for key in self._keys:
            if not isinstance(key, _SERIALIZABLE_KEY_TYPES):
                raise TypeError(
                    "IdMap keys must be JSON serializable (str/int/float/bool/None) to be "
                    f"saved with a model; got {key!r} of type {type(key).__name__}. Convert "
                    "your business keys to strings first."
                )
        payload = json.dumps({"keys": self._keys, "index_dtype": self.index_dtype.name})
        return payload

    @classmethod
    def from_json(cls, value):
        """Rebuild a map from the output of :meth:`to_json`."""
        # np.savez/np.load round-trips a python string as a 0-d array; unwrap it
        if isinstance(value, np.ndarray):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf8")
        data = json.loads(value)
        return cls(data["keys"], index_dtype=data.get("index_dtype", np.int64))

    @classmethod
    def from_keys(cls, keys, index_dtype=np.int64):
        """Alias for ``IdMap(keys, index_dtype=index_dtype)``."""
        return cls(keys, index_dtype=index_dtype)

    # -- dunder ------------------------------------------------------------------------
    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        try:
            return _normalize_key(key) in self._key_to_index
        except TypeError:
            return False

    def __iter__(self):
        return iter(self._keys)

    def __eq__(self, other):
        if not isinstance(other, IdMap):
            return NotImplemented
        return self.index_dtype == other.index_dtype and self._keys == other._keys

    def __repr__(self):
        return f"IdMap(keys={self._keys!r}, index_dtype={self.index_dtype.name!r})"


# ``KeyMap`` is provided as a more descriptive alias.
KeyMap = IdMap


def dedupe_keys(keys, on_duplicate="error", name="key"):
    """Validate that ``keys`` are unique, returning the rows to keep.

    Parameters
    ----------
    keys : iterable
        The business keys supplied to an update call.
    on_duplicate : {"error", "first", "last"}
        ``"error"`` (default) raises :class:`ValueError` when a key is repeated,
        ``"first"`` keeps the first occurrence of each key, and ``"last"`` keeps the last.
    name : str
        Used in error messages, e.g. ``"user"`` or ``"item"``.

    Returns
    -------
    tuple
        ``(unique_keys, positions)`` where ``positions`` is an integer array that can be
        used to select the matching rows from the input sparse matrix.
    """
    if on_duplicate not in ("error", "first", "last"):
        raise ValueError(
            f"on_duplicate must be one of 'error', 'first' or 'last', got {on_duplicate!r}"
        )

    normalized = [_normalize_key(key) for key in keys]
    seen = {}
    first_seen_order = []
    for position, key in enumerate(normalized):
        try:
            previous = seen.get(key)
        except TypeError as exc:
            raise TypeError(f"business keys must be hashable, got {key!r}") from exc

        if previous is None:
            first_seen_order.append(key)
            seen[key] = position
        elif on_duplicate == "error":
            raise ValueError(
                f"duplicate {name} key {key!r} in input; pass on_duplicate='first' or "
                "on_duplicate='last' to collapse duplicates explicitly"
            )
        elif on_duplicate == "last":
            seen[key] = position

    # keys keep the order in which they were first encountered; positions point at the
    # chosen representative row for each key
    positions = np.array([seen[key] for key in first_seen_order], dtype=np.int64)
    return list(first_seen_order), positions


def resolve_indices(id_map, values, num_known=None, allow_new=False, name="id"):
    """Translate business keys - or integer indices - into positional indices.

    The resolution rules are deliberately explicit:

    * When ``allow_new=True`` (model update path), non-integer values are treated as
      business keys and appended to ``id_map``; integer values are treated as positional
      indices (this keeps the historical integer-index API working).
    * When ``allow_new=False`` (lookup/recommend path) and ``id_map`` is empty, integer
      values are positional indices and anything else raises :class:`KeyError`.
    * When ``allow_new=False`` and ``id_map`` is non-empty, values are first looked up as
      business keys. If every value is an unknown *integer* that is nevertheless a valid
      index, they are interpreted as positional indices (so evaluation helpers that pass
      ``np.arange(...)`` keep working on keyed models).

    Returns a python ``int`` for scalar input, otherwise a numpy index array.
    """
    is_scalar = np.isscalar(values) or (
        isinstance(values, np.ndarray) and values.ndim == 0
    )
    if is_scalar:
        raw = values.item() if isinstance(values, np.ndarray) else values
        flat_values = [_normalize_key(raw)]
    elif isinstance(values, np.ndarray):
        flat_values = [_normalize_key(value) for value in values.ravel().tolist()]
    else:
        flat_values = [_normalize_key(value) for value in values]

    if not flat_values:
        empty = np.empty(0, dtype=id_map.index_dtype if id_map is not None else np.int64)
        return empty

    all_integer = all(_is_integer(value) for value in flat_values)

    def _finish(indices):
        if is_scalar:
            return int(np.asarray(indices).reshape(-1)[0])
        return indices

    if allow_new:
        # Keep the historical "these are positional indices" contract for integer input;
        # only non-integer values are interpreted (and appended) as business keys.
        if all_integer:
            indices = np.asarray(flat_values, dtype=np.int64)
            if (indices < 0).any():
                raise IndexError(f"{name} indices must be non-negative, got {indices.min()}")
            return _finish(indices)
        if id_map is None:
            raise KeyError(
                f"model has no {name} key map; attach one with set_id_maps() or train the "
                "model from business keys before updating by key"
            )
        return _finish(id_map.add_many(flat_values))

    if not all_integer:
        if id_map is None:
            raise KeyError(
                f"model has no {name} key map, so {name} values must be integer indices; "
                f"got {flat_values[0]!r}. Build the model from business keys first."
            )
        return _finish(id_map.indices(flat_values))

    if id_map is None or len(id_map) == 0:
        indices = np.asarray(flat_values, dtype=np.int64)
        if num_known is not None and ((indices < 0) | (indices >= num_known)).any():
            raise IndexError(
                f"some {name}ids are out of range for a model with {num_known} {name}s"
            )
        return _finish(indices)

    # A populated map: business keys win over positional indices, so integer business
    # keys (e.g. numeric user ids) are looked up first.
    if all(value in id_map for value in flat_values):
        return _finish(id_map.indices(flat_values))

    indices = np.asarray(flat_values, dtype=np.int64)
    if (indices >= 0).all() and (num_known is None or (indices < num_known).all()):
        return _finish(indices)
    raise IndexError(
        f"some {name} values are neither known business keys nor valid indices "
        f"for a model with {num_known} {name}s"
    )


def map_indices_to_keys(id_map, ids):
    """Translate positional ``ids`` back to business keys when a key map is populated.

    Returns the input unchanged for models without a key map (or with an empty one), so
    unkeyed models keep returning plain integer arrays. Out-of-range sentinels (such as the
    ``-1`` used to pad batch results when fewer than ``N`` items are available) are
    preserved as-is rather than raising, so padded results stay valid.
    """
    if id_map is None or len(id_map) == 0:
        return ids

    if np.isscalar(ids) or (isinstance(ids, np.ndarray) and ids.ndim == 0):
        return _index_to_key(id_map, int(ids))

    array = np.asarray(ids)
    if array.size == 0:
        return np.empty(array.shape, dtype=object)

    keys = [_index_to_key(id_map, int(index)) for index in array.ravel()]
    mapped = np.empty(len(keys), dtype=object)
    mapped[:] = keys
    return mapped.reshape(array.shape)


def _index_to_key(id_map, index):
    if 0 <= index < len(id_map):
        return id_map.key(index)
    # padding / sentinel value: keep the raw integer rather than raising
    return index


def build_interaction_matrix(
    users,
    items,
    values=None,
    user_id_map=None,
    item_id_map=None,
    on_duplicate="sum",
    shape=None,
    index_dtype=np.int64,
):
    """Build a ``user x item`` CSR matrix from business keys.

    Rows/columns are assigned stable indices via :class:`IdMap`: keys already in
    ``user_id_map`` / ``item_id_map`` keep their index, new keys are appended. Passing the
    same map object to :meth:`fit` (or to the model) therefore guarantees that a catalog
    reshuffle never changes which row belongs to which user or SKU.

    Parameters
    ----------
    users : sequence or 2d array
        Either one user key per interaction (parallel to ``items``), or a list of
        per-row iterables (one entry per row of ``items``). Empty/``None`` rows are
        allowed and produce empty matrix rows.
    items : sequence
        Either one item key per interaction, or a list of per-row iterables.
    values : sequence or float, optional
        Interaction weights. Either a scalar (applied to every interaction), one value
        per interaction, or a list of per-row iterables. Defaults to ``1.0``.
    user_id_map, item_id_map : IdMap, optional
        Existing maps to extend. When omitted new maps are created and returned.
    on_duplicate : {"sum", "max", "error", "last", "first"}
        How to handle the *same user/item pair* appearing more than once:

        * ``"sum"`` (default): add the weights together;
        * ``"max"``: keep the largest weight;
        * ``"last"``: keep the last value seen;
        * ``"first"``: keep the first value seen;
        * ``"error"``: raise :class:`ValueError`.

        When a user or item key is *repeated as an entity* (which is perfectly normal -
        a user that likes several items), the key simply resolves to the same index. This
        parameter only governs repeated (user, item) *pairs*.
    shape : tuple, optional
        Explicit ``(num_users, num_items)`` shape. Defaults to the size of the maps after
        all keys have been added. Any entry smaller than the map size is expanded.
    index_dtype : numpy dtype, optional
        Integer dtype used when creating new maps.

    Returns
    -------
    tuple
        ``(matrix, user_id_map, item_id_map)``.
    """
    if on_duplicate not in ("sum", "max", "error", "last", "first"):
        raise ValueError(
            "on_duplicate must be one of 'sum', 'max', 'error', 'last' or 'first', "
            f"got {on_duplicate!r}"
        )

    user_id_map = IdMap(index_dtype=index_dtype) if user_id_map is None else user_id_map
    item_id_map = IdMap(index_dtype=index_dtype) if item_id_map is None else item_id_map

    # Normalise the three parallel views: either flat per-interaction lists, or
    # list-of-lists with one entry per row. A flat ``users`` list is also accepted
    # alongside nested ``items`` - it is then read as one user key per row.
    users_nested = _is_nested(users)
    items_nested = _is_nested(items)
    row_wise = users_nested or items_nested
    if users_nested and items_nested:
        if len(users) != len(items):
            raise ValueError("users and items must have the same number of rows")
        num_rows = len(users)
    elif row_wise and items_nested:
        users = list(users)
        if len(users) != len(items):
            raise ValueError(
                "a flat users list must have one entry per row of the nested items list"
            )
        num_rows = len(users)
    elif row_wise:
        items = list(items)
        if len(users) != len(items):
            raise ValueError(
                "a flat items list must have one entry per row of the nested users list"
            )
        num_rows = len(items)
    else:
        users = list(users)
        items = list(items)
        if len(users) != len(items):
            raise ValueError(
                f"users and items must be the same length, got {len(users)} and {len(items)}"
            )
        num_rows = None

    if values is None:
        flat_values = None
    elif np.isscalar(values):
        flat_values = values
    elif row_wise:
        flat_values = values
    else:
        flat_values = list(values)
        if len(flat_values) != len(users):
            raise ValueError(
                f"values must be the same length as users, got {len(flat_values)} and {len(users)}"
            )

    # Resolve the entity keys first. This both extends the maps and gives us the row
    # index for every interaction (or every row, in the row-wise case).
    if row_wise:
        if users_nested:
            row_users = [_single_key(users, row) for row in range(num_rows)]
        else:
            # one user key per row
            row_users = list(users)
        user_rows = user_id_map.add_many([_normalize_key(user) for user in row_users])
        row_indices = []
        col_indices = []
        data = []
        for row in range(num_rows):
            row_items = _row_items(items[row])
            if not row_items:
                continue

            columns = item_id_map.add_many(row_items)
            row_values = _row_values(flat_values, row, len(row_items))
            row_indices.append(np.full(len(row_items), user_rows[row], dtype=np.int64))
            col_indices.append(columns)
            data.append(row_values)
    else:
        user_rows = user_id_map.add_many(users)
        column_rows = item_id_map.add_many(items)
        row_indices = np.asarray(user_rows, dtype=np.int64)
        col_indices = np.asarray(column_rows, dtype=np.int64)
        data = np.asarray(flat_values if flat_values is not None else 1.0, dtype=np.float64)
        if data.ndim == 0:
            data = np.full(len(row_indices), data)
        data = data.astype(np.float64)

    if row_wise:
        if row_indices:
            row_indices = np.concatenate(row_indices).astype(np.int64)
            col_indices = np.concatenate(col_indices).astype(np.int64)
            data = np.concatenate(
                [np.asarray(chunk, dtype=np.float64) for chunk in data]
            )
        else:
            row_indices = np.empty(0, dtype=np.int64)
            col_indices = np.empty(0, dtype=np.int64)
            data = np.empty(0, dtype=np.float64)

    default_shape = (len(user_id_map), len(item_id_map))
    if shape is None:
        shape = default_shape
    else:
        shape = (max(int(shape[0]), default_shape[0]), max(int(shape[1]), default_shape[1]))

    matrix = coo_matrix((data, (row_indices, col_indices)), shape=shape)
    matrix = _combine_duplicates(matrix, on_duplicate)
    return matrix, user_id_map, item_id_map


def _single_key(row_users, row):
    """Normalise a per-row user entry (a scalar or a 1-element list) to one key."""
    entry = row_users[row]
    if isinstance(entry, (str, bytes)) or not hasattr(entry, "__iter__"):
        return _normalize_key(entry)
    entry = list(entry)
    if len(entry) != 1:
        raise ValueError(
            f"users row {row} has {len(entry)} user keys; a user_items row belongs to "
            "exactly one user"
        )
    return _normalize_key(entry[0])


def _row_items(entry):
    """Normalise a per-row item entry (a scalar or an iterable of keys) to a flat list."""
    if entry is None:
        return []
    if isinstance(entry, (str, bytes)) or not hasattr(entry, "__iter__"):
        return [_normalize_key(entry)]
    return [_normalize_key(value) for value in entry]


def _is_nested(values):
    """True when ``values`` looks like a per-row sequence of iterables."""
    if isinstance(values, np.ndarray):
        return values.ndim > 1
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        return False
    values = list(values)
    if not values:
        return False
    first = values[0]
    if first is None:
        return True
    return not (isinstance(first, (str, bytes)) or not hasattr(first, "__iter__"))


def _row_values(flat_values, row, count):
    if flat_values is None:
        return np.ones(count, dtype=np.float64)
    if np.isscalar(flat_values):
        return np.full(count, float(flat_values), dtype=np.float64)
    entry = flat_values[row]
    if entry is None:
        return np.ones(count, dtype=np.float64)
    if np.isscalar(entry):
        return np.full(count, float(entry), dtype=np.float64)
    entry = np.asarray(list(entry), dtype=np.float64)
    if entry.shape[0] != count:
        raise ValueError(
            f"row {row} has {count} items but {entry.shape[0]} values; "
            "values must line up with items"
        )
    return entry


def _combine_duplicates(matrix, on_duplicate):
    coo = matrix.tocoo()
    if on_duplicate == "sum":
        coo.sum_duplicates()
        return coo.tocsr()

    if coo.nnz == 0:
        return coo.tocsr()

    keys = coo.row.astype(np.int64) * max(1, coo.shape[1]) + coo.col.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_data = coo.data[order]
    sorted_rows = coo.row[order]
    sorted_cols = coo.col[order]

    boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(sorted_keys)]])

    if on_duplicate == "error":
        duplicated = np.flatnonzero(np.diff(sorted_keys) == 0)
        if duplicated.size:
            index = duplicated[0]
            row = int(sorted_rows[index])
            col = int(sorted_cols[index])
            raise ValueError(
                f"duplicate interaction for the same (user, item) pair ({row}, {col}); "
                "pass on_duplicate='sum', 'max', 'first' or 'last' to combine explicitly"
            )
        return matrix

    if on_duplicate == "max":
        combined = np.array([sorted_data[start:end].max() for start, end in zip(starts, ends)])
    elif on_duplicate == "first":
        combined = sorted_data[starts]
    else:  # last
        combined = sorted_data[ends - 1]

    result = coo_matrix(
        (combined, (sorted_rows[starts], sorted_cols[starts])), shape=coo.shape
    ).tocsr()
    return result


build_interactions = build_interaction_matrix
build_user_items = build_interaction_matrix
