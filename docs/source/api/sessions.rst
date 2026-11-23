Session Co-occurrence
=====================

Item-item nearest neighbour models treat each row of their input as one basket of items viewed
together. When the input is a raw log of browsing events, all of a member's events end up in the
same row - so products browsed weeks apart get counted as co-occurring, and the similarity model
recommends unrelated items on each other's pages.

:mod:`implicit.sessions` fixes this by building the sparse co-occurrence input from *sessions*:
events are ordered per member and split whenever the member has been idle for longer than
``idle_timeout`` seconds, and repeated events for the same item inside a session are merged.
Because one row is one session rather than one member, items that only appear in different
sessions of the same member never co-occur.

.. code-block:: python

    import implicit

    model = implicit.nearest_neighbours.TFIDFRecommender()
    model.fit_sessions(browse_events, idle_timeout=30 * 60)
    ids, scores = model.similar_items(itemid)

``browse_events`` can be an iterable of ``(user, item, timestamp)`` records, a 2D numpy array, or
a pandas ``DataFrame``. To build the matrix yourself and feed it to ``fit`` directly:

.. code-block:: python

    from implicit.sessions import build_session_matrix

    session_items = build_session_matrix(events, idle_timeout=1800)
    model.fit(session_items, show_progress=False)

.. automodule:: implicit.sessions
   :members: split_sessions, build_sessions, matrix_from_sessions, build_session_matrix, cooccurrence_counts, sessions_from_arrays, matrix_from_arrays, Sessionizer, Event, Session
