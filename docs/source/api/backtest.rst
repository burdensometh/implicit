Time Based Backtesting
======================

The evaluation helpers in :mod:`implicit.evaluation` (such as
:func:`implicit.evaluation.train_test_split`) hold out a random subset of
interactions. Because the held out interactions are chosen at random, a model
trained on the remaining entries has already seen interactions that happened
*after* the ones it is evaluated on. That future leakage inflates offline
metrics.

:mod:`implicit.backtest` implements a temporal alternative. A single global
cutoff splits a timestamped event log into a training matrix (events at or
before the cutoff) and a validation matrix (positive events strictly after the
cutoff that the user had not already interacted with). The user and item axes
are fixed from the whole log, so both matrices share the same shape, and
accounts without enough pre-cutoff history are reported rather than silently
evaluated.

Example
-------

.. code-block:: python

    from implicit.backtest import time_based_split, evaluate_time_based_split
    from implicit.als import AlternatingLeastSquares

    # events can be a pandas DataFrame, a dict of columns, a structured array
    # or a 2d array of (user, item, timestamp[, value[, positive]])
    train, validation, report = time_based_split(
        events, cutoff="2024-06-01", min_history=5, return_report=True
    )
    print(report.summary())
    # report.insufficient_history_users lists accounts with too little history

    model = AlternatingLeastSquares()
    model.fit(train)

    # validation plugs straight into the real evaluation calls
    metrics = evaluate_time_based_split(model, train, validation, K=10)

Reference
---------

.. automodule:: implicit.backtest
   :members:
   :show-inheritance:
