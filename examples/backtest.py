"""A time based (backtest) evaluation example.

Randomly holding out sparse matrix entries for evaluation lets the model see
interactions that happened after the ones it is scored on, which inflates the
metrics.  This example instead builds a small timestamped event log, splits it
on a single global cutoff, and evaluates ALS on the post-cutoff targets that
were unseen at the cutoff.

Run with::

    python examples/backtest.py
"""

import logging

import numpy as np

from implicit.als import AlternatingLeastSquares
from implicit.backtest import evaluate_time_based_split, time_based_split


def generate_events(num_users=500, num_items=200, num_events=20000, seed=0):
    """Returns a synthetic event log as a dict of columns.

    Each user has a preferred cluster of items and buys from it over time; some
    events are non-positive (e.g. a refund) and are ignored by the split.  A
    handful of "new" accounts only appear after the cutoff to show how accounts
    without enough history are reported.
    """
    rng = np.random.default_rng(seed)
    user_pref = rng.integers(0, 10, size=num_users)

    users = rng.integers(0, num_users, size=num_events)
    item_cluster = rng.integers(0, 10, size=num_events)
    # most purchases come from the user's preferred cluster
    from_pref = rng.random(num_events) < 0.7
    clusters = np.where(from_pref, user_pref[users], item_cluster)
    items = clusters * (num_items // 10) + rng.integers(0, num_items // 10, size=num_events)

    timestamps = rng.integers(0, 100, size=num_events)
    positive = rng.random(num_events) > 0.05

    # append a few brand new accounts that only start after the cutoff
    new_users = num_users + np.arange(50)
    late_users = np.repeat(new_users, 20)
    late_items = rng.integers(0, num_items, size=len(late_users))
    late_timestamps = rng.integers(90, 100, size=len(late_users))

    users = np.concatenate([users, late_users])
    items = np.concatenate([items, late_items])
    timestamps = np.concatenate([timestamps, late_timestamps])
    positive = np.concatenate([positive, np.ones(len(late_users), dtype=bool)])

    return {
        "user": users,
        "item": items,
        "timestamp": timestamps,
        "value": np.ones(len(users)),
        "positive": positive,
    }


def main():
    logging.basicConfig(level=logging.INFO)
    events = generate_events()

    # split on a single global cutoff; every training event happens before every
    # validation target
    train, validation, report = time_based_split(
        events, cutoff=80, min_history=3, return_report=True
    )
    print(report.summary())
    print(f"{len(report.insufficient_history_users)} accounts had too little history")
    # accounts that could not be evaluated are listed (and their targets dropped)
    print(f"accounts with no validation targets: {report.users_without_validation[:10]}")

    model = AlternatingLeastSquares(factors=32, regularization=0.05, iterations=10)
    model.fit(train, show_progress=False)

    # validation plugs straight into the regular ranking metrics
    metrics = evaluate_time_based_split(model, train, validation, K=10, show_progress=False)
    for name, value in metrics.items():
        print(f"{name}@10 = {value:.4f}")


if __name__ == "__main__":
    main()
