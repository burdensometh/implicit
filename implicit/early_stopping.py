"""Hold-out evaluation and early stopping logic shared by the ALS models.

The training loss of ALS keeps decreasing with extra iterations even after the
ranking quality on withheld interactions has started to degrade. The
:class:`EarlyStopping` class evaluates the model on a held-out validation set at
a fixed interval while fitting, remembers the factors from the best validation
round, and (optionally) stops training once the validation score has stopped
improving for a patience window.
"""

import logging

import numpy as np
from scipy.sparse import csr_matrix

log = logging.getLogger("implicit")

# the metrics computed by implicit.evaluation.ranking_metrics_at_k
VALIDATION_METRICS = ("precision", "map", "ndcg", "auc")


def snapshot_factors(factors):
    """Returns an independent copy of a factor matrix.

    Works for both numpy arrays (CPU models) and implicit.gpu.Matrix instances
    (GPU models). The returned copy shares no storage with ``factors``, so it is
    safe to keep while the model's factors keep being updated in place.
    """
    # gpu factor matrices expose a to_numpy conversion method
    if hasattr(factors, "to_numpy"):
        import implicit.gpu

        return implicit.gpu.Matrix(factors.to_numpy())
    return factors.copy()


class EarlyStopping:
    """Tracks validation metrics during ``fit`` and retains the best factors.

    Parameters
    ----------
    model : MatrixFactorizationBase
        The model being fitted. Its ``user_factors`` and ``item_factors`` are
        snapshotted whenever the validation score improves.
    train_user_items : csr_matrix
        The training matrix, passed to the ranking metrics so that items already
        seen by a user are filtered out of the validation recommendations.
    validation_user_items : csr_matrix
        The held-out matrix of interactions to evaluate against. Must have the
        same shape as ``train_user_items``.
    metric : str, optional
        Which validation metric to optimize. One of "precision", "map", "ndcg"
        or "auc". Defaults to "ndcg".
    K : int, optional
        Number of recommendations to compute per user when scoring.
    patience : int, optional
        Number of consecutive validation rounds without improvement tolerated
        before training is stopped. ``None`` disables early stopping, while the
        best round is still tracked and restored in either case.
    validate_every : int, optional
        Evaluate the validation set once every ``validate_every`` ALS
        iterations.
    show_progress : bool, optional
        Whether to show a progress bar while evaluating.
    num_threads : int, optional
        Number of threads to use for the CPU evaluation. Ignored by GPU models.

    Attributes
    ----------
    history : list[dict]
        One dictionary per validation round, mapping ``iteration`` and every
        metric name to the values observed in that round.
    best_iteration : int, optional
        The zero-indexed ALS iteration that produced the best validation score,
        or ``None`` if no validation round has run yet.
    best_score : float, optional
        The best value observed for the optimized metric.
    best_metrics : dict, optional
        The full set of metrics measured in the best round.
    """

    def __init__(
        self,
        model,
        train_user_items,
        validation_user_items,
        metric="ndcg",
        K=10,
        patience=None,
        validate_every=1,
        show_progress=False,
        num_threads=0,
    ):
        if metric not in VALIDATION_METRICS:
            raise ValueError(
                f"Unknown validation metric '{metric}'. "
                f"Must be one of {', '.join(VALIDATION_METRICS)}."
            )
        if not isinstance(validation_user_items, csr_matrix):
            raise ValueError("validation_user_items needs to be a CSR sparse matrix")
        if not isinstance(train_user_items, csr_matrix):
            raise ValueError("train_user_items needs to be a CSR sparse matrix")
        if validation_user_items.shape != train_user_items.shape:
            raise ValueError(
                "validation_user_items must have the same shape as the training "
                f"matrix: got {validation_user_items.shape} vs {train_user_items.shape}"
            )
        if not isinstance(K, (int, np.integer)) or K < 1:
            raise ValueError("validation_K must be a positive integer")
        if not isinstance(validate_every, (int, np.integer)) or validate_every < 1:
            raise ValueError("validate_every must be a positive integer")
        if patience is not None and (not isinstance(patience, (int, np.integer)) or patience < 1):
            raise ValueError("early_stopping_patience must be a positive integer or None")

        self.model = model
        self.train_user_items = train_user_items
        # ranking_metrics_at_k only inspects the held-out item positions, but normalize
        # the dtype anyway so GPU/CPU recommend calls receive consistent inputs
        if validation_user_items.dtype != np.float32:
            validation_user_items = validation_user_items.astype(np.float32)
        self.validation_user_items = validation_user_items
        self.metric = metric
        self.K = int(K)
        self.patience = None if patience is None else int(patience)
        self.validate_every = int(validate_every)
        self.show_progress = show_progress
        self.num_threads = num_threads

        self.history = []
        self.best_iteration = None
        self.best_score = None
        self.best_metrics = None
        self.rounds_without_improvement = 0

        self._best_user_factors = None
        self._best_item_factors = None

    def should_evaluate(self, iteration):
        """Whether the validation set must be evaluated after this iteration."""
        return (iteration + 1) % self.validate_every == 0

    def evaluate(self):
        """Computes the full set of ranking metrics on the validation set."""
        # imported lazily to avoid pulling in the compiled evaluation module
        # whenever early stopping isn't used
        from .evaluation import ranking_metrics_at_k

        return ranking_metrics_at_k(
            self.model,
            self.train_user_items,
            self.validation_user_items,
            K=self.K,
            show_progress=self.show_progress,
            num_threads=self.num_threads,
        )

    def step(self, iteration):
        """Runs a validation round for this iteration, if one is due.

        Returns ``True`` when the patience window has been exhausted and
        training should stop.
        """
        if not self.should_evaluate(iteration):
            return False

        metrics = self.evaluate()
        record = {"iteration": int(iteration)}
        record.update(metrics)
        self.history.append(record)

        score = metrics[self.metric]
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_iteration = int(iteration)
            self.best_metrics = dict(metrics)
            self.rounds_without_improvement = 0

            # remember the factors that produced this best score
            self._best_user_factors = snapshot_factors(self.model.user_factors)
            self._best_item_factors = snapshot_factors(self.model.item_factors)
        else:
            self.rounds_without_improvement += 1

        log.info(
            "validation at iteration %i: %s (best %s=%.6f at iteration %s)",
            iteration,
            metrics,
            self.metric,
            self.best_score,
            self.best_iteration,
        )

        return (
            self.patience is not None
            and self.rounds_without_improvement >= self.patience
        )

    def restore_best(self):
        """Replaces the model's factors with the best validated snapshots.

        All factor-derived caches (norms, Gram matrices) are invalidated, so
        that recommendations, similarity lookups, explanations and user/item
        recalculations all use the restored factors consistently.
        """
        if self._best_user_factors is None:
            # no validation round ever ran (e.g. iterations=0): nothing to restore
            return

        self.model.user_factors = self._best_user_factors
        self.model.item_factors = self._best_item_factors

        # invalidate every cache derived from the factor matrices
        self.model._user_norms = None
        self.model._item_norms = None
        self.model._XtX = None
        self.model._YtY = None

        # gpu models additionally cache norms on the host
        if hasattr(self.model, "_user_norms_host"):
            self.model._user_norms_host = None
            self.model._item_norms_host = None


HISTORY_KEY_PREFIX = "validation_history_"


def add_validation_state(npz_args, model):
    """Serializes the per-round validation history into numpy npz arguments.

    Only plain numpy arrays are stored, so the model remains loadable with
    ``np.load(..., allow_pickle=False)``.
    """
    history = getattr(model, "validation_history", None)
    if not history:
        return

    npz_args[HISTORY_KEY_PREFIX + "iterations"] = np.array(
        [record["iteration"] for record in history], dtype=np.int64
    )
    for metric in VALIDATION_METRICS:
        npz_args[HISTORY_KEY_PREFIX + metric] = np.array(
            [record[metric] for record in history], dtype=np.float64
        )

    if getattr(model, "best_iteration", None) is not None:
        npz_args["best_iteration"] = np.int64(model.best_iteration)


def restore_validation_state(model, data):
    """Rebuilds ``validation_history`` and ``best_validation_metrics`` from npz data.

    Also removes the raw per-metric arrays that the generic npz loader would
    otherwise leave as unrelated attributes on the model.
    """
    keys = set(data.keys())

    for key in list(keys):
        if key.startswith(HISTORY_KEY_PREFIX) and hasattr(model, key):
            delattr(model, key)

    if HISTORY_KEY_PREFIX + "iterations" not in keys:
        model.validation_history = None
        model.best_iteration = None
        model.best_validation_metrics = None
        return

    iterations = data[HISTORY_KEY_PREFIX + "iterations"]
    history = []
    for i, iteration in enumerate(iterations):
        record = {"iteration": int(iteration)}
        for metric in VALIDATION_METRICS:
            record[metric] = float(data[HISTORY_KEY_PREFIX + metric][i])
        history.append(record)
    # normalize: no recorded rounds is the same as never having validated
    model.validation_history = history or None

    if "best_iteration" in keys:
        model.best_iteration = int(data["best_iteration"].item())
        model.best_validation_metrics = next(
            {key: value for key, value in record.items() if key != "iteration"}
            for record in history
            if record["iteration"] == model.best_iteration
        )
    else:
        model.best_iteration = None
        model.best_validation_metrics = None
