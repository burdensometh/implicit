"""Utilities for merging recommendation candidates from multiple models.

This module targets the serving scenario where each region trains its own model
on its own interaction data (e.g. dwell time) and its own inventory, and the
serving layer needs to merge candidates from two regional models for
cross-region accounts.

Since each regional model is trained independently, their raw scores are not
directly comparable, and their local integer item ids live in different index
spaces (local id ``7`` in region A is generally a different item than local id
``7`` in region B). The typical workflow is:

1. For each regional model, fit a :class:`RankCalibrator` on the model's scores
   for a shared set of historical validation requests, so that raw scores from
   different models are mapped to a common, comparable scale.
2. Register each model with a :class:`CandidateMerger`, along with the mapping
   from the model's local item ids to global business item ids.
3. Call :meth:`CandidateMerger.merge` with the ``recommend`` output of each
   model to get a single ranked list of business item ids, where every entry
   records which models contributed it.
"""

import numpy as np

__all__ = ["RankCalibrator", "CandidateMerger"]


class RankCalibrator:
    """Learns a monotone mapping from a single model's raw scores to a common scale.

    The mapping is estimated from the model's score distribution on a shared
    set of historical validation requests (e.g. requests from cross-region
    accounts that every regional model can score). After fitting each regional
    model's calibrator on the same validation traffic, the transformed scores
    are on a comparable scale and can be merged across models.

    Parameters
    ----------
    method : str, optional
        The calibration method to use:

        - 'quantile' (default): maps scores to their empirical quantile in the
          validation score distribution, producing comparable values in [0, 1].
        - 'zscore': subtracts the validation mean and divides by the validation
          standard deviation, giving each model zero mean and unit variance on
          the validation traffic.
    """

    METHODS = ("quantile", "zscore")

    def __init__(self, method="quantile"):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.method = method

    def fit(self, scores):
        """Learns the calibration from a sample of the model's raw scores.

        Parameters
        ----------
        scores : array_like
            Raw scores produced by the model on the shared validation requests.

        Returns
        -------
        self
        """
        scores = np.asarray(scores, dtype=np.float64).ravel()
        if scores.size == 0:
            raise ValueError("cannot fit a calibrator on an empty score sample")
        if self.method == "quantile":
            self.reference_ = np.sort(scores)
        else:
            self.mean_ = scores.mean()
            self.std_ = scores.std()
            if self.std_ == 0.0:
                self.std_ = 1.0
        return self

    def transform(self, scores):
        """Maps raw scores onto the common scale.

        Parameters
        ----------
        scores : array_like
            Raw model scores to calibrate.

        Returns
        -------
        ndarray
            The calibrated scores, with the same shape as the input.
        """
        if not hasattr(self, "reference_") and not hasattr(self, "mean_"):
            raise ValueError("calibrator is not fitted yet")
        scores = np.asarray(scores, dtype=np.float64)
        if self.method == "quantile":
            ranks = np.searchsorted(self.reference_, scores.ravel(), side="right")
            return (ranks / self.reference_.size).reshape(scores.shape)
        return (scores - self.mean_) / self.std_

    def fit_transform(self, scores):
        """Fits the calibrator on ``scores`` and returns them calibrated."""
        return self.fit(scores).transform(scores)


class CandidateMerger:
    """Merges recommendation candidates from several regional models.

    Each model is registered with a calibrator (learned from shared validation
    requests) and a mapping from its local integer item ids to global business
    item ids. Merging is keyed on the business item id, so local ids that
    collide across models are never confused with each other. Items returned by
    only some of the models are kept, and every merged candidate records which
    models recommended it.

    Parameters
    ----------
    method : str, optional
        Calibration method passed to :class:`RankCalibrator` when a model is
        registered with validation scores instead of a fitted calibrator.
    combine : str, optional
        How to combine the calibrated scores of a business item recommended by
        multiple models: 'sum' (default), 'mean' or 'max'.
    on_unknown : str, optional
        What to do when a model returns a local item id that has no business
        item id in its mapping: 'drop' (default) silently skips it, 'raise'
        raises a KeyError.
    """

    COMBINE_METHODS = ("sum", "mean", "max")

    def __init__(self, method="quantile", combine="sum", on_unknown="drop"):
        if combine not in self.COMBINE_METHODS:
            raise ValueError(f"combine must be one of {self.COMBINE_METHODS}")
        if on_unknown not in ("drop", "raise"):
            raise ValueError("on_unknown must be 'drop' or 'raise'")
        self.method = method
        self.combine = combine
        self.on_unknown = on_unknown
        self.calibrators = {}
        self.item_mappings = {}

    def add_model(self, name, item_id_mapping, validation_scores=None, calibrator=None):
        """Registers a regional model with the merger.

        Parameters
        ----------
        name : str
            Name of the model (e.g. the region). Used to report which models
            contributed each merged candidate.
        item_id_mapping : array_like or dict
            Mapping from the model's local integer item ids to global business
            item ids. When array-like, ``item_id_mapping[local_id]`` must be
            the business item id; when a dict, only the local ids present as
            keys are known.
        validation_scores : array_like, optional
            Raw scores of this model on the shared historical validation
            requests. Used to fit a :class:`RankCalibrator` for this model.
        calibrator : RankCalibrator, optional
            An already fitted calibrator to use instead of fitting one from
            ``validation_scores``. When neither is passed, an unfitted
            calibrator is created that must be fitted later, e.g. with
            :meth:`fit_from_requests`.

        Returns
        -------
        self
        """
        if calibrator is None:
            calibrator = RankCalibrator(self.method)
            if validation_scores is not None:
                calibrator.fit(validation_scores)
        self.calibrators[name] = calibrator
        self.item_mappings[name] = item_id_mapping
        return self

    def fit_from_requests(self, name, model, user_items, userids, N=100, **recommend_kwargs):
        """Fits this model's calibrator from shared historical validation requests.

        Runs ``model.recommend`` for the given validation users and learns the
        score calibration from the returned raw scores. The userids are the
        rows of the shared validation accounts in this model's own user/index
        space; call this once per regional model with the same validation
        accounts to get comparable scales.

        Parameters
        ----------
        name : str
            Name of a model already registered with :meth:`add_model`.
        model : RecommenderBase
            The fitted regional model.
        user_items : csr_matrix
            The user/item matrix of the regional model.
        userids : array_like
            The validation user ids, in this model's local user index space.
        N : int, optional
            The number of items scored per validation request.
        recommend_kwargs
            Extra keyword arguments are passed to ``model.recommend``.

        Returns
        -------
        self
        """
        if name not in self.calibrators:
            raise KeyError(f"unknown model {name!r}: register it with add_model first")
        userids = np.asarray(userids)
        _, scores = model.recommend(userids, user_items[userids], N=N, **recommend_kwargs)
        self.calibrators[name].fit(np.ravel(scores))
        return self

    def merge(self, recommendations, N=10, weights=None):
        """Merges the candidates of all registered models for a single request.

        Parameters
        ----------
        recommendations : dict
            Maps a model name to its ``(ids, scores)`` output of
            ``model.recommend`` for one request. Ids are the model's local
            integer item ids; they are translated to business item ids with the
            mapping registered for that model, so identical local ids from
            different models are never merged together.
        N : int, optional
            The maximum number of merged candidates to return.
        weights : dict, optional
            Per-model weights applied to the calibrated scores before
            combining. Models not present in the dict get weight 1.0.

        Returns
        -------
        tuple
            A tuple of ``(item_ids, scores, sources)``: the business item ids
            sorted by descending merged score, the merged scores, and a list
            with one tuple of contributing model names per returned item.
        """
        # business id -> [weighted score sum, weight sum, max score, sources]
        combined = {}
        order = []
        for name, (ids, scores) in recommendations.items():
            if name not in self.calibrators:
                raise KeyError(f"unknown model {name!r}: register it with add_model first")
            ids = np.asarray(ids, dtype=np.int64)
            scores = np.asarray(scores, dtype=np.float64)
            if ids.ndim != 1 or scores.ndim != 1:
                raise ValueError("merge handles a single request; use merge_batch instead")
            if ids.shape[0] != scores.shape[0]:
                raise ValueError("ids and scores must have the same length")

            business_ids, known = self._to_business_ids(name, ids)
            calibrated = self.calibrators[name].transform(scores)[known]
            weight = 1.0 if weights is None else weights.get(name, 1.0)

            for business_id, score in zip(business_ids, calibrated):
                entry = combined.get(business_id)
                if entry is None:
                    combined[business_id] = [weight * score, weight, score, [name]]
                    order.append(business_id)
                else:
                    entry[0] += weight * score
                    entry[1] += weight
                    entry[2] = max(entry[2], score)
                    entry[3].append(name)

        merged_ids = np.asarray(order)
        if self.combine == "sum":
            merged_scores = np.array([combined[b][0] for b in order])
        elif self.combine == "mean":
            merged_scores = np.array([combined[b][0] / combined[b][1] for b in order])
        else:
            merged_scores = np.array([combined[b][2] for b in order])

        # sort by descending score, keeping first-seen order on ties
        top = np.argsort(-merged_scores, kind="stable")[:N]
        sources = [tuple(combined[b][3]) for b in merged_ids[top]]
        return merged_ids[top], merged_scores[top], sources

    def merge_batch(self, recommendations, N=10, weights=None):
        """Merges the candidates of all registered models for a batch of requests.

        Parameters
        ----------
        recommendations : dict
            Maps a model name to its ``(ids, scores)`` output of
            ``model.recommend`` for a batch of users. All models must have the
            same number of rows (requests).
        N : int, optional
            The maximum number of merged candidates to return per request.
        weights : dict, optional
            Per-model weights applied to the calibrated scores before
            combining.

        Returns
        -------
        list
            A list with one ``(item_ids, scores, sources)`` tuple per request,
            as returned by :meth:`merge`.
        """
        rows = None
        arrays = {}
        for name, (ids, scores) in recommendations.items():
            ids = np.asarray(ids, dtype=np.int64)
            scores = np.asarray(scores, dtype=np.float64)
            if ids.ndim != 2 or scores.ndim != 2:
                raise ValueError("merge_batch expects 2-dimensional ids and scores")
            if rows is None:
                rows = ids.shape[0]
            elif ids.shape[0] != rows:
                raise ValueError("all models must have the same number of rows")
            arrays[name] = (ids, scores)

        results = []
        for row in range(rows or 0):
            single = {name: (ids[row], scores[row]) for name, (ids, scores) in arrays.items()}
            results.append(self.merge(single, N=N, weights=weights))
        return results

    def _to_business_ids(self, name, ids):
        """Translates local item ids to business item ids for one model.

        Local ids without a known business id are dropped or raise a KeyError,
        depending on the ``on_unknown`` setting.
        """
        mapping = self.item_mappings[name]
        if isinstance(mapping, dict):
            known = np.array([i in mapping for i in ids])
            business_ids = np.array([mapping.get(i) for i in ids], dtype=object)
        else:
            mapping = np.asarray(mapping)
            known = (ids >= 0) & (ids < len(mapping))
            business_ids = np.empty(len(ids), dtype=object)
            business_ids[known] = mapping[ids[known]]

        if not known.all():
            if self.on_unknown == "raise":
                unknown = ids[~known]
                raise KeyError(
                    f"model {name!r} returned local item ids with no business id: "
                    f"{unknown.tolist()}"
                )
        return business_ids[known], known
