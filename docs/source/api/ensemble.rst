Merging Candidates Across Models
================================

This library supports merging recommendation candidates from models that were
trained independently for different regions -- each with its own interaction
data, inventory and local item id space -- into a single ranked list for
cross-region accounts.

Score scales are made comparable by fitting a calibrator per model on a shared
set of historical validation requests, and candidates are merged by their
global business item id so that colliding local integer ids are never confused.

RankCalibrator
--------------
.. autoclass:: implicit.ensemble.RankCalibrator
   :members:

CandidateMerger
---------------
.. autoclass:: implicit.ensemble.CandidateMerger
   :members:
