Implicit
=======

[![Build
Status](https://github.com/benfred/implicit/workflows/Build/badge.svg)](https://github.com/benfred/implicit/actions?query=workflow%3ABuild+branch%3Amain)
[![Documentation](https://img.shields.io/badge/documentation-blue.svg)](https://benfred.github.io/implicit/)


Fast Python Collaborative Filtering for Implicit Datasets.

This project provides fast Python implementations of several different popular recommendation algorithms for
implicit feedback datasets:

 * Alternating Least Squares as described in the papers [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) and [Applications of the Conjugate Gradient Method for Implicit
Feedback Collaborative Filtering](https://pdfs.semanticscholar.org/bfdf/7af6cf7fd7bb5e6b6db5bbd91be11597eaf0.pdf).

 * [Bayesian Personalized Ranking](https://arxiv.org/pdf/1205.2618.pdf).

 * [Logistic Matrix Factorization](https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)

 * Item-Item Nearest Neighbour models using Cosine, TFIDF or BM25 as a distance metric.

All models have multi-threaded training routines, using Cython and OpenMP to fit the models in
parallel among all available CPU cores.  In addition, the ALS and BPR models both have custom CUDA
kernels - enabling fitting on compatible GPU's. Approximate nearest neighbours libraries such as [Annoy](https://github.com/spotify/annoy), [NMSLIB](https://github.com/searchivarius/nmslib)
and [Faiss](https://github.com/facebookresearch/faiss) can also be used by Implicit to [speed up
making recommendations](https://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/).

#### Installation

Implicit can be installed from pypi with:

```
pip install implicit
```

Installing with pip will use prebuilt binary wheels on x86_64 Linux, Windows
and OSX. These wheels include GPU support on Linux.

Implicit can also be installed with conda:

```
# CPU only package
conda install -c conda-forge implicit

# CPU+GPU package
conda install -c conda-forge implicit implicit-proc=*=gpu
```

#### Basic Usage

```python
import implicit

# initialize a model
model = implicit.als.AlternatingLeastSquares(factors=50)

# train the model on a sparse matrix of user/item/confidence weights
model.fit(user_item_data)

# recommend items for a user
recommendations = model.recommend(userid, user_item_data[userid])

# find related items
related = model.similar_items(itemid)
```

The examples folder has a program showing how to use this to [compute similar artists on the
last.fm dataset](https://github.com/benfred/implicit/blob/master/examples/lastfm.py).

For more information see the [documentation](https://benfred.github.io/implicit/).

#### Building a feedback matrix from events

Raw event streams (page views, carts, purchases, rejections) need to be converted into the
`(users, items)` confidence matrix the models are trained on. `implicit.feedback` implements a
reproducible event-to-feedback path: events are deduplicated by event id (so receiver replays and
at-least-once deliveries collapse), conflicting duplicates are resolved by an explicit policy,
each event is weighted by behaviour type and decayed by its age against a training cutoff, and
positive/negative evidence for the same user/item pair is combined with a configurable
cancellation rule before the matrix is handed to `fit`.

```python
from datetime import datetime, timedelta
import implicit
from implicit.feedback import FeedbackConfig, build_feedback_matrix

config = FeedbackConfig(
    training_cutoff=datetime(2024, 6, 1),
    event_weights={"view": 1.0, "cart": 3.0, "purchase": 5.0, "reject": -6.0},
    half_life={"view": timedelta(days=2), "cart": timedelta(days=7),
               "purchase": timedelta(days=90), "reject": timedelta(days=90)},
    cancellation_policy="net",
)

feedback = build_feedback_matrix(events, config)  # events: list of dicts or column arrays
model = implicit.als.AlternatingLeastSquares(factors=50)
feedback.fit(model)                               # trains ALS on the generated matrix

# map model indices back to the original ids
row = feedback.user_index["alice"]
items = feedback.item_ids
```

The resulting matrix is deterministic for a given event set, independent of input order. Set
`cancellation_policy="rejection_wins"` if a purchase followed by a return should always be a
negative, or `"latest_wins"` if the most recent event should decide the sign. Events after the
training cutoff are dropped by default to avoid leaking future information; see the API docs for
the full set of options.

#### Articles about Implicit

These blog posts describe the algorithms that power this library:

 * [Finding Similar Music with Matrix Factorization](https://www.benfrederickson.com/matrix-factorization/)
 * [Faster Implicit Matrix Factorization](https://www.benfrederickson.com/fast-implicit-matrix-factorization/)
 * [Implicit Matrix Factorization on the GPU](https://www.benfrederickson.com/implicit-matrix-factorization-on-the-gpu/)
 * [Approximate Nearest Neighbours for Recommender Systems](https://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/)
 * [Distance Metrics for Fun and Profit](https://www.benfrederickson.com/distance-metrics/)

There are also several other articles about using Implicit to build recommendation systems:
 * [H&M Personalized Fashion Recommendations Kaggle Competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324129)
 * [Yandex Cup 2022: Like Prediction](https://github.com/greenwolf-nsk/yandex-cup-2022-recsys)
 * [Recommending GitHub Repositories with Google BigQuery and the implicit library](https://medium.com/@jbochi/recommending-github-repositories-with-google-bigquery-and-the-implicit-library-e6cce666c77)
 * [Intro to Implicit Matrix Factorization: Classic ALS with Sketchfab Models](http://blog.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/)
 * [A Gentle Introduction to Recommender Systems with Implicit Feedback](https://jessesw.com/Rec-System/).


#### Requirements

This library requires SciPy version 0.16 or later and Python version 3.9 or later.

GPU Support requires version 13 of the [NVidia CUDA
Toolkit](https://developer.nvidia.com/cuda-downloads), and requires that RMM be installed `pip install rmm-cu13`

This library is tested with Python 3.9 to 3.14 on Ubuntu, OSX and Windows.

#### Benchmarks

Simple benchmarks comparing the ALS fitting time versus [Spark can be found here](https://github.com/benfred/implicit/tree/master/benchmarks).

#### Optimal Configuration

I'd recommend configuring SciPy to use Intel's MKL matrix libraries. One easy way of doing this is by installing the Anaconda Python distribution.

For systems using OpenBLAS, I highly recommend setting 'export OPENBLAS_NUM_THREADS=1'. This
disables its internal multithreading ability, which leads to substantial speedups for this
package. Likewise for Intel MKL, setting 'export MKL_NUM_THREADS=1' should also be set.

Released under the MIT License
