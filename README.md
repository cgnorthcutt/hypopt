# `hyperopt`
A Python machine learning package for grid search hyper-parameter optimization of a classifier using a validation set (not cross validation).

`scikit-learn` implements a [package for grid search hyper-parameter optimization **using cross-validation** on the training dataset](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV). Unfortunately, cross-validation is extremely slow! If your classifier takes 1 hour to train, then with cross-validation and 10 folds, it would take you 9 hours just to try only one (of potentially hundreds) setting of parameters! This won't do.

`hyperopt` fills a much needed gap for Python and the `scikit-learn` community by supporting grid search hyper-parameter optimization **when you already have a validation set**, eliminating the extra 8 hours of training time required when using cross-validation. 

This package works for Python 2.7+ and Python 3+, for any model, and importantly, **runs in parallel on all threads on your CPU automatically**.

## Use `hyperopt` with any model (PyTorch, Tensorflow, caffe2, scikit-learn, mxnet, etc.)
All of the features of the `hyperopt` package work with **any model**. Yes, any model. Feel free to use PyTorch, Tensorflow, caffe2, scikit-learn, mxnet, etc. If you use a scikit-learn model, all `hyperopt` methods will work out-of-the-box. It's also easy to use your favorite model from a non-scikit-learn package, just wrap your model into a Python class that inherets the `sklearn.base.BaseEstimator`. Here's an example for a generic classifier:
```python
from sklearn.base import BaseEstimator
class YourModel(BaseEstimator): # Inherits sklearn base classifier
    def __init__(self, ):
        pass
    def fit(self, X, y, sample_weight = None):
        pass
    def predict(self, X):
        pass
    def predict_proba(self, X):
        pass
    def score(self, X, y, sample_weight = None):
        pass
```

## Installation

Python 2.7 and Python 3.5 are supported.

To install the `hyperopt` package with pip, just run:

```
$ pip install git+https://github.com/cgnorthcutt/hyperopt.git
```

If you have issues, you can also clone the repo and install:

```
$ conda update pip # if you use conda
$ git clone https://github.com/cgnorthcutt/hyperopt.git
$ cd hyperopt
$ pip install -e .
```
