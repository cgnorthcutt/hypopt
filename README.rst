
``hypopt``
================

|pypi| |py_versions| |build_status| |coverage|

.. |pypi| image:: https://img.shields.io/pypi/v/hyperopt.svg
    :target: https://pypi.org/pypi/hypopt/
.. |py_versions| image:: https://img.shields.io/pypi/pyversions/hypopt.svg
    :target: https://pypi.org/pypi/hypopt/
.. |build_status| image:: https://travis-ci.com/cgnorthcutt/hypopt.svg?branch=master
    :target: https://travis-ci.com/cgnorthcutt/hypopt
.. |coverage| image:: https://codecov.io/gh/cgnorthcutt/hypopt/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/cgnorthcutt/hypopt
    
    

A Python machine learning package for grid search **hyper-parameter optimization using a validation set** (defaults to cross validation when no validation set is available). This package works for Python 2.7+ and Python 3+, for any model (classification and regression), and **runs in parallel on all threads on your CPU automatically**.

``scikit-learn`` provides a package for `grid-search hyper-parameter optimization **using cross-validation** on the training dataset <http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV>`_. Unfortunately, cross-validation is impractically slow for large datasets and fails for small datasets due to the lack of data in each class needed to properly train each fold. Instead, we use a constant validation set to optimize hyper-parameters -- the ``hypopt`` package makes this fast (distributed on all CPU threads) and easy (one line of code).

``hypopt.model_selection.fit_model_with_grid_search`` supports grid search hyper-parameter optimization **when you already have a validation set** , eliminating the extra hours of training time required when using cross-validation. However, when no validation set is given, it defaults to using cross-validation on the training set. This allows you to alows use ``hypopt`` anytime you need to do hyper-parameter optimization with grid-search, regardless of whether you use a validation set or cross-validation.

Installation
------------

Python 2.7, 3.4, 3.5, and 3.6 are supported.

Stable release:

.. code-block::

   $ pip install hypopt

Developer (unstable) release:

.. code-block::

   $ pip install git+https://github.com/cgnorthcutt/hypopt.git

To install the codebase (enabling you to make modifications):

.. code-block::

   $ conda update pip # if you use conda
   $ git clone https://github.com/cgnorthcutt/hypopt.git
   $ cd hypopt
   $ pip install -e .

Examples
--------

Basic usage
^^^^^^^^^^^

.. code-block:: python

   # Assuming you already have train, test, val sets and a model.
   from hypopt import GridSearch
   param_grid = [
     {'C': [1, 10, 100], 'kernel': ['linear']},
     {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
   # Grid-search all parameter combinations using a validation set.
   gs = GridSearch(model = SVR(), param_grid = param_grid)
   gs.fit(X_train, y_train, X_val, y_val)
   print('Test Score for Optimized Parameters:', gs.score(X_test, y_test))
   
Choosing the scoring metric to optimize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default metric is the the ``model.score()`` function, so in the previous example ``SVR().score()`` is optimized, which defaults to accuracy.

It's easy to use a different scoring metric using the ``scoring`` parameter in ``hypopt.GridSearch.fit()``:

.. code-block:: python

    # This will use f1 score as the scoring metric that you optimize.
    gs.fit(X_train, y_train, X_val, y_val, scoring='f1')

* For classification, `hypopt` supports these string-named metrics: 'accuracy', 'brier_score_loss', 'average_precision', 'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'neg_log_loss', 'precision', 'recall', or 'roc_auc'. 
* For regression, `hypopt` supports: "explained_variance", "neg_mean_absolute_error", "neg_mean_squared_error", "neg_mean_squared_log_error", "neg_median_absolute_error", "r2".

You can also create your own metric ``your_custom_score_func(y_true, y_pred)`` by wrapping it into an object using `sklearn.metrics.make_scorer <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html>`_ like:

.. code-block:: python

    from sklearn.metrics import make_scorer
    scorer = make_scorer(your_custom_scoring_func)
    opt.fit(X_train, y_train, X_val, y_val, scoring=scorer)

Minimal working examples
^^^^^^^^^^^^^^^^^^^^^^^^


* `Classification minimal working example <https://github.com/cgnorthcutt/hypopt/blob/master/examples/classification_minimal_working_example.ipynb>`_
* `Regression minimal working example <https://github.com/cgnorthcutt/hypopt/blob/master/examples/regression_minimal_working_example.ipynb>`_

Other Examples including a working example with MNIST
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* `A simple tutorial of the hypopt package. <https://github.com/cgnorthcutt/hypopt/blob/master/examples/simple_tutorial.ipynb>`_
* `A working example with MNIST. <https://github.com/cgnorthcutt/hypopt/blob/master/examples/mnist_example.ipynb>`_

Use ``hypopt`` with any model (PyTorch, Tensorflow, caffe2, scikit-learn, etc.)
-------------------------------------------------------------------------------------

All of the features of the ``hypopt`` package work with **any model**. Yes, any model. Feel free to use PyTorch, Tensorflow, caffe2, scikit-learn, mxnet, etc. If you use a scikit-learn model, all ``hypopt`` methods will work out-of-the-box. It's also easy to use your favorite model from a non-scikit-learn package, just wrap your model into a Python class that inherets the ``sklearn.base.BaseEstimator``. Here's an example for a generic classifier:

.. code-block:: python

   from sklearn.base import BaseEstimator
   class YourModel(BaseEstimator): # Inherits sklearn base classifier
       def __init__(self, ):
           pass
       def fit(self, X, y, sample_weight = None):
           pass
       def predict(self, X):
           pass
       def score(self, X, y, sample_weight = None):
           pass

       # Inherting BaseEstimator gives you these for free!
       # So if you inherit, there's no need to implement these.
       def get_params(self, deep = True):
           pass
       def set_params(self, **params):
           pass
           
PyTorch MNIST CNN Example
^^^^^^^^^^^^^^^^^^^^^^^^^

Check out a PyTorch MNIST CNN wrapped in the above class `here <https://github.com/cgnorthcutt/cleanlab/blob/master/cleanlab/models/mnist_pytorch.py#L28>`__. You use any object instantion of this class with `hypopt` just as you would any scikit-learn model. Another example of a fully compliant class is the `LearningWithNoisyLabels() model <https://github.com/cgnorthcutt/cleanlab/blob/master/cleanlab/classification.py#L48>`__.

If you don't wish to write this code yourself, there are existing packages to do this for you. For PyTorch, check out
the ``skorch`` `Python package <https://skorch.readthedocs.io/en/stable/>` which will wrap your ``pytorch`` model
into a ``scikit-learn`` compliant model.
