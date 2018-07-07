
# coding: utf-8

# In[ ]:


from __future__ import print_function, absolute_import, division, unicode_literals, with_statement
from sklearn.model_selection import train_test_split
from hypopt import GridSearch
from sklearn.svm import SVC
from unittest import TestCase


# In[ ]:


from sklearn.base import BaseEstimator
class model_without_score(BaseException):
    '''Simple test class of an sklearn model without score function'''

    def __init__(self, model):
        self.model = model
        self.seed = 0
    
    def fit(self, X, y, sample_weight = None):
        return self.model.fit(X, y, sample_weight=sample_weight)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep = True):
        return self.model.get_params(deep = deep)
    
    def set_params(self, **params):
        return self.model.set_params(**params)      


# In[ ]:


from sklearn.base import BaseEstimator
class model_that_throws_exception(BaseException):
    '''Simple test class of an sklearn model without score function'''

    def __init__(self, model):
        self.model = model
        self.seed = 0
    
    def fit(self, X, y, sample_weight = None):
        raise ValueError('Generic Exception for testing purposes.')
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep = True):
        return self.model.get_params(deep = deep)
    
    def set_params(self, **params):
        return self.model.set_params(**params)  


# In[ ]:


def test_classification(model = SVC(random_state=0), return_model = False):
    from sklearn.datasets import load_breast_cancer
    
    data = load_breast_cancer()

    # Create test and train sets from one dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data["data"], 
        data["target"], 
        test_size = 0.3, 
        random_state = 0,
        stratify = data["target"],
    )

    # Create a validation set.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, 
        y_train, 
        test_size = 0.3, 
        random_state = 0,
        stratify = y_train,
    )

    # List the parameters to search across
    # List the parameters to search across
    param_grid = {
        'C': [1, 10, 100, 120, 150], 
        'gamma': [0.001, 0.0001], 
        'kernel': ['rbf'],
    }

    # Grid-search all parameter combinations using a validation set.
    gs = GridSearch(model)
    gs.fit(X_train, y_train, param_grid, X_val, y_val, verbose = False)

    # Compare with default model without hyperopt
    default = SVC(random_state=0)
    default.fit(X_train, y_train)
    
    assert(round(default.score(X_test, y_test), 4) == 0.6257)
    assert(round(gs.score(X_test, y_test), 4) == 0.9240)
    
    if return_model:
        return gs


# In[ ]:


def test_regression():
    from sklearn.datasets import load_boston
    from sklearn.svm import SVR
    
    data = load_boston()

    # Create test and train sets from one dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data["data"], 
        data["target"], 
        test_size = 0.1, 
        random_state = 0,
    )

    # Create a validation set.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, 
        y_train, 
        test_size = 0.1, 
        random_state = 0,
    )

    # List the parameters to search across
    param_grid = {
        'C': [1, 10, 100, 120, 150], 
        'gamma': [0.001, 0.0001], 
        'kernel': ['rbf'],
    }

    # Grid-search all parameter combinations using a validation set.
    gs = GridSearch(model = SVR())
    gs.fit(X_train, y_train, param_grid, X_val, y_val, verbose = False)

    # Compare with default model without hyperopt
    default = SVR()
    default.fit(X_train, y_train)

    assert(round(default.score(X_test, y_test), 4) == .0175)
    assert(round(gs.score(X_test, y_test), 4) == .4532)


# In[ ]:


def test_gridsearch_crossval():
    from sklearn.datasets import load_breast_cancer
    from sklearn.svm import SVC
    
    data = load_breast_cancer()

    # Create test and train sets from one dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data["data"], 
        data["target"], 
        test_size = 0.3, 
        random_state = 0,
        stratify = data["target"],
    )

    # List the parameters to search across
    # List the parameters to search across
    param_grid = {
        'C': [1, 10, 100, 120, 150], 
        'gamma': [0.001, 0.0001], 
        'kernel': ['rbf'],
    }

    # Grid-search all parameter combinations WITHOUT a validation set.
    gs = GridSearch(model = SVC(random_state=0))
    gs.fit(X_train, y_train, param_grid, verbose = False)
    assert(round(gs.score(X_test, y_test), 4) == 0.9298)


# In[ ]:


def test_accessors():
    model = test_classification(return_model=True)
    assert(model.get_best_params() == {'C': 150, 'gamma': 0.0001, 'kernel': 'rbf'})
    assert(round(model.get_best_score(), 4) == 0.9583)
    assert(model.get_param_scores()[0][0] == model.get_best_params())
    assert(model.get_param_scores()[0][1] == model.get_best_score())
    assert(model.get_params()[0] == model.get_best_params())
    assert(model.get_scores()[0] == model.get_best_score())


# In[ ]:


def test_no_score_model():
    model = model_without_score(SVC())
    test_classification(model)


# In[ ]:


def test_exception():
    model = model_that_throws_exception(SVC())
    TestCase.assertRaises(ValueError, test_classification(model))

