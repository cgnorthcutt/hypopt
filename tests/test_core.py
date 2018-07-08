
# coding: utf-8

# In[6]:


from __future__ import print_function, absolute_import, division, unicode_literals, with_statement
print('here')
from sklearn.model_selection import train_test_split
from hypopt import GridSearch
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
import pytest


# In[15]:


from sklearn.base import BaseEstimator

class Model_without_score(BaseException):
    '''Simple test class of an sklearn model without score function'''

    def __init__(self, model = None):
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

class Model_without_sample_weight(BaseException):
    '''Simple test class of an sklearn model without score function'''

    def __init__(self, model = None):
        self.model = model
        self.seed = 0

    def fit(self, X, y, sample_weight = None):
        return self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y, sample_weight = None)

    def get_params(self, deep = True):
        return self.model.get_params(deep = deep)

    def set_params(self, **params):
        return self.model.set_params(**params)    
    
    
class Model_that_throws_exception(BaseException):
    '''Simple test class of an sklearn model without score function'''

    def __init__(self, model = None):
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


# In[7]:


def test_classification(
    model = SVC(random_state=0), 
    return_model = False,
    param_grid = None,
    opt_score = 0.9240,
    assertions = True,
):    
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
    if param_grid is None:
        param_grid = {
            'C': [1, 10, 100, 120, 150], 
            'gamma': [0.001, 0.0001], 
            'kernel': ['rbf'],
        }

    # Grid-search all parameter combinations using a validation set.
    opt = GridSearch(model)
    opt.fit(X_train, y_train, param_grid, X_val, y_val, verbose = False)

    # Compare with default model without hyperopt
    default = SVC(random_state=0)
    default.fit(X_train, y_train)
    
    if assertions:
        assert(round(default.score(X_test, y_test), 4) == 0.6257)
        assert(round(opt.score(X_test, y_test), 4) == opt_score)
    
    if return_model:
        return opt


# In[8]:


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


# In[9]:


def test_gridsearch_crossval():    
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


# In[10]:


def test_accessors():
    model = test_classification(return_model=True)
    assert(model.get_best_params() == {'C': 150, 'gamma': 0.0001, 'kernel': 'rbf'})
    assert(round(model.get_best_score(), 4) == 0.9583)
    assert(model.get_param_scores()[0][0] == model.get_best_params())
    assert(model.get_param_scores()[0][1] == model.get_best_score())
    assert(model.get_ranked_params()[0] == model.get_best_params())
    assert(model.get_ranked_scores()[0] == model.get_best_score())


# In[11]:


def test_no_score_model():
    model = Model_without_sample_weight(model=SVC())
    test_classification(model)


# In[ ]:


def test_no_sample_weight_model():
    model = Model_without_score(model=SVC())
    test_classification(model)


# In[12]:


def test_exception():
    model = Model_that_throws_exception(model=SVC())
    with pytest.raises(ValueError):
        test_classification(model)


# In[16]:


def test_prob_methods():
    from sklearn.linear_model import LogisticRegression   
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
    param_grid = {'C': [1, 10, 100, 120, 150]}
        
    # Grid-search all parameter combinations using a validation set.
    model = GridSearch(LogisticRegression())
    model.fit(X_train, y_train, param_grid, verbose = True)
    
    assert(model.predict(X_test) is not None)
    assert(model.predict_proba(X_test) is not None)


# In[42]:


# def test_external_model():
#     '''Requires the confidentlearning package'''
#     from confidentlearning.classification import RankPruning
#     test_classification(
#         model = RankPruning(SVC(probability=True)),
#         param_grid = {
#             'prune_method':[
#                 'prune_by_noise_rate', 
#                 'prune_by_class', 
#                 'both',
#             ],
#             'prune_count_method':['inverse_nm_dot_s', 'calibrate_confident_joint'],
#         },
#         opt_score = 0.6257,
#     )

