
# coding: utf-8

# In[ ]:


from __future__ import print_function, absolute_import, division, unicode_literals, with_statement
from sklearn.model_selection import train_test_split
from hypopt import GridSearch
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression   
import pytest


# In[ ]:


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


# In[ ]:


def test_classification(
    model = SVC(random_state=0), 
    return_model = False,
    param_grid = None,
    opt_score = 0.9240,
    assertions = True,
    scoring = None,
    verbose = False,
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
    opt.fit(
        X_train, 
        y_train, 
        param_grid, 
        X_val, 
        y_val, 
        scoring = scoring,
        verbose = True,
    )

    # Compare with default model without hyperopt
    default = SVC(random_state=0)
    default.fit(X_train, y_train)
    
    default_score = round(default.score(X_test, y_test), 4)
    optimal_score = round(opt.score(X_test, y_test), 4)
    
    if verbose:
        print(
            'Default score:', default_score, 
            '| GridSearch Score:', optimal_score
        )
    
    if assertions:
        assert(default_score == 0.6257)
        assert(optimal_score == opt_score)
    
    if return_model:
        return opt


# In[ ]:


def test_regression(
    model = SVR(), 
    return_model = False,
    param_grid = None,
    gs_score = .4532,
    assertions = True,
    scoring = None,
    verbose = False,
):    
    from sklearn.datasets import load_boston
    
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
    if param_grid is None:
        param_grid = {
            'C': [1, 10, 100, 120, 150], 
            'gamma': [0.001, 0.0001], 
            'kernel': ['rbf'],
        }

    # Grid-search all parameter combinations using a validation set.
    gs = GridSearch(model = model)
    gs.fit(
        X_train, 
        y_train, 
        param_grid, 
        X_val, 
        y_val, 
        scoring = scoring,
        verbose = True,
    )

    # Compare with default model without hyperopt
    default = model
    default.fit(X_train, y_train)
    
    default_score = round(default.score(X_test, y_test), 4)
    gridsearch_score = round(gs.score(X_test, y_test), 4)
    
    if verbose:
        print(
            'Default score:', default_score, 
            '| GridSearch Score:', gridsearch_score
        )
    
    if assertions:
        assert(default_score == .0175)
        assert(gridsearch_score is not None)
        
    if return_model:
        return gs


# In[ ]:


def test_gridsearch_crossval(
    model = SVC(random_state=0), 
    return_model = False,
    param_grid = None,
    opt_score = 0.9298,
    assertions = True,
    scoring = None,
    verbose = False,
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

    # List the parameters to search across
    if param_grid is None:
        param_grid = {
            'C': [1, 10, 100, 120, 150], 
            'gamma': [0.001, 0.0001], 
            'kernel': ['rbf'],
        }

    # Grid-search all parameter combinations WITHOUT a validation set.
    gs = GridSearch(model = model)
    gs.fit(X_train, y_train, param_grid, scoring = scoring, verbose = False)
        
    # Compare with default model without hyperopt
    default = SVC(random_state=0)
    default.fit(X_train, y_train)
    
    default_score = round(default.score(X_test, y_test), 4)
    gs_score = round(gs.score(X_test, y_test), 4)
    
    if verbose:
        print(
            'Default score:', default_score, 
            '| GridSearch Score:', gs_score
        )
    
    if assertions:
        assert(gs_score == opt_score)
    
    if return_model:
        return gs


# In[ ]:


def test_accessors():
    model = test_classification(return_model=True)
    assert(model.get_best_params() == {'C': 150, 'gamma': 0.0001, 'kernel': 'rbf'})
    assert(round(model.get_best_score(), 4) == 0.9583)
    assert(model.get_param_scores()[0][0] == model.get_best_params())
    assert(model.get_param_scores()[0][1] == model.get_best_score())
    assert(model.get_ranked_params()[0] == model.get_best_params())
    assert(model.get_ranked_scores()[0] == model.get_best_score())


# In[ ]:


def test_no_score_model():
    model = Model_without_sample_weight(model=SVC())
    test_classification(model)


# In[ ]:


def test_no_sample_weight_model():
    model = Model_without_score(model=SVC())
    test_classification(model)


# In[ ]:


def test_exception():
    model = Model_that_throws_exception(model=SVC())
    with pytest.raises(ValueError):
        test_classification(model)


# In[ ]:


def test_prob_methods():
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
    model.fit(X_train, y_train, param_grid, verbose = False)
    
    assert(model.predict(X_test) is not None)
    assert(model.predict_proba(X_test) is not None)


# In[ ]:


# Create custom regression scoring function
def reg_custom_score_func(y, y_pred):    
    from sklearn.metrics import r2_score
    return 2 * r2_score(y, y_pred)

def clf_custom_score_func(y, y_pred):
    from sklearn.metrics import accuracy_score
    return 2 * accuracy_score(y, y_pred)
    
def test_scoring():
    from sklearn.metrics import make_scorer
    
    print('Testing scoring... printing out each test result.')
    print('*'*80)
    
    # Test Regression
    for regression_metric in [
        None,
        make_scorer(reg_custom_score_func),
        "explained_variance",
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "neg_mean_squared_log_error",
        "neg_median_absolute_error",
        "r2",
    ]:
        print('\nRegression Metric:', regression_metric)
        test_regression(verbose = True, scoring = regression_metric)
    
    # Test classification    
    lr_param_grid = {'C': [1, 10, 100, 120, 150]}
    for classification_metric in [
        None,
        make_scorer(clf_custom_score_func),
        'accuracy',
        'brier_score_loss',
        'f1',
        'f1_micro',
        'f1_macro',
        'f1_weighted',
        'neg_log_loss',
        'precision',
        'recall',
        'roc_auc',
    ]:
        print('\nClassification Metric (with validation set):', classification_metric)
        test_classification(
            model = LogisticRegression(), 
            param_grid=lr_param_grid,
            verbose = True, 
            assertions = False, 
            scoring = classification_metric,
        )
        
        if classification_metric not in [
            'brier_score_loss',
        ]:
            print('\nClassification Metric (using cross validation):', classification_metric)
            test_gridsearch_crossval(
                model = LogisticRegression(), 
                param_grid = lr_param_grid,
                verbose = True, 
                assertions = False, 
                scoring = classification_metric,
            )


# In[ ]:


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

