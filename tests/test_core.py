
# coding: utf-8

# In[ ]:


from __future__ import print_function, absolute_import, division, unicode_literals, with_statement
from sklearn.model_selection import train_test_split
from hypopt import GridSearch


# In[ ]:


def test_classification():
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
    gs = GridSearch(model = SVC(random_state=0))
    gs.fit(X_train, y_train, param_grid, X_val, y_val, verbose = False)

    # Compare with default model without hyperopt
    default = SVC(random_state=0)
    default.fit(X_train, y_train)
    
    assert(round(default.score(X_test, y_test), 4) == 0.6257)
    assert(round(gs.score(X_test, y_test), 4) == 0.9240)


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
    gs.fit(X_train, y_train, param_grid, X_val, y_val)

    # Compare with default model without hyperopt
    default = SVR()
    default.fit(X_train, y_train)

    assert(round(default.score(X_test, y_test), 4) == .0175)
    assert(round(gs.score(X_test, y_test), 4) == .4532)

