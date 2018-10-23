
# coding: utf-8

# # Minimal Classification Example

# In[1]:


from __future__ import print_function, absolute_import, division, unicode_literals, with_statement
from hypopt import GridSearch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[2]:


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
# You can choose the metric to optimize (f1, auc_roc, accuracy, etc.)
# scoring = None will default to optimizing model.score()
_ = gs.fit(X_train, y_train, param_grid, X_val, y_val, scoring = 'f1')

# Compare with default model without hyperopt
default = SVC(random_state=0)
_ = default.fit(X_train, y_train)
print('\nTest score comparison (larger is better):')
print('Non-optimized Parameters:', round(default.score(X_test, y_test), 4))
print('Optimized Parameters:', round(gs.score(X_test, y_test), 4))

