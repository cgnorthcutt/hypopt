
# coding: utf-8

# # Minimal Regression Example

# In[1]:


from __future__ import print_function, absolute_import, division, unicode_literals, with_statement
from hypopt import GridSearch
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


# In[2]:


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
print('\nTest score comparison (larger is better):')
print('Non-optimized Parameters:', round(default.score(X_test, y_test), 4))
print('Optimized Parameters:', round(gs.score(X_test, y_test), 4))

