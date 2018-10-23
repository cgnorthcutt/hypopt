
# coding: utf-8

# ## MNIST example with hyper-parameter optimization on a validation set using `hyperopt`
# 
# #### In this simple tutorial, we show how to use hyperopt on the well known MNIST handwritten digits dataset. We use a neural network as the model, but any model works.

# In[1]:


from __future__ import print_function, absolute_import, division, unicode_literals, with_statement

from hypopt import GridSearch
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split

# Neural Network imports (simple sklearn Neural Network)
from sklearn.neural_network import MLPClassifier
# Silence neural network SGD convergence warnings.
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# requires pip install mnist
import mnist 


# In[2]:


param_grid = {
    'learning_rate': ["constant"],
    'hidden_layer_sizes': [(1000,20), (10,5)],
    'alpha': [0.0001], # minimal effect
    'warm_start': [False], # minimal effect
    'momentum': [0.9], # minimal effect
    'learning_rate_init': [0.001, 0.005],
    'random_state': [0],
    'activation': ['relu'],
}


# In[3]:


# Get data - this make take some time to download the first time.
X_train, X_val, y_train, y_val = train_test_split(
    mnist.train_images().reshape(60000, 28 * 28).astype(float), 
    mnist.train_labels(), 
    stratify=mnist.train_labels(),
    test_size = 0.2, 
    random_state = 0,
)

X_test, y_test = mnist.test_images().reshape(10000, 28 * 28).astype(float), mnist.test_labels()
print('Set sizes:', len(X_train), '(train),', len(X_val), '(val),', len(X_test), '(test)')

# Normalize data 
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_val = scaler.transform(X_val)  
X_test = scaler.transform(X_test) 


# ### Grid-search time comparison using validation set versus cross-validation. 
# #### The hyperopt package automatically distributes work on all CPU threads regardless of if you use a validation set or cross-validation.

# In[4]:


print("First let's try the neural network with default parameters.")
default = MLPClassifier(max_iter=6, random_state=0)
get_ipython().magic(u'time default.fit(X_train, y_train)')
test_score = round(default.score(X_test, y_test), 4)
val_score = round(default.score(X_val, y_val), 4)
print('\nTEST SCORE (default parameters):', test_score)
print('VALIDATION SCORE (default parameters):', val_score)


# In[5]:


gs_val = GridSearch(model = MLPClassifier(max_iter=6, random_state=0))
print("Grid-search using a validation set.\n","-"*79)
get_ipython().magic(u'time gs_val.fit(X_train, y_train, param_grid, X_val, y_val)')
test_score = round(gs_val.score(X_test, y_test), 4)
val_score = round(gs_val.score(X_val, y_val), 4)
print('\nTEST SCORE (hyper-parameter optimization with validation set):', test_score)
print('VALIDATION SCORE (hyper-parameter optimization with validation set):', val_score)


# In[6]:


gs_cv = GridSearch(model = MLPClassifier(max_iter=6, random_state=0), cv_folds=5)
print("\n\nLet's see how long grid-search takes to run when we don't use a validation set.")
print("Grid-search using cross-validation.\n","-"*79)
get_ipython().magic(u'time gs_cv.fit(X_train, y_train, param_grid)')
test_score = round(gs_cv.score(X_test, y_test), 4)
val_score = round(gs_cv.score(X_val, y_val), 4)
print('\nTEST SCORE (hyper-parameter optimization with cross-validation):', test_score)
print('VALIDATION SCORE (hyper-parameter optimization with cross-validation):', val_score)
print('''\nNote that although its slower, cross-validation has many benefits (e.g. uses all
your training data). Thats why hyperopt also supports cross-validation when no validation 
set is provided as in the example above.''')


# In[7]:


print('We can view the best performing parameters and their scores.')
for z in gs_val.get_param_scores()[:2]:
    p, s = z
    print(p)
    print('Score:', s)
print()
print('Verify that the lowest scoring parameters make sense.')
for z in gs_val.get_param_scores()[-2:]:
    p, s = z
    print(p)
    print('Score:', s)
print('\nAlas, these did poorly because the hidden layers are too small (10,5).')

