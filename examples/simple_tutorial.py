
# coding: utf-8

# ## Simple Tutorial using `hyperopt`
# 
# In this simple tutorial, we show how to use hyperopt on the well known Iris dataset from scikit-learn. We use a neural network as the model, but any model works.

# In[1]:


from hyperopt.model_selection import GridSearch
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split

# Neural Network imports (simple sklearn Neural Network)
from sklearn.neural_network import MLPClassifier
# Silence neural network SGD convergence warnings.
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)


# In[2]:


param_grid = {
    'learning_rate': ["constant", "adaptive"],
    'hidden_layer_sizes': [(100,20), (500,20), (20,10), (50,20), (4,2)],
    'alpha': [0.0001], # minimal effect
    'warm_start': [False], # minimal effect
    'momentum': [0.1, 0.9], # minimal effect
    'learning_rate_init': [0.001, 0.01, 1],
    'max_iter': [50],
    'random_state': [0],
    'activation': ['relu'],
}


# In[3]:


# Get data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris["data"], iris["target"], test_size = 0.3, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train,test_size = 0.3, random_state = 0)
print('Set sizes:', len(X_train), '(train),', len(X_val), '(val),', len(X_test), '(test)')

# Normalize data 
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_val = scaler.transform(X_val)  
X_test = scaler.transform(X_test) 


# ### Grid-search time comparison using validation set versus cross-validation. 
# ### The hyperopt package automatically distributes work on all CPU threads regardless of if you use a validation set or cross-validation.

# In[4]:


print("First let's try the neural network with default parameters.")
default = MLPClassifier(max_iter=50, random_state=0)
default.fit(X_train, y_train)
test_score = round(default.score(X_test, y_test), 4)
val_score = round(default.score(X_val, y_val), 4)
print('\nTEST SCORE (default parameters):', test_score)
print('VALIDATION SCORE (default parameters):', val_score)


# In[5]:


gs_val = GridSearch(model = MLPClassifier(max_iter=50, random_state=0))
print("Grid-search using a validation set.\n","-"*79)
get_ipython().run_line_magic('time', 'gs_val.fit(X_train, y_train, param_grid, X_val, y_val)')
test_score = round(gs_val.score(X_test, y_test), 4)
val_score = round(gs_val.score(X_val, y_val), 4)
print('\nTEST SCORE (hyper-parameter optimization with validation set):', test_score)
print('VALIDATION SCORE (hyper-parameter optimization with validation set):', val_score)


# In[6]:


gs_cv = GridSearch(model = MLPClassifier(max_iter=50, random_state=0), cv_folds=6)
print("\n\nLet's see how long grid-search takes to run when we don't use a validation set.")
print("Grid-search using cross-validation.\n","-"*79)
get_ipython().run_line_magic('time', 'gs_cv.fit(X_train, y_train, param_grid)')
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
print('\nAlas, these did poorly because the hidden layers are too small (4,2) and the learning rate is too high (1).')
print('Also, note that some of the scores are the same. With hyperopt, you can view all scores and settings to see which parameters really matter.')

