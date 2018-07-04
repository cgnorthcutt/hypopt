
# coding: utf-8

# In[1]:


# Python 2 and 3 compatibility
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement

# Imports
import inspect
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import warnings

# For parallel processing
import multiprocessing as mp
from multiprocessing import Pool
max_threads = mp.cpu_count()


# In[28]:


# Analyze results in parallel on all cores.
def _run_thread_job(params):  
    try:
        job_params, model_params = params
        model = job_params["model"]
        
        # Seeding may be important for fair comparison of param settings.
        np.random.seed(seed = 0)
        if hasattr(model, 'seed') and not callable(model.seed): 
            model.seed = 0
        if hasattr(model, 'random_state') and not callable(model.random_state): 
            model.random_state = 0
            
        model.set_params(**model_params)    
        model.fit(job_params["X_train"], job_params["y_train"])
        
        if hasattr(model, 'score'):        
            score = model.score(job_params["X_val"], job_params["y_val"])
        else:            
            score = accuracy_score(y_val, model.predict(job_params["X_val"]))
        return (model, score)

    except Exception as e:
        # Supress warning
#         warnings.warn('ERROR in thread' + str(mp.current_process()) + "with exception:\n" + str(e))
        return None

def _parallel_param_opt(lst, threads=max_threads):
    pool = mp.Pool(threads)
    results = pool.map(_run_thread_job, lst)
    pool.close()
    pool.join()
    return results


# In[30]:


def fit_model_with_grid_search(
    model,
    X_train,
    y_train,
    param_grid,
    X_val = None, # validation data if it exists (if None, use crossval)
    y_val = None, # validation labels if they exist (if None, use crossval)
    num_threads = max_threads, # Chooses max threads by default
    cv_folds = 3, # Only used if X_val, y_val are None
    seed = 0,
    verbose = True,
):
    '''Returns the model trained with the hyperparameters that maximize accuracy
    on the (X_val, y_val) validation data (if specified), else the parameters
    that maximize cross fold validation score. Uses grid search to find the best
    hyper-parameters.
    
    Parameters
    ----------
        
    model : class inheriting sklearn.base.BaseEstimator
        The classifier whose hyperparams you need to optimize with grid search.
        The model must have model.fit(X,y) and model.predict(X) defined. Although it can
        work without it, its best if you also define model.score(X,y) so you can decide
        the scoring function for deciding the best parameters. If you are using an
        sklearn model, everything will work out of the box. To use a model from a 
        different library is no problem, but you need to wrap it in a class and
        inherit sklearn.base.BaseEstimator as seen in:
        https://github.com/cgnorthcutt/hyperopt
        
    X_train : np.array of shape (n, m)
        The training data.
        
    y_train : np.array of shape (n,) or (n, 1)
        The training labels. They can be noisy if you use model = RankPruning().
        
    param_grid : dict
        The parameters to train with out on the validation set. Dictionary with
        parameters names (string) as keys and lists of parameter settings to try
        as values, or a list of such dictionaries, in which case the grids spanned
        by each dictionary in the list are explored. This enables searching over
        any sequence of parameter settings. Format is:
        {'param1': ['list', 'of', 'options'], 'param2': ['l', 'o', 'o'], ...}\
        e.g. scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html
        
    X_val : np.array of shape (n0, m)
        The validation data to optimize paramters with. If you do not provide this,
        cross validation on the training set will be used. 
        
    y_val : np.array of shape (n0,) or (n0, 1)
        The validation labels to optimize paramters with. If you do not provide this,
        cross validation on the training set will be used. 
        
    num_threads : int (chooses max # of threads by default),
        The number of CPU threads to use.
        
    cv_folds : int (default 3)
        The number of cross-validation folds to use if no X_val, y_val is specified.
        
    seed : int (default 0)
        Calls np.random.seed(seed = seed)
        
    verbose : bool
        Print out useful information when running.'''
        
    if X_val is not None and y_val is not None:
        # Duplicate data for each job (expensive)
        job_params = {
            "model": model,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
        }
        param_settings = list(ParameterGrid(param_grid))
        jobs = list(zip([job_params]*len(param_settings), param_settings))
        if verbose:
            print("Running", len(jobs)//num_threads, "job(s) on", num_threads, "thread(s).")
        results = _parallel_param_opt(jobs, threads = num_threads)
        results = [result for result in results if result is not None]
        models, scores = list(zip(*results))
        best_idx = np.argmax(scores)
        return models[best_idx]
    else:
        np.random.seed(seed = seed)
        model_cv = GridSearchCV(estimator = model, param_grid = param_grid, cv = cv_folds, n_jobs = num_threads)
        model_cv.fit(X_train, y_train)
        return model_cv.best_estimator_

