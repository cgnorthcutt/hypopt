
# coding: utf-8

# In[1]:


# Python 2 and 3 compatibility
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement

# Imports
import inspect
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
import warnings

# For parallel processing
import multiprocessing as mp
from multiprocessing import Pool
max_threads = mp.cpu_count()

SUPPRESS_WARNINGS = False


# In[2]:


def _compute_score(model, X, y, scoring_metric = None, scoring_params = None):
    '''Helper function that maps metric string names to their function calls.
    
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
        
    X : np.array of shape (n, m)
        The training data.

    y : np.array of shape (n,) or (n, 1)
        Corresponding labels.
        
    scoring_metric : str
        See hypopt.GridSearch.fit() scoring parameter docstring 
        for list of options.
        
    scoring_params : dict
        All other params you want passed to the scoring function.
        Params will be passed as scoring_func(**scoring_params).'''
    
    if scoring_params is None:
        scoring_params = {}
    
    if scoring_metric == 'accuracy':
        return metrics.accuracy_score(y, model.predict(X), **scoring_params)
    elif scoring_metric == 'brier_score_loss':
        return metrics.brier_score_loss(y, model.predict(X), **scoring_params)
    elif scoring_metric == 'average_precision':
        return metrics.average_precision_score(y, model.predict_proba(X)[:,1], **scoring_params)
    elif scoring_metric == 'f1':
        return metrics.f1_score(y, model.predict(X), **scoring_params)
    elif scoring_metric == 'f1_micro':
        return metrics.f1_score(y, model.predict(X), average = 'micro', **scoring_params)
    elif scoring_metric == 'f1_macro':
        return metrics.f1_score(y, model.predict(X), average = 'macro', **scoring_params)
    elif scoring_metric == 'f1_weighted':
        return metrics.f1_score(y, model.predict(X), average = 'weighted', **scoring_params)
    elif scoring_metric == 'neg_log_loss':
        return -1. * metrics.log_loss(y, model.predict(X), **scoring_params)
    elif scoring_metric == 'precision':
        return metrics.precision_score(y, model.predict(X), **scoring_params)
    elif scoring_metric == 'recall':
        return metrics.recall_score(y, model.predict(X), **scoring_params)
    elif scoring_metric == 'roc_auc':
        return metrics.roc_auc_score(y, model.predict_proba(X)[:,1], **scoring_params)
    elif scoring_metric == 'explained_variance':
        return metrics.explained_variance_score(y, model.predict(X), **scoring_params)
    elif scoring_metric == 'neg_mean_absolute_error':
        return -1. * metrics.mean_absolute_error(y, model.predict(X), **scoring_params)
    elif scoring_metric == 'neg_mean_squared_error':
        return -1. * metrics.mean_squared_error(y, model.predict(X), **scoring_params)
    elif scoring_metric == 'neg_mean_squared_log_error':
        return -1. * metrics.mean_squared_log_error(y, model.predict(X), **scoring_params)
    elif scoring_metric == 'neg_median_absolute_error':
        return -1. * metrics.median_absolute_error(y, model.predict(X), **scoring_params)
    elif scoring_metric == 'r2':
        return metrics.r2_score(y, model.predict(X), **scoring_params)
    else:
        raise ValueError(scoring_metric + 'is not a supported metric.')
    
    

# Analyze results in parallel on all cores.
def _run_thread_job(params):  
    try:
        job_params, model_params = params
        model = job_params["model"]
        scoring = job_params["scoring"]
        scoring_params = job_params["scoring_params"]
        # Seeding may be important for fair comparison of param settings.
        np.random.seed(seed = 0)
        if hasattr(model, 'seed') and not callable(model.seed): 
            model.seed = 0
        if hasattr(model, 'random_state') and not callable(model.random_state): 
            model.random_state = 0
            
        model.set_params(**model_params)    
        model.fit(job_params["X_train"], job_params["y_train"])
        # Compute the score for the given parameters, scoring metric, and model.
        if scoring is None: # use default model.score() if it exists, else use accuracy
            if hasattr(model, 'score'):        
                score = model.score(job_params["X_val"], job_params["y_val"])
            else:            
                score = metrics.accuracy_score(
                    job_params["y_val"], 
                    model.predict(job_params["X_val"]),
                )
        # You provided your own scoring function.
        elif type(scoring) == metrics.scorer._PredictScorer: 
            score = scoring(model, job_params["X_val"], job_params["y_val"])
        # You provided a string specifying the metric, e.g. 'accuracy'
        else:
            score = _compute_score(
                model = model,
                X = job_params["X_val"], 
                y = job_params["y_val"],
                scoring_metric = scoring,
                scoring_params = scoring_params,
            )
        return (model, score)

    except Exception as e:
        if not SUPPRESS_WARNINGS:
            warnings.warn('ERROR in thread' + str(mp.current_process()) + "with exception:\n" + str(e))
            return None

def _parallel_param_opt(lst, threads=max_threads):
    pool = mp.Pool(threads)
    results = pool.map(_run_thread_job, lst)
    pool.close()
    pool.join()
    return results


# In[ ]:


from sklearn.base import BaseEstimator
class GridSearch(BaseEstimator):
    '''docstring

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

    num_threads : int (chooses max # of threads by default),
        The number of CPU threads to use.

    cv_folds : int (default 3)
        The number of cross-validation folds to use if no X_val, y_val is specified.

    seed : int (default 0)
        Calls np.random.seed(seed = seed)'''


    def __init__(self, model, num_threads = max_threads, seed = 0, cv_folds = 3):
        self.model = model
        self.num_threads = num_threads
        self.cv_folds = cv_folds
        self.seed = seed
        np.random.seed(seed = seed)
        
        # Pre-define attributes for access after .fit() is called
        self.param_scores = None
        self.best_params = None
        self.best_score = None
        self.params = None
        self.scores = None
        
    
    def fit(
        self,
        X_train,
        y_train,
        param_grid,
        X_val = None, # validation data if it exists (if None, use crossval)
        y_val = None, # validation labels if they exist (if None, use crossval)
        scoring = None,
        scoring_params = None,
        verbose = True,
    ):
        '''Returns the model trained with the hyperparameters that maximize accuracy
        on the (X_val, y_val) validation data (if specified), else the parameters
        that maximize cross fold validation score. Uses grid search to find the best
        hyper-parameters.

        Parameters
        ----------

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
            For an example, check out:
            scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html

        X_val : np.array of shape (n0, m)
            The validation data to optimize paramters with. If you do not provide this,
            cross validation on the training set will be used. 

        y_val : np.array of shape (n0,) or (n0, 1)
            The validation labels to optimize paramters with. If you do not provide this,
            cross validation on the training set will be used.
            
        scoring : str or metrics.scorer._PredictScorer object
            If a str is passed in, it must be in ['accuracy', 'brier_score_loss',
            'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'neg_log_loss',
            'average_precision', precision', 'recall', 'roc_auc',
            'explained_variance', 'neg_mean_absolute_error','neg_mean_squared_error', 
            'neg_mean_squared_log_error','neg_median_absolute_error', 'r2']
            This includes every scoring metric available here:
            http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
            If you'd like to create your own scoring function, create an object by passing 
            your custom function into make_scorer() like this:
            sklearn.metrics.make_scorer(your_custom_metric_scoring_function). 
            Then pass that object in as the value for this scoring parameter. See:
            http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
            If scoring is None, model.score() is used by default.
        
        scoring_params : dict
            All other params you want passed to the scoring function.
            Params will be passed as scoring_func(**scoring_params).
            This will NOT be used if X_val and y_val are None (not provided).

        verbose : bool
            Print out useful information when running.'''
        
        validation_data_exists = X_val is not None and y_val is not None
        if validation_data_exists:
            # Duplicate data for each job (expensive)
            job_params = {
                "model": self.model,
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "scoring": scoring,
                "scoring_params": scoring_params,
            }
            params = list(ParameterGrid(param_grid))
            jobs = list(zip([job_params]*len(params), params))
            if verbose:
                print("Comparing", len(jobs), "parameter setting(s) using", self.num_threads, "CPU thread(s)", end=' ')
                print("(", max(1, len(jobs) // self.num_threads), "job(s) per thread ).")
            results = _parallel_param_opt(jobs, threads = self.num_threads)
            results = [result for result in results if result is not None]
            models, scores = list(zip(*results))
            self.model = models[np.argmax(scores)]
        else:
            model_cv = GridSearchCV(
                estimator = self.model, 
                param_grid = param_grid,
                scoring = scoring,
                cv = self.cv_folds, 
                n_jobs = self.num_threads,
                return_train_score = False,
            )
            model_cv.fit(X_train, y_train)
            scores = model_cv.cv_results_['mean_test_score']
            params = model_cv.cv_results_['params']
            self.model = model_cv.best_estimator_
            
        best_score_ranking_idx = np.argsort(scores)[::-1]
        self.scores = [scores[z] for z in best_score_ranking_idx]
        self.params = [params[z] for z in best_score_ranking_idx]
        self.param_scores = list(zip(self.params, self.scores))
        self.best_score = self.scores[0]
        self.best_params = self.params[0]
        return self.model
    
    
    def predict(self, X):
        '''Returns a binary vector of predictions.

        Parameters
        ----------
        X : np.array of shape (n, m)
          The test data as a feature matrix.'''

        return self.model.predict(X)
  
  
    def predict_proba(self, X):
        '''Returns a vector of probabilties P(y=k)
        for each example in X.

        Parameters
        ----------
        X : np.array of shape (n, m)
          The test data as a feature matrix.'''

        return self.model.predict_proba(X)
    
    
    def score(self, X, y, sample_weight=None):
        '''Returns the model's score on a test set X with labels y.
        Uses the models default scoring function.

        Parameters
        ----------
        X : np.array of shape (n, m)
          The test data as a feature matrix.
          
        y : np.array<int> of shape (n,) or (n, 1)
          The test classification labels as an array.
          
        y : np.array<int> of shape (n,) or (n, 1)
          The test classification labels as an array.
          
        sample_weight : np.array<float> of shape (n,) or (n, 1)
          Weights each example when computing the score / accuracy.'''
        
        if hasattr(self.model, 'score'):
        
            # Check if sample_weight in clf.score(). Compatible with Python 2/3.
            if hasattr(inspect, 'getfullargspec') and                 'sample_weight' in inspect.getfullargspec(self.model.score).args or                 hasattr(inspect, 'getargspec') and                 'sample_weight' in inspect.getargspec(self.model.score).args:  
                return self.model.score(X, y, sample_weight=sample_weight)
            else:
                return self.model.score(X, y)
        else:
            return metrics.accuracy_score(y, self.model.predict(X), sample_weight=sample_weight) 
        
    
    def get_param_scores(self):
        '''Accessor to return param_scores, a list of tuples
        containing pairs of parameters and the associated score
        on the validation set, ordered by descending score.
        e.g. [({'a':1}, 0.95), ({'a':2}, 0.93), ({'a':0}, 0.87)]'''
        return self.param_scores


    def get_best_params(self):
        '''Accessor to return best_params, a dictionary of the
        parameters that scored the best on the validation set.'''
        return self.best_params


    def get_best_score(self):
        '''Accessor to return best_score, the highest score on the val set.'''
        return self.best_score


    def get_ranked_params(self):
        '''Accessor to return params, a list of parameter dicts,
        ordered by descending score on the validation set.'''
        return self.params


    def get_ranked_scores(self):
        '''Accessor to return scores, a list of scores ordered
        by descending score on the validation set.'''
        return self.scores

