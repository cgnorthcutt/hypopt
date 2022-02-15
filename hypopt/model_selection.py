
# coding: utf-8

# Python 2 and 3 compatibility
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement

# Imports
import sys
import inspect
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
try:
    from sklearn.metrics import _scorer as scorer
except:
    from sklearn.metrics import scorer
import numpy as np
import warnings

# For parallel processing
import multiprocessing
import multiprocessing.pool

SUPPRESS_WARNINGS = False

# tqdm is a module used to print time-to-complete when multiprocessing is used.
# This module is not necessary, and therefore is not a package dependency, but 
# when installed it improves user experience for large datsets.
try:
    import tqdm
    tqdm_exists = True
except ImportError as e:
    tqdm_exists = False
    import warnings
    w = '''If you want to see estimated completion times
    while running methods in cleanlab.pruning, install tqdm
    via "pip install tqdm".'''
    warnings.warn(w)


# Set-up multiprocessing classes and methods

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
    
# For python 2/3 compatibility, define pool context manager
# to support the 'with' statement in Python 2
if sys.version_info[0] == 2:
    from contextlib import contextmanager
    @contextmanager
    def multiprocessing_context(*args, **kwargs):
        pool = MyPool(*args, **kwargs)
        yield pool
        pool.terminate()
else:
    multiprocessing_context = MyPool


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
        return -1. * metrics.log_loss(y, model.predict_proba(X), **scoring_params)
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
def _run_thread_job(model_params):  # pragma: no cover
    try:
        # Seeding may be important for fair comparison of param settings.
        np.random.seed(seed = 0)
        if hasattr(model, 'seed') and not callable(model.seed): 
            model.seed = 0
        if hasattr(model, 'random_state') and not callable(model.random_state): 
            model.random_state = 0
        model.set_params(**model_params)
        model_clone = clone(model)    
        model_clone.fit(X_train, y_train)
        # Compute the score for the given parameters, scoring metric, and model.
        if scoring is None: # use default model.score() if it exists, else use accuracy
            if hasattr(model_clone, 'score'):        
                score = model_clone.score(X_val, y_val)
            else:            
                score = metrics.accuracy_score(
                    y_val,
                    model_clone.predict(X_val),
                )
        # Or you provided your own scoring class
        elif type(scoring) in [scorer._ThresholdScorer, scorer._PredictScorer, scorer._ProbaScorer] \
            or scorer._PredictScorer in type(scoring).__bases__ \
            or scorer._ProbaScorer in type(scoring).__bases__ \
            or scorer._ThresholdScorer in type(scoring).__bases__:
            score = scoring(model_clone, X_val, y_val)
        # You provided a string specifying the metric, e.g. 'accuracy'
        else:
            score = _compute_score(
                model = model_clone,
                X = X_val, 
                y = y_val,
                scoring_metric = scoring,
                scoring_params = scoring_params,
            )
        return (model_clone, score)

    except Exception as e:
        if not SUPPRESS_WARNINGS:
            pname = str(multiprocessing.current_process())
            warnings.warn('ERROR in thread' + pname + "with exception:\n" + str(e))
            return None


def _parallel_param_opt(
    jobs,
    num_threads=None,
):
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()
    K = len(jobs)
    with multiprocessing_context(
        num_threads,
#         initializer=_make_shared_immutables_global,
#         initargs=(model, X_train, y_train, X_val, y_val, scoring, scoring_params),
    ) as p:
        if tqdm_exists:
            results = tqdm.tqdm(p.imap(_run_thread_job, jobs), total=K)
        else:
            results = p.map(_run_thread_job, jobs)
        return [r for r in results if r is not None]


def _make_shared_immutables_global(
    _model,
    _X_train,
    _y_train,
    _X_val,
    _y_val,
    _scoring,
    _scoring_params,
):
    '''Shares memory objects across child processes.
    ASSUMES none of these will change!'''

    global model, X_train, y_train, X_val, y_val, scoring, scoring_params
    model = _model
    X_train = _X_train
    y_train = _y_train
    X_val = _X_val
    y_val = _y_val
    scoring = _scoring
    scoring_params = _scoring_params


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

    param_grid : dict
        The parameters to train with out on the validation set. Dictionary with
        parameters names (string) as keys and lists of parameter settings to try
        as values, or a list of such dictionaries, in which case the grids spanned
        by each dictionary in the list are explored. This enables searching over
        any sequence of parameter settings. Format is:
        {'param1': ['list', 'of', 'options'], 'param2': ['l', 'o', 'o'], ...}\
        For an example, check out:
        scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html

    num_threads : int (chooses max # of threads by default),
        The number of CPU threads to use.

    cv_folds : int (default 3)
        The number of cross-validation folds to use if no X_val, y_val is specified.

    seed : int (default 0)
        Calls np.random.seed(seed = seed)

    parallelize : bool
        Default (true). set to False if you have problems. Will make hypopt slower.'''


    def __init__(
        self,
        model,        
        param_grid,
        num_threads=None,
        seed=0,
        cv_folds=3,
        parallelize=True,
    ):
        if num_threads is None:
            num_threads = multiprocessing.cpu_count()
        self.model = model
        self.param_grid = param_grid
        self.num_threads = num_threads
        self.cv_folds = cv_folds
        self.seed = seed
        self.parallelize = parallelize

        np.random.seed(seed = seed)

        # Pre-define attributes for access after .fit() is called
        self.param_scores = None
        self.best_params = None
        self.best_score = None
        self.best_estimator_ = None
        self.params = None
        self.scores = None


    def fit(
        self,
        X_train,
        y_train,
        X_val = None, # validation data if it exists (if None, use crossval)
        y_val = None, # validation labels if they exist (if None, use crossval)
        scoring = None,
        scoring_params = None,
        verbose = False,
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

        X_val : np.array of shape (n0, m)
            The validation data to optimize paramters with. If you do not provide this,
            cross validation on the training set will be used. 

        y_val : np.array of shape (n0,) or (n0, 1)
            The validation labels to optimize paramters with. If you do not provide this,
            cross validation on the training set will be used.
            
        scoring : str or metrics._scorer._PredictScorer object
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
            params = list(ParameterGrid(self.param_grid))
            if verbose:
                if self.parallelize:
                    print("Comparing", len(params), "parameter setting(s) using",
                          self.num_threads, "CPU thread(s)", end=' ')
                    print("(", max(1, len(params) // self.num_threads), "job(s) per thread ).")
                else:
                    print("Parallelization disabled. Comparing", len(params), "parameter setting(s) sequentially.")
            _make_shared_immutables_global(
                _model=self.model,
                _X_train=X_train,
                _y_train=y_train,
                _X_val=X_val,
                _y_val=y_val,
                _scoring=scoring,
                _scoring_params=scoring_params,
            )
            if self.parallelize:
                results = _parallel_param_opt(params, self.num_threads)
            else:
                results = [_run_thread_job(job) for job in params]
            models, scores = list(zip(*results))
            self.model = models[np.argmax(scores)]
        else:
            model_cv = GridSearchCV(
                estimator = self.model, 
                param_grid = self.param_grid,
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

        # Create alias to enable the same interface as sklearn.GridSearchCV
        self.best_estimator_ = self.model

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
            if hasattr(inspect, 'getfullargspec') and \
            'sample_weight' in inspect.getfullargspec(self.model.score).args or \
            hasattr(inspect, 'getargspec') and \
            'sample_weight' in inspect.getargspec(self.model.score).args:  
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
