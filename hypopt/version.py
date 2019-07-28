__version__ = '1.0.9'

# 1.0.9 - MAJOR UPDATE. Previous versions are now deprecated.
#       - Parallelization in folds now uses Daemon-free processes to avoid colluding model.fit() child daemon processes.
#       - Parallelization can now be toggled off via GridSearch(parallelize=False)

# 1.0.8 - Moved param_grid parameter from GridSearch.fit() to the constructor
# 1.0.7 - Added scoring support (F1, accuracy, etc.). 100% testing coverage.
# 1.0.6 (and below) - completely deprecated. Do not use.
