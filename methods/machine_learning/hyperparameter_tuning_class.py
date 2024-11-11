import sys
if not '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods' in sys.path:
    sys.path.append('/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods')
from machine_learning import machine_learning_utils


import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from IPython.display import display


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)





class HyperparameterTuning:
    
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train



    def random_search(self, grid, model, n_iter=50, n_folds=5, n_repeats=3, verbose=4, random_state=42):
        # define cross validation function
        cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
        # define search
        self.random_search = RandomizedSearchCV(estimator = model, param_distributions = grid, n_iter = n_iter, cv = cv, verbose = verbose, random_state = random_state, n_jobs = -1)
        self.random_search.fit(self.X_train, self.y_train)
        self.random_result = pd.DataFrame(self.random_search.cv_results_).sort_values(by=['rank_test_score', 'mean_fit_time'])
        self.random_result.head(5)

        machine_learning_utils.plot_grid_search(self.random_search)
        self.random_result_table = machine_learning_utils.table_grid_search(self.random_search, save=False)
        display(self.random_result_table)


    def grid_search(self, grid, model, n_folds=5, n_repeats=3, verbose=4, random_state=42):
        # print the number of combinations in the grid
        num_combinations = 1
        for key, value in grid.items():
            num_combinations = num_combinations * len(value)
        print('There are', num_combinations, 'combinations in the grid.')

        # define cross validation function
        cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
        # define search
        self.grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
        self.grid_search.fit(self.X_train, self.y_train)
        self.grid_result = pd.DataFrame(self.grid_search.cv_results_).sort_values(by=['rank_test_score', 'mean_fit_time'])
        self.grid_result.head(5)

        machine_learning_utils.plot_grid_search(self.grid_search)
        self.grid_result_table = machine_learning_utils.table_grid_search(self.grid_search, save=False)
        display(self.grid_result_table)


        

