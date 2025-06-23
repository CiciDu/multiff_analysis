import sys
from machine_learning.ml_methods import regression_utils, classification_utils, prep_ml_data_utils


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
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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
        cv = RepeatedStratifiedKFold(
            n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
        # define search
        self.random_search = RandomizedSearchCV(
            estimator=model, param_distributions=grid, n_iter=n_iter, cv=cv, verbose=verbose, random_state=random_state, n_jobs=-1)
        self.random_search.fit(self.X_train, self.y_train)
        self.random_result = pd.DataFrame(self.random_search.cv_results_).sort_values(
            by=['rank_test_score', 'mean_fit_time'])
        self.random_result.head(5)

        plot_grid_search(self.random_search)
        self.random_result_table = table_grid_search(
            self.random_search, save=False)
        display(self.random_result_table)

    def grid_search(self, grid, model, n_folds=5, n_repeats=3, verbose=4, random_state=42):
        # print the number of combinations in the grid
        num_combinations = 1
        for key, value in grid.items():
            num_combinations = num_combinations * len(value)
        print('There are', num_combinations, 'combinations in the grid.')

        # define cross validation function
        cv = RepeatedStratifiedKFold(
            n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
        # define search
        self.grid_search = GridSearchCV(
            estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
        self.grid_search.fit(self.X_train, self.y_train)
        self.grid_result = pd.DataFrame(self.grid_search.cv_results_).sort_values(
            by=['rank_test_score', 'mean_fit_time'])
        self.grid_result.head(5)

        plot_grid_search(self.grid_search)
        self.grid_result_table = table_grid_search(
            self.grid_search, save=False)
        display(self.grid_result_table)


# Source for below: https://www.kaggle.com/code/juanmah/grid-search-utils
# Note: functions were slightly modified
"""Utility script with functions to be used with the results of GridSearchCV.

**plot_grid_search** plots as many graphs as parameters are in the grid search results.

**table_grid_search** shows tables with the grid search results.

Inspired in [Displaying the results of a Grid Search](https://www.kaggle.com/grfiv4/displaying-the-results-of-a-grid-search) notebook,
of [George Fisher](https://www.kaggle.com/grfiv4)
"""


__author__ = "Juanma Hernández"
__copyright__ = "Copyright 2019"
__credits__ = ["Juanma Hernández", "George Fisher"]
__license__ = "GPL"
__maintainer__ = "Juanma Hernández"
__email__ = "https://twitter.com/juanmah"
__status__ = "Utility script"


def plot_grid_search(clf):
    """Plot as many graphs as parameters are in the grid search results.

    Each graph has the values of each parameter in the X axis and the Score in the Y axis.

    Parameters
    ----------
    clf: estimator object result of a GridSearchCV
        This object contains all the information of the cross validated results for all the parameters combinations.
    """
    # Convert the cross validated results in a DataFrame ordered by `rank_test_score` and `mean_fit_time`.
    # As it is frequent to have more than one combination with the same max score,
    # the one with the least mean fit time SHALL appear first.
    cv_results = pd.DataFrame(clf.cv_results_).sort_values(
        by=['rank_test_score', 'mean_fit_time'])

    # Get parameters
    parameters = cv_results['params'][0].keys()

    # Calculate the number of rows and columns necessary
    rows = -(-len(parameters) // 2)
    columns = min(len(parameters), 2)
    # Create the subplot
    fig = make_subplots(rows=rows, cols=columns)
    # Initialize row and column indexes
    row = 1
    column = 1

    # For each of the parameters
    for parameter in parameters:
        # As all the graphs have the same traces, and by default all traces are shown in the legend,
        # the description appears multiple times. Then, only show legend of the first graph.
        if row == 1 and column == 1:
            show_legend = True
        else:
            show_legend = False

        # Mean test score
        mean_test_score = cv_results[cv_results['rank_test_score'] != 1]
        fig.add_trace(go.Scatter(
            name='Mean test score',
            x=mean_test_score['param_' + parameter],
            y=mean_test_score['mean_test_score'],
            mode='markers',
            marker=dict(size=mean_test_score['mean_fit_time'],
                        color='SteelBlue',
                        sizeref=2. *
                        cv_results['mean_fit_time'].max() / (40. ** 2),
                        sizemin=4,
                        sizemode='area'),
            text=mean_test_score['params'].apply(
                lambda x: pprint.pformat(x, width=-1).replace('{', '').replace('}', '').replace('\n', '<br />')),
            showlegend=show_legend),
            row=row,
            col=column)

        # Best estimators
        rank_1 = cv_results[cv_results['rank_test_score'] == 1]
        fig.add_trace(go.Scatter(
            name='Best estimators',
            x=rank_1['param_' + parameter],
            y=rank_1['mean_test_score'],
            mode='markers',
            marker=dict(size=rank_1['mean_fit_time'],
                        color='Crimson',
                        sizeref=2. *
                        cv_results['mean_fit_time'].max() / (40. ** 2),
                        sizemin=4,
                        sizemode='area'),
            text=rank_1['params'].apply(str),
            showlegend=show_legend),
            row=row,
            col=column)

        fig.update_xaxes(title_text=parameter, row=row, col=column)
        fig.update_yaxes(title_text='Score', row=row, col=column)

        # Check the linearity of the series
        # Only for numeric series
        if pd.to_numeric(cv_results['param_' + parameter], errors='coerce').notnull().all():
            x_values = cv_results['param_' +
                                  parameter].sort_values().unique().tolist()
            r = stats.linregress(x_values, range(0, len(x_values))).rvalue
            # If not so linear, then represent the data as logarithmic
            if r < 0.86:
                fig.update_xaxes(type='log', row=row, col=column)

        # Increment the row and column indexes
        column += 1
        if column > columns:
            column = 1
            row += 1

            # Show first the best estimators
    fig.update_layout(legend=dict(traceorder='reversed'),
                      width=columns * 560 + 100,
                      height=rows * 460,
                      #   title=dict(text='Best score: {:.3f} with {}'.format(cv_results['mean_test_score'].iloc[0],
                      #                                             str(cv_results['params'].iloc[0]).replace('{',
                      #                                                                                       '').replace(
                      #                                                 '}', '')),
                      #              font=dict(size=14), automargin=True, yref='paper'),
                      hovermode='closest',
                      template='none')
    fig.show()


def table_grid_search(clf, all_columns=False, all_ranks=False, save=True):
    """Show tables with the grid search results.

    Parameters
    ----------
    clf: estimator object result of a GridSearchCV
        This object contains all the information of the cross validated results for all the parameters combinations.

    all_columns: boolean, default: False
        If true all columns are returned. If false, the following columns are dropped:

        - params. As each parameter has a column with the value.
        - std_*. Standard deviations.
        - split*. Split scores.

    all_ranks: boolean, default: False
        If true all ranks are returned. If false, only the rows with rank equal smaller than 6 are returned.

    save: boolean, default: True
        If true, results are saved to a CSV file.
    """
    # Convert the cross validated results in a DataFrame ordered by `rank_test_score` and `mean_fit_time`.
    # As it is frequent to have more than one combination with the same max score,
    # the one with the least mean fit time SHALL appear first.
    cv_results = pd.DataFrame(clf.cv_results_).sort_values(
        by=['rank_test_score', 'mean_fit_time'])

    # Reorder
    columns = cv_results.columns.tolist()
    # rank_test_score first, mean_test_score second and std_test_score third
    columns = columns[-1:] + columns[-3:-1] + columns[:-3]
    cv_results = cv_results[columns]

    if save:
        cv_results.to_csv(
            '--'.join(cv_results['params'][0].keys()) + '.csv', index=True, index_label='Id')

    # Unless all_columns are True, drop not wanted columns: params, std_* split*
    if not all_columns:
        cv_results.drop('params', axis='columns', inplace=True)
        cv_results.drop(list(cv_results.filter(regex='^std_.*')),
                        axis='columns', inplace=True)
        cv_results.drop(list(cv_results.filter(regex='^split.*')),
                        axis='columns', inplace=True)

    # Unless all_ranks are True, only keep the top 20 results
    if not all_ranks:
        cv_results = cv_results[cv_results['rank_test_score'] <= 20]

    return cv_results
