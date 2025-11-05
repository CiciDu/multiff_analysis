import sys
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import CCA as SklearnCCA
import statsmodels.api as sm
from cca_zoo.linear import rCCA
from types import SimpleNamespace


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)
np.set_printoptions(threshold=sys.maxsize)


def conduct_linear_regression(X, y):
    # Add a column of ones to X to include the intercept term
    X_with_const = sm.add_constant(X)
    # Perform multilinear regression
    model = LinearRegression()
    model.fit(X_with_const, y)

    # Get the results
    slope = model.coef_
    intercept = model.intercept_
    r_squared = model.score(X_with_const, y)
    r_value = np.sqrt(r_squared)
    y_pred = model.predict(X_with_const)

    # Calculate p-values using statsmodels
    ols_model = sm.OLS(y, X_with_const).fit()
    p_values = ols_model.pvalues
    f_p_value = ols_model.f_pvalue

    return slope, intercept, r_value, r_squared, p_values, f_p_value, y_pred


def get_y_var_lr_df(binned_spikes_matrix, final_behavioral_data, verbose=False):
    # conduct linear regression on X and y
    x_var = binned_spikes_matrix.copy()
    all_r = []
    all_r_squared = []
    all_p_values = []
    for i, column in enumerate(final_behavioral_data.columns):
        y_var = final_behavioral_data[column]
        slope, intercept, r_value, r_squared, p_values, f_p_value, y_pred = conduct_linear_regression(
            x_var, y_var)
        all_r.append(r_value)
        all_r_squared.append(r_squared)
        all_p_values.append(f_p_value)
        if verbose:
            print(
                f'{column} r: {round(r_value, 3)}, r_squared: {round(r_squared, 3)}, p_values: {round(f_p_value, 3)}')
    y_var_lr_df = pd.DataFrame({'feature': final_behavioral_data.columns,
                               'r': all_r, 'r_squared': all_r_squared, 'p_values': all_p_values})
    y_var_lr_df['significant'] = y_var_lr_df['p_values'] < 0.05
    y_var_lr_df.sort_values(by='r_squared', ascending=False, inplace=True)
    y_var_lr_df.reset_index(drop=True, inplace=True)

    return y_var_lr_df


def conduct_cca(X1_sc, X2_sc, n_components=10, plot_correlations=True, reg=1e-2):
    # Prefer cca_zoo regularized CCA; fall back to sklearn CCA if unavailable/incompatible
    try:
        cca = rCCA(latent_dimensions=n_components, c=[reg, reg])
        cca.fit([X1_sc, X2_sc])
        X1_c, X2_c = cca.transform([X1_sc, X2_sc])
    except Exception as e:
        print(f"Fallback: use scikit-learn CCA (no ridge regularization). Error: {e}")
        cca = SklearnCCA(n_components=n_components, scale=False, max_iter=5000)
        cca.fit(X1_sc, X2_sc)
        X1_c, X2_c = cca.transform(X1_sc, X2_sc)

    # Compute canonical correlations
    k = min(X1_c.shape[1], X2_c.shape[1])
    canon_corr = np.array(
        [np.corrcoef(X1_c[:, i], X2_c[:, i])[0, 1] for i in range(k)])

    print('Canonical Correlation per Component Pair:', canon_corr)
    print('% Shared Variance:', np.round(canon_corr ** 2 * 100, 2))

    # Optional: plot canonical correlations
    if plot_correlations:
        bar_names = [f'CC {i+1}' for i in range(len(canon_corr))]
        plt.bar(bar_names, canon_corr, color='lightgrey', edgecolor='k')
        for i, val in enumerate(canon_corr):
            plt.text(i, val, f'{val:.2f}', ha='center', va='bottom')
        plt.ylabel('Correlation')
        plt.title('Canonical Correlations per Component')
        plt.show()

    return cca, X1_c, X2_c, canon_corr


def calculate_loadings(original_data, canonical_components):
    loadings = np.corrcoef(original_data.T, canonical_components.T)[
        :original_data.shape[1], original_data.shape[1]:]
    return loadings


def make_loading_or_weight_df(loading, columns, lagging_included=False):
    loading_df = pd.DataFrame(loading)
    loading_df['feature'] = columns
    if lagging_included:
        loading_df['feature_category'] = loading_df['feature'].apply(
            lambda x: '_'.join(x.split('_')[:-1]))
    else:
        loading_df['feature_category'] = loading_df['feature']
    return loading_df


def print_weights(name, weights):
    first = weights[:, 0] / np.max(np.abs(weights[:, 0]))
    print(name + ': ' + ', '.join(['{:.3f}'.format(item) for item in first]))
