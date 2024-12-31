import sys
from planning_analysis.plan_factors import plan_factors_utils, plan_factors_class, test_vs_control_utils
from planning_analysis.show_planning import show_planning_utils
from planning_analysis.only_stop_ff import features_to_keep_utils, only_stop_ff_utils
from planning_analysis import ml_methods_utils
from non_behavioral_analysis.neural_data_analysis.neural_vs_behavioral import neural_data_modeling, plot_modeling_result
from machine_learning import machine_learning_utils
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
import gc
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats.mstats import winsorize
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
import numpy as np
import matplotlib.pyplot as plt




def add_interaction_terms_to_df(df, specific_columns=None):

    if specific_columns is not None:
        if len(specific_columns) > 0:
            # Assuming df is your DataFrame and specific_columns is a list of column names
            df_selected = df[specific_columns]

            # Initialize PolynomialFeatures
            poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

            # Fit and transform the selected DataFrame
            df_interactions = poly.fit_transform(df_selected)

            # Generate new column names (including original and interaction terms)
            new_column_names = poly.get_feature_names_out(input_features=df_selected.columns)

            # Create a new DataFrame with the interaction terms and new column names
            df_with_interactions = pd.DataFrame(df_interactions, columns=new_column_names)

            # df_with_interactions now contains the original columns, squared terms, and interaction terms with appropriate names

            # drop the original columns
            df_with_interactions.drop(columns=specific_columns, inplace=True)
            df_with_interactions.index = df.index

            df = pd.concat([df, df_with_interactions], axis=1)
            print('Added interaction terms.')
    return df


def winsorize_x_df(x_features_df):
    for feature in x_features_df.columns:
        if ('angle' in feature) & ('rank' not in feature):
            # Winsorize the feature column at 5th and 95th percentiles
            x_features_df[feature] = winsorize(x_features_df[feature], limits=[0.01, 0.01])
    return x_features_df


def streamline_preparing_for_ml(x_df,
                                y_df,
                                y_var_column,
                                ref_columns_only=False,
                                cluster_to_keep='none',
                                cluster_for_interaction='none',
                                add_ref_interaction=True,
                                winsorize_angle_features=True, 
                                using_lasso=True, 
                                ensure_stop_ff_at_front=True,
                                use_pca=False,
                                use_combd_features_for_cluster_only=False,
                                for_classification=False):

    if len(x_df) != len(y_df):
        raise ValueError('x_df and y_df should have the same length.')

    x_df = x_df.reset_index(drop=True).copy()
    y_df = y_df.reset_index(drop=True).copy()

    if ensure_stop_ff_at_front:
        x_df = x_df[x_df['stop_ff_angle_at_ref'].between(-math.pi/2, math.pi/2)].copy()
        y_df = y_df.loc[x_df.index].copy().reset_index(drop=True)
        x_df.reset_index(drop=True, inplace=True)

    minimal_features_to_keep = features_to_keep_utils.get_minimal_features_to_keep(x_df, for_classification=for_classification)
    x_df = x_df[minimal_features_to_keep].copy()

    ref_columns = [column for column in x_df.columns if 'ref' in column]
    if ref_columns_only:
        x_df = x_df[ref_columns].copy()

    if cluster_to_keep == 'all':
        columns_to_delete = []
    else:
        columns_to_delete = [column for column in x_df.columns if 'cluster' in column]
    if cluster_to_keep != 'none':
        # separate clusters in cluster_to_keep by _PLUS_
        clusters_to_keep = cluster_to_keep.split('_PLUS_')
        columns_to_delete = [column for column in columns_to_delete if all([cluster not in column for cluster in clusters_to_keep])]

    if len(columns_to_delete) > 0:
        x_df.drop(columns=columns_to_delete, inplace=True)

    if use_combd_features_for_cluster_only:
        new_columns_to_delete = [column for column in x_df.columns if ('combd' not in column) &
                             ('cluster' in column)] 
        x_df.drop(columns=new_columns_to_delete, inplace=True)       

    if add_ref_interaction:
        x_df = ml_methods_utils.add_interaction_terms_to_df(x_df, specific_columns=ref_columns)

    if cluster_for_interaction != 'none':
        specific_columns = [column for column in x_df if (cluster_for_interaction in column) & ('combd' in column)]
        x_df = ml_methods_utils.add_interaction_terms_to_df(x_df, specific_columns=specific_columns)
    
    if winsorize_angle_features:
        x_df = ml_methods_utils.winsorize_x_df(x_df)

    y_var_df = y_df[[y_var_column]].copy()
    x_var_df, y_var_df = make_x_and_y_var_df(x_df, y_var_df, use_pca=use_pca)

    print('num_features_before_lasso:', x_var_df.shape[1])

    if len(x_var_df) > 0:
        if using_lasso:
            lasso = LassoCV(cv=5, tol=0.15, max_iter=400).fit(x_var_df, y_var_df.values.reshape(-1))
            # Selected variables (non-zero coefficients)
            selected_features = x_var_df.columns[(lasso.coef_ != 0)]
            if len(selected_features) == 0:
                # try again with a lower tolerance and greater max iterations
                lasso = LassoCV(cv=5, tol=0.05, max_iter=700).fit(x_var_df, y_var_df.values.reshape(-1))
                # Selected variables (non-zero coefficients)
                selected_features = x_var_df.columns[(lasso.coef_ != 0)]           
            x_var_df = x_var_df[selected_features].copy()

    print('num_features_after_lasso:', x_var_df.shape[1])
    return x_var_df, y_var_df


def make_x_and_y_var_df(x_df, y_df, drop_na=True, scale_x_var=True, use_pca=False, n_components_for_pca=None):
    x_var_df = x_df.copy()
    y_var_df = y_df.copy()
    
    # scale the variables
    if scale_x_var:
        sc = StandardScaler()
        sc.fit(x_var_df)
        columns = x_var_df.columns
        index = x_var_df.index
        x_var_df = sc.transform(x_var_df)
        x_var_df = pd.DataFrame(x_var_df, columns=columns, index=index)

    if drop_na:
        if len(y_var_df.shape) > 1:
            if y_var_df.shape[1] > 1:
                x_var_df, y_var_df = plan_factors_utils.drop_na_in_x_var(x_var_df, y_var_df)
            else:
                x_var_df, y_var_df = plan_factors_utils.drop_na_in_x_and_y_var(x_var_df, y_var_df)
        else:
            x_var_df, y_var_df = plan_factors_utils.drop_na_in_x_and_y_var(x_var_df, y_var_df)

    if use_pca:
        if n_components_for_pca is None:
            n_components_for_pca = min(x_var_df.shape[0], x_var_df.shape[1])
        pca = PCA(n_components=n_components_for_pca)  # 'mle' automatically selects the number of components or choose a fixed number
        x_var_df = pca.fit_transform(x_var_df)
        x_var_df = pd.DataFrame(x_var_df, columns=[f'x{i+1}' for i in range(n_components_for_pca)])

    x_var_df = sm.add_constant(x_var_df)
    x_var_df.reset_index(drop=True, inplace=True)
    y_var_df.reset_index(drop=True, inplace=True)
    
    return x_var_df, y_var_df


def further_prepare_x_var_and_y_var(x_var_df, y_var_df, y_var_column='d_monkey_angle_since_stop_ff_first_seen', remove_outliers=True):

    x_var_df = x_var_df.reset_index(drop=True)
    y_var_df = y_var_df.reset_index(drop=True)
    y_var = y_var_df[y_var_column].copy()
    x_var = x_var_df.copy()

    if remove_outliers:
        # remove rows in y_var_df that are 3 std above the mean, and teh corresponding rows in x_var_df
        x_var, y_var = show_planning_utils.remove_outliers(x_var, y_var)
    
    return x_var, y_var


def get_significant_features_in_one_row(summary_df, max_features_to_save=None, add_coeff=True):
    summary_df = summary_df.copy()
    summary_df.rename(columns={'index': 'feature'}, inplace=True)
    if max_features_to_save is not None:
        summary_df = summary_df.set_index('rank_by_abs_coeff').iloc[:max_features_to_save].copy()
    summary_df.index = summary_df.index.astype(str)
    temp_info = summary_df[['feature']].T.reset_index(drop=True).copy() 
    if add_coeff:
        temp_info2 = summary_df[['Coefficient']].copy()
        temp_info2.index = 'coeff_' + np.array(summary_df.index.astype(str))
        temp_info2 = temp_info2.T.reset_index(drop=True)
        temp_info = pd.concat([temp_info, temp_info2], axis=1)

    if temp_info.shape[0] > 1:
        raise ValueError('temp_info should only have one row')
    temp_info.columns.name = ''  
    return temp_info



def run_cca(X1, X2, n_comp=5, n_splits=5, show_plots=True):

    # Drop rows with NA in either X1 or X2
    X1 = X1.copy()
    X2 = X2.copy()
    X1, X2 = plan_factors_utils.drop_na_in_x_and_y_var(X1, X2)

    # Initialize KFold
    kf = KFold(n_splits=n_splits)

    # Number of components for CCA
    n_comp = min(X1.shape[1], X2.shape[1], n_comp)
    
    # Store canonical correlations and loadings from each fold
    all_canon_corrs = []
    all_x_loadings = []
    all_y_loadings = []

    for train_index, test_index in kf.split(X1):
        # Split data into training and testing sets
        X1_train, X1_test = X1.iloc[train_index], X1.iloc[test_index]
        X2_train, X2_test = X2.iloc[train_index], X2.iloc[test_index]
        
        # Initialize and fit scaler on training data
        scaler = StandardScaler()
        X1_train_sc = scaler.fit_transform(X1_train)
        X1_test_sc = scaler.transform(X1_test)

        X2_train_sc = scaler.fit_transform(X2_train)
        X2_test_sc = scaler.transform(X2_test)
        
        # Define and fit CCA on scaled training data
        cca = CCA(scale=False, n_components=n_comp).fit(X1_train_sc, X2_train_sc)
        
        # Store loadings
        all_x_loadings.append(cca.x_loadings_)
        all_y_loadings.append(cca.y_loadings_)
        
        # Transform test datasets to obtain canonical variates
        X1_test_c, X2_test_c = cca.transform(X1_test_sc, X2_test_sc)
        
        # Calculate and store canonical correlations for this fold
        canon_corr = [np.corrcoef(X1_test_c[:, i], X2_test_c[:, i])[1][0] for i in range(n_comp)]
        all_canon_corrs.append(canon_corr)

    # Calculate average canonical correlations and loadings across all folds
    avg_canon_corrs = np.mean(all_canon_corrs, axis=0)
    avg_x_loadings = np.mean(all_x_loadings, axis=0)
    avg_y_loadings = np.mean(all_y_loadings, axis=0)

    if show_plots:
        plot_correlation_coefficients(avg_canon_corrs, n_comp)
    return avg_x_loadings, avg_y_loadings, avg_canon_corrs


def plot_correlation_coefficients(avg_canon_corrs):
    # Plot average canonical correlations
    bar_names = [f'CC {i+1}' for i in range(len(avg_canon_corrs))]
    plt.bar(bar_names, avg_canon_corrs, color='lightgrey', width=0.8, edgecolor='k')

    # Label y value on each bar
    for i, val in enumerate(avg_canon_corrs):
        plt.text(i, val, f'{val:.2f}', ha='center', va='bottom')

    plt.title('Average Canonical Correlations Across Folds')
    plt.show()
    return 


def plot_x_loadings(avg_x_loadings, avg_canon_corrs, X1):

    squared_loading = pd.DataFrame(np.round(avg_x_loadings**2, 3))
    squared_loading['feature'] = X1.columns 
    squared_loading['feature_category'] = squared_loading['feature']

    num_variates = avg_x_loadings.shape[1]
    plot_modeling_result.make_a_series_of_barplots_of_ranked_loadings_or_weights(squared_loading, avg_canon_corrs, num_variates, keep_one_value_for_each_feature = True, max_features_to_show_per_plot = 20)
    return 


def plot_y_loadings(avg_y_loadings, avg_canon_corrs, X2):

    squared_loading = pd.DataFrame(np.round(avg_y_loadings**2, 3))
    squared_loading['feature'] = X2.columns 
    squared_loading['feature_category'] = squared_loading['feature']

    num_variates = avg_y_loadings.shape[1]
    plot_modeling_result.make_a_series_of_barplots_of_ranked_loadings_or_weights(squared_loading, avg_canon_corrs, num_variates, keep_one_value_for_each_feature = True, max_features_to_show_per_plot = 5)
    return 