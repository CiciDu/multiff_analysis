from machine_learning.ml_methods import prep_ml_data_utils
from planning_analysis.show_planning import show_planning_utils
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


def add_interaction_terms_to_df(df, specific_columns=None):

    if specific_columns is not None:
        if len(specific_columns) > 0:
            # Assuming df is your DataFrame and specific_columns is a list of column names
            df_selected = df[specific_columns]

            # Initialize PolynomialFeatures
            poly = PolynomialFeatures(
                degree=2, interaction_only=False, include_bias=False)

            # Fit and transform the selected DataFrame
            df_interactions = poly.fit_transform(df_selected)

            # Generate new column names (including original and interaction terms)
            new_column_names = poly.get_feature_names_out(
                input_features=df_selected.columns)

            # Create a new DataFrame with the interaction terms and new column names
            df_with_interactions = pd.DataFrame(
                df_interactions, columns=new_column_names)

            # df_with_interactions now contains the original columns, squared terms, and interaction terms with appropriate names

            # drop the original columns
            df_with_interactions.drop(columns=specific_columns, inplace=True)
            df_with_interactions.index = df.index

            df = pd.concat([df, df_with_interactions], axis=1)
            print('Added interaction terms.')
    return df


def make_x_and_y_var_df(x_df, y_df, drop_na=True, scale_x_var=True, use_pca=False, n_components_for_pca=None):
    x_var_df = x_df.copy()
    y_var_df = y_df.copy()
    pca = None

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
                x_var_df, y_var_df = prep_ml_data_utils.drop_na_in_x_var(
                    x_var_df, y_var_df)
                print(
                    'Dropped rows with NA only in x_var_df (and the corresponding rows in y_var_df).')
            else:
                x_var_df, y_var_df = prep_ml_data_utils.drop_na_in_x_and_y_var(
                    x_var_df, y_var_df)
                print('Dropped rows with NA in both x_var_df and y_var_df.')
        else:
            x_var_df, y_var_df = prep_ml_data_utils.drop_na_in_x_and_y_var(
                x_var_df, y_var_df)
            print('Dropped rows with NA in y_var_df.')

    if use_pca:
        if n_components_for_pca is None:
            n_components_for_pca = min(x_var_df.shape[1], 10)
        # 'mle' automatically selects the number of components or choose a fixed number
        pca = PCA(n_components=n_components_for_pca)
        x_var_df = pca.fit_transform(x_var_df)
        x_var_df = pd.DataFrame(
            x_var_df, columns=[f'x{i+1}' for i in range(n_components_for_pca)])

    x_var_df = sm.add_constant(x_var_df)
    x_var_df.reset_index(drop=True, inplace=True)
    y_var_df.reset_index(drop=True, inplace=True)

    return x_var_df, y_var_df, pca


def further_prepare_x_var_and_y_var(x_var_df, y_var_df, y_var_column='d_monkey_angle_since_cur_ff_first_seen', remove_outliers=True):

    if y_var_column not in y_var_df.columns:
        raise ValueError(
            f'{y_var_column} is not in the y_var_df.columns. Please check the y_var_column.')

    x_var_df = x_var_df.reset_index(drop=True)
    y_var_df = y_var_df.reset_index(drop=True)
    y_var = y_var_df[y_var_column].copy()
    x_var = x_var_df.copy()

    if remove_outliers:
        # remove rows in y_var_df that are 3 std above the mean, and teh corresponding rows in x_var_df
        x_var, y_var = show_planning_utils.remove_outliers(x_var, y_var)

    return x_var, y_var


def drop_na_in_x_var(x_var_df, y_var_df):
    x_var_df.reset_index(drop=True, inplace=True)
    y_var_df.reset_index(drop=True, inplace=True)

    if x_var_df.isnull().any(axis=1).sum() > 0:
        print(
            f'Number of rows with NaN values in x_var_df: {x_var_df.isnull().any(axis=1).sum()} out of {x_var_df.shape[0]} rows. The rows with NaN values will be dropped.')
        # drop rows with NA in x_var_df
        x_var_df = x_var_df.dropna()
        y_var_df = y_var_df.loc[x_var_df.index].copy()

    x_var_df.reset_index(drop=True, inplace=True)
    y_var_df.reset_index(drop=True, inplace=True)
    return x_var_df, y_var_df


def drop_na_in_x_and_y_var(x_var_df, y_var_df):
    x_var_df.reset_index(drop=True, inplace=True)
    y_var_df.reset_index(drop=True, inplace=True)
    
    # if y_var_df is one dimensional, convert it to a dataframe
    if len(y_var_df.shape) == 1:
        y_var_df = pd.DataFrame(y_var_df)

    if x_var_df.isnull().any(axis=1).sum() > 0:
        print(
            f'Number of rows with NaN values in x_var_df: {x_var_df.isnull().any(axis=1).sum()} out of {x_var_df.shape[0]} rows. The rows with NaN values will be dropped.')
        # drop rows with NA in x_var_df
        x_var_df = x_var_df.dropna()
        y_var_df = y_var_df.loc[x_var_df.index].copy()
    if y_var_df.isnull().any(axis=1).sum() > 0:
        print(
            f'Number of rows with NaN values in y_var_df (after cleaning x_var_df and the corresponding rows in y_var_df): {y_var_df.isnull().any(axis=1).sum()} out of {y_var_df.shape[0]} rows. The rows with NaN values will be dropped.')
        # drop rows with NA in y_var_df
        y_var_df = y_var_df.dropna()
        x_var_df = x_var_df.loc[y_var_df.index].copy()

    x_var_df.reset_index(drop=True, inplace=True)
    y_var_df.reset_index(drop=True, inplace=True)
    return x_var_df, y_var_df
