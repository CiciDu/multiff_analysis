
import numpy as np

from neural_data_analysis.topic_based_neural_analysis.target_decoder import prep_target_decoder
from neural_data_analysis.neural_analysis_tools.model_neural_data import drop_high_corr_vars, drop_high_vif_vars

def get_strong_correlations(design_df, threshold=0.9):
    # Keep only numeric columns
    numeric_df = design_df.select_dtypes(include=[np.number])
    
    # Compute correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Unstack to long format
    corr_long = (
        corr_matrix
        .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # upper triangle only
        .stack()
        .reset_index()
    )
    
    corr_long.columns = ['var1', 'var2', 'correlation']
    
    # Filter by threshold
    strong_corr_df = corr_long.loc[corr_long['correlation'].abs() >= threshold]
    
    # Sort by absolute correlation
    strong_corr_df = strong_corr_df.reindex(
        strong_corr_df['correlation'].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
    
    return strong_corr_df


def reduce_encoding_design(df, corr_threshold_for_lags=0.99, 
                         vif_threshold=None, 
                         verbose=True):
        
    df_reduced_initial = prep_target_decoder.remove_zero_var_cols(
        df)

    # Call the function to iteratively drop lags with high correlation for each feature
    df_reduced = drop_high_corr_vars.drop_columns_with_high_corr(df_reduced_initial,
                                                                                corr_threshold_for_lags=corr_threshold_for_lags,
                                                                                verbose=verbose,
                                                                                filter_by_feature=False,
                                                                                filter_by_subsets=False,
                                                                                filter_by_all_columns=True)

    if vif_threshold is not None:
        df_reduced = drop_high_vif_vars.drop_columns_with_high_vif(df_reduced,
                                                                            vif_threshold=vif_threshold,
                                                                            verbose=verbose,
                                                                            filter_by_feature=False,
                                                                            filter_by_subsets=False,
                                                                            filter_by_all_columns=True,
                                                                            get_column_subsets_func=False)
    
    return df_reduced
