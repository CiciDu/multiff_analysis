import pandas as pd
from typing import Optional
from planning_analysis.show_planning.cur_vs_nxt_ff import cvn_from_ref_class


def get_test_heading_df(raw_data_folder_path):
    cvn = cvn_from_ref_class.CurVsNxtFfFromRefClass(
        raw_data_folder_path=raw_data_folder_path)
    # Quick method - tries to retrieve first, creates if needed
    cvn.make_heading_info_df_without_long_process(
        test_or_control='test',  # or 'control'
        ref_point_mode='distance',  # or 'time after cur ff visible'
        ref_point_value=-100,  # or 0.0 for time mode
        heading_info_df_exists_ok=True,  # Set to False to force recreation
        stops_near_ff_df_exists_ok=True,
        save_data=True
    )

    # Access the result
    heading_info_df = cvn.heading_info_df
    heading_df = heading_info_df[['cur_ff_index',
                                  'diff_in_abs_angle_to_nxt_ff']].copy()
    heading_df = heading_df.sort_values(
        by='diff_in_abs_angle_to_nxt_ff', ascending=False).reset_index(drop=True)

    return heading_info_df, heading_df


def select_ff_subset(
    heading_df: pd.DataFrame,
    rebinned_x_var: pd.DataFrame,
    rebinned_y_var: pd.DataFrame,
    top: bool = True,
    n: Optional[int] = None,
    pct: Optional[float] = None,
    col: str = 'cur_ff_index',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select a subset of trials (top or bottom fraction/number by heading_df order) 
    and apply the same filter to both X and Y re-binned variables.
    """
    total = len(heading_df)
    if pct is not None:
        k = int(total * pct)
    elif n is not None:
        k = n
    else:
        raise ValueError("Must provide either `n` or `pct`.")

    if top:
        selected_ff = heading_df.iloc[:k][col].to_numpy()
    else:
        selected_ff = heading_df.iloc[-k:][col].to_numpy()

    direction = 'top' if top else 'bottom'
    print(
        f'Selecting {k} rows from the {direction} of heading_df (total={total})')

    print('rebinned_y_var.shape (before):', rebinned_y_var.shape)
    mask = rebinned_y_var[col].isin(selected_ff)
    rebinned_y_var_filt = rebinned_y_var[mask]
    rebinned_x_var_filt = rebinned_x_var[mask]
    print('rebinned_y_var.shape (after):', rebinned_y_var_filt.shape)

    return rebinned_x_var_filt, rebinned_y_var_filt


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are constant (only one unique value)."""
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if len(constant_cols) > 0:
        print('Number of columns (before):', df.shape[1])
        df = df.drop(columns=constant_cols)
        print('Number of columns (after):', df.shape[1])
    return df


def select_ff_subset_by_dir_from_cur_ff_same_side(
    heading_df: pd.DataFrame,
    rebinned_x_var: pd.DataFrame,
    rebinned_y_var: pd.DataFrame,
    same_side=True,

) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select a subset of trials (top or bottom by heading_df order) 
    and apply the same filter to both X and Y re-binned variables.
    """
    if same_side:
        selected_ff = heading_df.loc[heading_df['dir_from_cur_ff_same_side']
                                     == 1, 'cur_ff_index'].to_numpy()
    else:
        selected_ff = heading_df.loc[heading_df['dir_from_cur_ff_same_side']
                                     == 0, 'cur_ff_index'].to_numpy()

    print('rebinned_y_var.shape (before):', rebinned_y_var.shape)
    mask = rebinned_y_var['cur_ff_index'].isin(selected_ff)
    rebinned_y_var_filt = rebinned_y_var[mask]
    rebinned_x_var_filt = rebinned_x_var[mask]
    print('rebinned_y_var.shape (after):', rebinned_y_var_filt.shape)

    return rebinned_x_var_filt, rebinned_y_var_filt
