import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# ------------------------
# Constants (task-aware)
# ------------------------

STOP_EPS = 5          # numerical zero for speed
CAPTURE_RADIUS = 25.0   # cm

DEG15 = np.deg2rad(15)
DEG45 = np.deg2rad(45)
DEG90 = np.deg2rad(90)


def get_neural_feature_columns(df):
    """
    Returns sorted neural feature columns: unit_0, unit_1, ...
    """
    return sorted([c for c in df.columns if c.startswith('unit_')])


def add_speed_band(df):
    """
    Adds categorical speed bands:
    STOP / SLOW / CRUISE / FAST
    """
    speed = df['speed'].values

    speed_band = np.full(len(df), 'CRUISE', dtype=object)

    speed_band[np.abs(speed) <= STOP_EPS] = 'STOP'
    speed_band[(speed > STOP_EPS) & (speed <= 40)] = 'SLOW'
    speed_band[(speed > 170)] = 'FAST'

    df = df.copy()
    df['speed_band'] = pd.Categorical(
        speed_band,
        categories=['STOP', 'SLOW', 'CRUISE', 'FAST'],
        ordered=True
    )

    return df


def add_accel_band(df):
    """
    Adds acceleration control bands:
    HARD_BRAKE / BRAKE / NEUTRAL / ACCEL / HARD_ACCEL
    """
    accel = df['accel'].values

    accel_band = np.full(len(df), 'NEUTRAL', dtype=object)

    accel_band[accel < -700] = 'HARD_BRAKE'
    accel_band[(accel >= -700) & (accel < -50)] = 'BRAKE'
    accel_band[(accel > 50) & (accel <= 700)] = 'ACCEL'
    accel_band[accel > 700] = 'HARD_ACCEL'

    df = df.copy()
    df['accel_band'] = pd.Categorical(
        accel_band,
        categories=['HARD_BRAKE', 'BRAKE', 'NEUTRAL', 'ACCEL', 'HARD_ACCEL'],
        ordered=True
    )

    return df


def add_cur_ff_distance_band(df):
    """
    Adds distance-to-current-FF bands:
    VERY_NEAR / NEAR / MID / FAR
    """
    d = df['cur_ff_distance'].values

    dist_band = np.full(len(df), 'MID', dtype=object)

    dist_band[d <= CAPTURE_RADIUS] = 'VERY_NEAR'
    dist_band[(d > CAPTURE_RADIUS) & (d <= 80)] = 'NEAR'
    dist_band[d > 170] = 'FAR'

    df = df.copy()
    df['cur_ff_dist_band'] = pd.Categorical(
        dist_band,
        categories=['VERY_NEAR', 'NEAR', 'MID', 'FAR'],
        ordered=True
    )

    return df


def add_nxt_ff_distance_band(df):
    """
    Adds distance-to-next-FF bands:
    CLOSE / MID / FAR
    """
    d = df['nxt_ff_distance'].values

    nxt_band = np.full(len(df), 'MID', dtype=object)

    nxt_band[d <= 200] = 'CLOSE'
    nxt_band[d > 400] = 'FAR'

    df = df.copy()
    df['nxt_ff_dist_band'] = pd.Categorical(
        nxt_band,
        categories=['CLOSE', 'MID', 'FAR'],
        ordered=True
    )

    return df


def add_cur_ff_angle_band(df):
    """
    Adds binned angle-to-current-FF bands:
    AHEAD / SLIGHT_TURN / LARGE_TURN / BEHIND

    Assumes cur_ff_angle is signed radians.
    """
    theta = np.abs(df['cur_ff_angle'].values)

    angle_band = np.full(len(df), 'LARGE_TURN', dtype=object)

    angle_band[theta <= DEG15] = 'AHEAD'
    angle_band[(theta > DEG15) & (theta <= DEG45)] = 'SLIGHT_TURN'
    angle_band[(theta > DEG45) & (theta <= DEG90)] = 'LARGE_TURN'
    angle_band[theta > DEG90] = 'BEHIND'

    df = df.copy()
    df['cur_ff_angle_band'] = pd.Categorical(
        angle_band,
        categories=['AHEAD', 'SLIGHT_TURN', 'LARGE_TURN', 'BEHIND'],
        ordered=True
    )

    return df


def add_behavior_bands(df):
    """
    Convenience wrapper: adds all behavioral bands.
    """
    df = add_speed_band(df)
    df = add_accel_band(df)
    df = add_cur_ff_distance_band(df)
    df = add_nxt_ff_distance_band(df)
    df = add_cur_ff_angle_band(df)

    return df


def add_speed_angle_interaction(df):
    """
    Adds joint speed × angle interaction label.
    """
    df = df.copy()

    df['speed_angle_state'] = (
        df['speed_band'].astype(str) + '__' +
        df['cur_ff_angle_band'].astype(str)
    )

    df['speed_angle_state'] = pd.Categorical(df['speed_angle_state'])

    return df


def add_speed_angle_distance_interaction(df):
    """
    Adds speed × angle × current-FF-distance interaction label.
    """
    df = df.copy()

    df['speed_angle_dist_state'] = (
        df['speed_band'].astype(str) + '__' +
        df['cur_ff_angle_band'].astype(str) + '__' +
        df['cur_ff_dist_band'].astype(str)
    )

    df['speed_angle_dist_state'] = pd.Categorical(
        df['speed_angle_dist_state']
    )

    return df


def add_planning_interaction(df):
    """
    Adds interaction probing anticipation of next FF.
    """
    df = df.copy()

    df['planning_state'] = (
        df['speed_band'].astype(str) + '__' +
        df['cur_ff_angle_band'].astype(str) + '__' +
        df['nxt_ff_dist_band'].astype(str)
    )

    df['planning_state'] = pd.Categorical(df['planning_state'])

    return df


def prune_rare_states(df, label_col, min_count=20):
    """
    Drops rows belonging to rare interaction states.
    """
    counts = df[label_col].value_counts()
    keep_states = counts[counts >= min_count].index

    return df[df[label_col].isin(keep_states)].copy()


def prune_rare_states_two_dfs(
    df_behavior,
    df_neural,
    label_col,
    min_count=200,
):
    """
    Prune rare interaction states while keeping two DataFrames aligned.

    Assumes df_behavior and df_neural have identical row order.
    """
    # Sanity check
    assert len(df_behavior) == len(df_neural), 'DataFrames must align row-wise'

    # Compute which states to keep
    state_counts = df_behavior[label_col].value_counts()
    keep_states = state_counts[state_counts >= min_count].index

    # Build boolean mask
    keep_mask = df_behavior[label_col].isin(keep_states).values

    # Apply mask to both
    df_behavior_pruned = df_behavior.loc[keep_mask].reset_index(drop=True)
    df_neural_pruned = df_neural.loc[keep_mask].reset_index(drop=True)

    return df_behavior_pruned, df_neural_pruned


def add_pairwise_interaction(
    *,
    df,
    var_a,
    var_b,
    new_col,
    sep='__',
    drop_unused_categories=False,
):
    """
    Add a categorical interaction label between two variables.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    var_a : str
        First variable name (e.g. 'speed_band')
    var_b : str
        Second variable name (e.g. 'cur_ff_angle_band')
    new_col : str
        Name of the new interaction column
    sep : str
        Separator between variable levels
    drop_unused_categories : bool
        If True, remove unused categories after construction
    """
    df = df.copy()

    # Construct interaction label
    df[new_col] = (
        df[var_a].astype(str) + sep +
        df[var_b].astype(str)
    )

    df[new_col] = pd.Categorical(df[new_col])

    if drop_unused_categories:
        df[new_col] = df[new_col].cat.remove_unused_categories()

    return df


def add_pairwise_interactions_from_spec(
    *,
    df,
    interaction_specs,
    sep='__',
    drop_unused_categories=False,
):
    """
    Add multiple pairwise interaction columns from an explicit spec list.

    Parameters
    ----------
    df : pandas.DataFrame
    interaction_specs : list of (var_a, var_b, new_col)
    sep : str
    drop_unused_categories : bool

    Returns
    -------
    df : pandas.DataFrame
    added_cols : list of str
    """
    df = df.copy()
    added_cols = []

    for var_a, var_b, new_col in interaction_specs:
        df = add_pairwise_interaction(
            df=df,
            var_a=var_a,
            var_b=var_b,
            new_col=new_col,
            sep=sep,
            drop_unused_categories=drop_unused_categories,
        )
        added_cols.append(new_col)

    return df, added_cols

