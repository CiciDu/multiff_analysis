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


# ------------------------
# Constants (task-aware)
# ------------------------

STOP_EPS = 5.0
CAPTURE_RADIUS = 25.0

# Angular thresholds (radians)
DEG15 = np.deg2rad(15)
DEG45 = np.deg2rad(45)
DEG90 = np.deg2rad(90)

# Turn-rate thresholds (from quantiles)
ANG_SLOW = 0.3    # ~75th percentile
ANG_FAST = 0.8    # ~90th percentile

# Angular acceleration thresholds
ANG_ACC_SMOOTH = 1.0
ANG_ACC_JERK = 4.0


# ============================================================
# Angular speed (turn rate)
# ============================================================

def add_ang_speed_band(df):
    """
    Adds signed angular speed bands:
    LEFT_FAST / LEFT_SLOW / STRAIGHT / RIGHT_SLOW / RIGHT_FAST
    """
    omega = df['ang_speed'].values

    band = np.full(len(df), 'STRAIGHT', dtype=object)

    band[(omega < -ANG_FAST)] = 'LEFT_FAST'
    band[(omega >= -ANG_FAST) & (omega < -ANG_SLOW)] = 'LEFT_SLOW'

    band[(omega > ANG_SLOW) & (omega <= ANG_FAST)] = 'RIGHT_SLOW'
    band[(omega > ANG_FAST)] = 'RIGHT_FAST'

    df = df.copy()
    df['ang_speed_band'] = pd.Categorical(
        band,
        categories=[
            'LEFT_FAST',
            'LEFT_SLOW',
            'STRAIGHT',
            'RIGHT_SLOW',
            'RIGHT_FAST',
        ],
        ordered=True
    )

    return df


# ============================================================
# Angular acceleration (change in turning)
# ============================================================

def add_ang_accel_band(df):
    """
    Adds signed angular acceleration bands:
    LEFT_JERK / LEFT_SMOOTH / NEUTRAL / RIGHT_SMOOTH / RIGHT_JERK
    """
    a = df['ang_accel'].values

    band = np.full(len(df), 'NEUTRAL', dtype=object)

    band[a < -ANG_ACC_JERK] = 'LEFT_JERK'
    band[(a >= -ANG_ACC_JERK) & (a < -ANG_ACC_SMOOTH)] = 'LEFT_SMOOTH'

    band[(a > ANG_ACC_SMOOTH) & (a <= ANG_ACC_JERK)] = 'RIGHT_SMOOTH'
    band[a > ANG_ACC_JERK] = 'RIGHT_JERK'

    df = df.copy()
    df['ang_accel_band'] = pd.Categorical(
        band,
        categories=[
            'LEFT_JERK',
            'LEFT_SMOOTH',
            'NEUTRAL',
            'RIGHT_SMOOTH',
            'RIGHT_JERK',
        ],
        ordered=True
    )

    return df


# ============================================================
# Next firefly angle (planning-relevant, coarse)
# ============================================================

def add_nxt_ff_angle_band(df):
    """
    Adds coarse signed angle-to-next-FF bands:
    LEFT / AHEAD / RIGHT / BEHIND
    """
    theta = df['nxt_ff_angle'].values
    abs_theta = np.abs(theta)

    band = np.full(len(df), 'AHEAD', dtype=object)

    band[(abs_theta > DEG90)] = 'BEHIND'
    band[(theta < -DEG15) & (abs_theta <= DEG90)] = 'LEFT'
    band[(theta > DEG15) & (abs_theta <= DEG90)] = 'RIGHT'

    df = df.copy()
    df['nxt_ff_angle_band'] = pd.Categorical(
        band,
        categories=['LEFT', 'AHEAD', 'RIGHT', 'BEHIND'],
        ordered=True
    )

    return df


# ============================================================
# Egocentric lateral offset (cur FF)
# ============================================================

def add_cur_ff_rel_x_band(df):
    """
    Adds lateral offset bands for current FF:
    LEFT_FAR / LEFT_NEAR / CENTER / RIGHT_NEAR / RIGHT_FAR
    """
    x = df['cur_ff_rel_x'].values

    band = np.full(len(df), 'CENTER', dtype=object)

    band[x < -50] = 'LEFT_FAR'
    band[(x >= -50) & (x < -20)] = 'LEFT_NEAR'

    band[(x > 20) & (x <= 50)] = 'RIGHT_NEAR'
    band[x > 50] = 'RIGHT_FAR'

    df = df.copy()
    df['cur_ff_rel_x_band'] = pd.Categorical(
        band,
        categories=[
            'LEFT_FAR',
            'LEFT_NEAR',
            'CENTER',
            'RIGHT_NEAR',
            'RIGHT_FAR',
        ],
        ordered=True
    )

    return df


# ============================================================
# Egocentric forward offset (cur FF)
# ============================================================

def add_cur_ff_rel_y_band(df):
    """
    Adds forward-distance bands for current FF:
    VERY_NEAR / NEAR / MID / FAR
    """
    y = df['cur_ff_rel_y'].values

    band = np.full(len(df), 'MID', dtype=object)

    band[y <= CAPTURE_RADIUS] = 'VERY_NEAR'
    band[(y > CAPTURE_RADIUS) & (y <= 80)] = 'NEAR'
    band[y > 180] = 'FAR'

    df = df.copy()
    df['cur_ff_rel_y_band'] = pd.Categorical(
        band,
        categories=['VERY_NEAR', 'NEAR', 'MID', 'FAR'],
        ordered=True
    )

    return df


# ============================================================
# Distance at reference (commitment timing)
# ============================================================

def add_cur_ff_distance_at_ref_band(df):
    """
    Adds commitment-distance bands:
    EARLY_COMMIT / MID_COMMIT / LATE_COMMIT
    """
    d = df['cur_ff_distance_at_ref'].values

    band = np.full(len(df), 'MID_COMMIT', dtype=object)

    band[d <= 130] = 'EARLY_COMMIT'
    band[d > 200] = 'LATE_COMMIT'

    df = df.copy()
    df['cur_ff_dist_ref_band'] = pd.Categorical(
        band,
        categories=['EARLY_COMMIT', 'MID_COMMIT', 'LATE_COMMIT'],
        ordered=True
    )

    return df


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
    Adds signed angle-to-current-FF bands.

    Categories (ordered):
    LEFT_BEHIND / LEFT_LARGE / LEFT_SLIGHT / AHEAD /
    RIGHT_SLIGHT / RIGHT_LARGE / RIGHT_BEHIND

    Assumes cur_ff_angle is signed radians:
        negative = left, positive = right
    """
    theta = df['cur_ff_angle'].values

    angle_band = np.full(len(df), 'AHEAD', dtype=object)

    # Ahead
    angle_band[np.abs(theta) <= DEG15] = 'AHEAD'

    # Left side
    angle_band[(theta < -DEG15) & (theta >= -DEG45)] = 'LEFT_SLIGHT'
    angle_band[(theta < -DEG45) & (theta >= -DEG90)] = 'LEFT_LARGE'
    angle_band[theta < -DEG90] = 'LEFT_BEHIND'

    # Right side
    angle_band[(theta > DEG15) & (theta <= DEG45)] = 'RIGHT_SLIGHT'
    angle_band[(theta > DEG45) & (theta <= DEG90)] = 'RIGHT_LARGE'
    angle_band[theta > DEG90] = 'RIGHT_BEHIND'

    df = df.copy()
    df['cur_ff_angle_band'] = pd.Categorical(
        angle_band,
        categories=[
            'LEFT_BEHIND',
            'LEFT_LARGE',
            'LEFT_SLIGHT',
            'AHEAD',
            'RIGHT_SLIGHT',
            'RIGHT_LARGE',
            'RIGHT_BEHIND',
        ],
        ordered=True
    )

    return df


def add_log1p_cur_ff_distance_band(df):
    """
    Adds fine-grained current-firefly distance bands:
    VERY_NEAR / NEAR / MID / FAR
    """
    d = df['log1p_cur_ff_distance'].values

    band = np.full(len(df), 'MID', dtype=object)

    band[d <= 2.74800] = 'VERY_NEAR'     # 5%
    band[(d > 2.74800) & (d <= 3.88725)] = 'NEAR'  # 25%
    band[d > 5.40585] = 'FAR'            # 90%

    df = df.copy()
    df['log1p_cur_ff_distance_band'] = pd.Categorical(
        band,
        categories=['VERY_NEAR', 'NEAR', 'MID', 'FAR'],
        ordered=True
    )

    return df


def add_log1p_nxt_ff_distance_band(df):
    """
    Adds next-firefly distance bands based on log1p distance:
    NEAR / MID / FAR
    """
    d = df['log1p_nxt_ff_distance'].values

    band = np.full(len(df), 'MID', dtype=object)

    band[d <= 5.02961] = 'NEAR'
    band[d > 6.21049] = 'FAR'

    df = df.copy()
    df['log1p_nxt_ff_distance_band'] = pd.Categorical(
        band,
        categories=['NEAR', 'MID', 'FAR'],
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
    df = add_ang_speed_band(df)
    df = add_ang_accel_band(df)
    df = add_nxt_ff_angle_band(df)
    df = add_cur_ff_rel_x_band(df)
    df = add_cur_ff_rel_y_band(df)
    df = add_cur_ff_distance_at_ref_band(df)
    df = add_log1p_cur_ff_distance_band(df)
    df = add_log1p_nxt_ff_distance_band(df)

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
