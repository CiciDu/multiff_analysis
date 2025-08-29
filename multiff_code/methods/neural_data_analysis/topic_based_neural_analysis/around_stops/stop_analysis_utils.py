

import numpy as np

def filter_no_capture_stops_vectorized(no_capture_stops_df, ff_caught_T_new, capture_match_window):
    cap = np.sort(np.asarray(ff_caught_T_new))
    t = no_capture_stops_df["time"].to_numpy()

    # For each stop time t, find insertion point among capture times (all at once)
    idx = np.searchsorted(cap, t, side="left")  # shape (N_stops,)

    # Distances to nearest capture on the left and right (vectorized)
    left_dt = np.full(t.shape, np.inf, dtype=float)
    right_dt = np.full(t.shape, np.inf, dtype=float)

    valid_left = idx > 0
    valid_right = idx < cap.size

    left_dt[valid_left]  = np.abs(t[valid_left] - cap[idx[valid_left]-1])
    right_dt[valid_right] = np.abs(cap[idx[valid_right]] - t[valid_right])

    min_dt = np.minimum(left_dt, right_dt)

    keep_mask = min_dt >= capture_match_window
    
    df_filtered = no_capture_stops_df[keep_mask].reset_index(drop=True)
    return df_filtered


