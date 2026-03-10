import pandas as pd

import numpy as np

def clean_binary_and_drop_constant(df, tol=1e-8):
    df = df.copy()
    drop_cols = []

    for col in df.columns:
        s = df[col]
        vals = s.dropna().values

        # quantize to remove floating noise
        vals_q = np.round(vals / tol) * tol
        unique_vals = np.unique(vals_q)

        # constant column
        if unique_vals.size <= 1:
            drop_cols.append(col)
            continue

        # exactly two unique values → map to {0,1}
        if unique_vals.size == 2:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}

            s_q = np.round(s / tol) * tol
            df[col] = s_q.map(mapping)

    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df