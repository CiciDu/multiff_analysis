import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error


from itertools import product

def add_interaction_terms_and_features(concat_behav_trials):
    cols_added = []
    for var_a, var_b in product(
            ['cur_vis', 'nxt_vis', 'nxt_in_memory'],
            ['cur_ff_distance', 'nxt_ff_distance']):
        col_name = f'{var_a}*{var_b}'
        concat_behav_trials[col_name] = (
            concat_behav_trials[var_a] * concat_behav_trials[var_b]
        )
        cols_added.append(col_name)

    return concat_behav_trials, cols_added


def prep_behav(df,
                cont_cols=('cur_ff_distance', 'nxt_ff_distance', 'time_since_last_capture'),
                cat_vars=('cur_vis', 'nxt_vis', 'nxt_in_memory', 'any_ff_visible')):
    # keep only requested features (that exist), copy to avoid side effects
    out = df.copy()
    added_cols = []

    # add log1p features (clip negatives to 0 to keep log1p valid)
    for c in cont_cols:
        if c in out.columns:
            out[f'log1p_{c}'] = np.log1p(pd.to_numeric(out[c], errors='coerce').clip(lower=0))
            added_cols.append(f'log1p_{c}')

    # binarize categorical indicators ( > 0 → 1; else 0 )
    for v in cat_vars:
        if v in out.columns:
            x = pd.to_numeric(out[v], errors='coerce')
            out[v] = (x.fillna(0) > 0).astype('int8')
            added_cols.append(v)
    return out, added_cols


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

def eval_decoder(y_true, y_pred, title='Decoder fit', sample_ts=None, n_show=500):
    """
    Back-compatible evaluation (no need for sklearn>=0.22).
    - y_true, y_pred: 1D array-like
    - sample_ts: optional 1D times for the overlay plot
    - n_show: how many samples to show in the overlay
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    # drop NaNs in a paired way
    ok = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[ok], y_pred[ok]
    if sample_ts is not None:
        sample_ts = np.asarray(sample_ts).ravel()[ok]

    # metrics (old sklearn safe)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # Pearson R (scale-invariant)
    if y_true.std() == 0 or y_pred.std() == 0:
        R = np.nan
    else:
        R = np.corrcoef(y_true, y_pred)[0, 1]

    print(f'R = {R:.3f} | R^2 = {r2:.3f} | RMSE = {rmse:.3f}')

    # scatter with unity line
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=6, alpha=0.5)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title(f'{title}  (R={R:.2f}, R^2={r2:.2f}, RMSE={rmse:.2f})')
    plt.tight_layout()

    # short time overlay
    plt.figure(figsize=(9, 3))
    idx = slice(0, min(n_show, len(y_true)))
    if sample_ts is not None:
        plt.plot(sample_ts[idx], y_true[idx], label='True', linewidth=1.2)
        plt.plot(sample_ts[idx], y_pred[idx], label='Pred', linewidth=1.2)
        plt.xlabel('Time')
    else:
        plt.plot(y_true[idx], label='True', linewidth=1.2)
        plt.plot(y_pred[idx], label='Pred', linewidth=1.2)
        plt.xlabel('Samples')
    plt.ylabel('Value')
    plt.title('First segment: true vs predicted')
    plt.legend()
    plt.tight_layout()

    return {'R': R, 'R2': r2, 'RMSE': rmse}


from dataclasses import dataclass

@dataclass
class DecodingRunConfig:
    cv_splits: list | None = None
    fast_mode: bool = False
    n_jobs: int = -1
    make_plots: bool = True
    use_early_stopping: bool = False
    od_wait: int = 20
    
def run_cv_decoding(
    X,
    y_df,
    behav_features,
    groups,
    model_factory=None,
    n_splits=5,
    config: DecodingRunConfig | None = None,
    context_label=None,
):
    if config is None:
        config = DecodingRunConfig()

    if model_factory is None:
        model_factory = lambda: CatBoostRegressor(
            iterations=150 if config.fast_mode else 300,
            depth=4 if config.fast_mode else 6,
            learning_rate=0.05,
            loss_function='RMSE',
            verbose=0,
            thread_count=2 if config.fast_mode else None,
        )

    X = np.asarray(X)
    groups_arr = np.asarray(groups)

    # Precompute CV splits once
    if config.cv_splits is None:
        gcv = GroupKFold(n_splits=n_splits)
        cv_splits = list(gcv.split(X, groups=groups_arr))
    else:
        cv_splits = config.cv_splits

    results = []

    for behav_feature in behav_features:
        if y_df[behav_feature].nunique() <= 1:
            continue

        print(behav_feature)

        y = y_df[behav_feature].to_numpy().ravel()

        ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X_ok = X[ok]
        y_ok = y[ok]

        # Filter CV splits for valid rows
        idx = np.where(ok)[0]
        idx_map = {i: j for j, i in enumerate(idx)}
        cv_splits_ok = [
            ([idx_map[i] for i in tr if i in idx_map],
             [idx_map[i] for i in te if i in idx_map])
            for tr, te in cv_splits
        ]

        model = model_factory()


        if config.use_early_stopping:
            y_cv_pred = np.full_like(y_ok, np.nan, dtype=float)

            for train_idx, test_idx in cv_splits_ok:
                X_tr, X_te = X_ok[train_idx], X_ok[test_idx]
                y_tr, y_te = y_ok[train_idx], y_ok[test_idx]

                model = model_factory()
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=(X_te, y_te),
                    use_best_model=True,
                    od_type='Iter',
                    od_wait=config.od_wait,
                    verbose=False,
                )
                y_cv_pred[test_idx] = model.predict(X_te)
        else:
            y_cv_pred = cross_val_predict(
                model,
                X_ok,
                y_ok,
                cv=cv_splits_ok,
                method='predict',
                n_jobs=config.n_jobs,
            )


        r2 = r2_score(y_ok, y_cv_pred)
        rmse = np.sqrt(mean_squared_error(y_ok, y_cv_pred))
        r = np.corrcoef(y_ok, y_cv_pred)[0, 1]

        results.append({
            'behav_feature': behav_feature,
            'n_samples': len(y_ok),
            'r2_cv': r2,
            'rmse_cv': rmse,
            'r_cv': r,
            'context': context_label,
            'model_name': model.__class__.__name__,
        })

        if config.make_plots and not config.fast_mode:
            lo = float(min(y_ok.min(), y_cv_pred.min()))
            hi = float(max(y_ok.max(), y_cv_pred.max()))

            plt.figure(figsize=(5, 5))
            plt.scatter(y_ok, y_cv_pred, s=6, alpha=0.5)
            plt.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
            plt.xlabel('True')
            plt.ylabel('Predicted (CV)')
            plt.title(
                behav_feature if context_label is None
                else f'{behav_feature} | {context_label}'
            )
            plt.tight_layout()
            plt.show()

    return pd.DataFrame(results)

def decode_by_num_ff_visible(
    x_var,
    y_var,
    behav_features,
    model_factory=None,
    group_col='new_segment',
    n_splits=5,
    config: DecodingRunConfig | None = None,
    save_path=None,
):
    if config is None:
        config = DecodingRunConfig()

    num_ff_visible_int = np.ceil(y_var['num_ff_visible'])
    all_results = []

    for num_ff in np.sort(num_ff_visible_int.unique()):
        print(f'\nnum_ff: {num_ff}')

        mask = num_ff_visible_int == num_ff
        X_sub = x_var[mask]
        y_sub = y_var.loc[mask]
        groups_sub = y_sub[group_col].values

        # Precompute splits ONCE per num_ff
        if config.cv_splits is None:
            gcv = GroupKFold(n_splits=n_splits)
            cv_splits = list(gcv.split(X_sub, groups=groups_sub))
        else:
            cv_splits = config.cv_splits

        local_config = DecodingRunConfig(
            cv_splits=cv_splits,
            fast_mode=config.fast_mode,
            n_jobs=config.n_jobs,
            make_plots=config.make_plots,
        )

        results_df = run_cv_decoding(
            X=X_sub,
            y_df=y_sub,
            behav_features=behav_features,
            groups=groups_sub,
            model_factory=model_factory,
            n_splits=n_splits,
            config=local_config,
            context_label=f'num_ff={num_ff}',
        )

        results_df['num_ff_visible'] = num_ff
        all_results.append(results_df)

    results_df = pd.concat(all_results, ignore_index=True)

    if save_path is not None:
        results_df.to_csv(save_path, index=False)

    return results_df
