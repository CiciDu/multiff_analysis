import os, time, json, inspect
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import sklearn

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    HuberRegressor, RANSACRegressor, SGDRegressor
)
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor

def use_advanced_ml_model_for_regression(
    X_train, y_train, X_test, y_test,
    model_names=(
        'linreg','ridge','lasso','elastic',
        'linear_svr','svr','knn',
        'dt','rf','extra_trees',
        'grad_boosting','hist_gb','boosting','bagging',
        'huber','ransac','sgd','pls','mlp'   # <- MLP included
    ),
    use_cv=True,
    cv_splits=5,
    random_state=42,
    verbose=True,
    save_dir=None,   # e.g., "pgam_res/ml_runs"
    resume=True
):
    """
    Tuned for ~15k x 20 regression.
    - Robust RMSE (old/new sklearn), robust CV scoring.
    - Handles RANSAC(estimator/base_estimator) across versions.
    - Prints progress; checkpoints each model (CSV + estimator .joblib); resume support.
    """

    # -------- helpers: RMSE + safe GridSearchCV scoring ----------
    def _rmse(y_true, y_pred):
        # Works on any sklearn version
        try:
            return mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    def _fit_grid(base_model, grid, cv, X, y, prefer='neg_root_mean_squared_error'):
        """
        Try preferred RMSE scoring; if unsupported, fall back to neg_mean_squared_error.
        Returns (best_estimator, best_score_as_RMSE, best_params, used_scoring)
        """
        used_scoring = prefer
        try:
            gs = GridSearchCV(base_model, grid, cv=cv, scoring=prefer, n_jobs=-1)
            gs.fit(X, y)
            best_rmse = -gs.best_score_  # already RMSE
        except Exception:
            used_scoring = 'neg_mean_squared_error'
            gs = GridSearchCV(base_model, grid, cv=cv, scoring=used_scoring, n_jobs=-1)
            gs.fit(X, y)
            best_rmse = (-gs.best_score_) ** 0.5  # convert MSE -> RMSE
        return gs.best_estimator_, float(best_rmse), gs.best_params_, used_scoring

    # -------- models tuned for 15k x 20 ----------
    models = {
        'linreg': LinearRegression(),
        'ridge':  make_pipeline(StandardScaler(with_mean=False), Ridge(random_state=random_state)),
        'lasso':  make_pipeline(StandardScaler(with_mean=False), Lasso(random_state=random_state, max_iter=20000)),
        'elastic':make_pipeline(StandardScaler(with_mean=False), ElasticNet(random_state=random_state, max_iter=20000)),

        'linear_svr': make_pipeline(StandardScaler(), LinearSVR(random_state=random_state, max_iter=20000)),
        'svr':        make_pipeline(StandardScaler(), SVR()),

        'knn':        make_pipeline(StandardScaler(), KNeighborsRegressor()),

        'dt':     DecisionTreeRegressor(random_state=random_state, min_samples_leaf=10, max_depth=12),
        'rf':     RandomForestRegressor(
                    n_estimators=300, max_depth=None, min_samples_leaf=5,
                    max_features='sqrt', n_jobs=-1, random_state=random_state),
        'extra_trees': ExtraTreesRegressor(
                    n_estimators=400, max_depth=None, min_samples_leaf=5,
                    max_features='sqrt', n_jobs=-1, random_state=random_state),

        'grad_boosting': GradientBoostingRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=3, random_state=random_state),
        'hist_gb': HistGradientBoostingRegressor(
                    learning_rate=0.06, max_depth=None, max_leaf_nodes=31,
                    min_samples_leaf=20, random_state=random_state),
        'boosting': AdaBoostRegressor(n_estimators=300, learning_rate=0.05, random_state=random_state),
        'bagging': BaggingRegressor(
                    n_estimators=200, max_samples=0.8, bootstrap=True, n_jobs=-1, random_state=random_state),

        'huber':  make_pipeline(StandardScaler(), HuberRegressor(max_iter=4000)),
        # RANSAC API compatibility
        'ransac': (RANSACRegressor(estimator=LinearRegression(), random_state=random_state)
                   if 'estimator' in inspect.signature(RANSACRegressor).parameters
                   else RANSACRegressor(base_estimator=LinearRegression(), random_state=random_state)),
        'sgd':    make_pipeline(StandardScaler(), SGDRegressor(random_state=random_state, max_iter=20000)),
        'pls':    make_pipeline(StandardScaler(with_mean=False), PLSRegression()),

        # ---- MLP (small, regularized, early stopping) ----
        'mlp':    make_pipeline(
                    StandardScaler(),
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        activation='relu',
                        alpha=1e-3,                 # L2
                        learning_rate='adaptive',
                        learning_rate_init=1e-3,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=15,
                        max_iter=500,
                        random_state=random_state,
                        verbose=False
                    )
                  )
    }

    # -------- compact CV grids (kept small for ~15k rows) ----------
    param_grids = {
        'ridge':        {'ridge__alpha': [0.3, 1, 3, 10, 30]},
        'lasso':        {'lasso__alpha': [0.001, 0.01, 0.1, 0.5]},
        'elastic':      {'elasticnet__alpha': [0.001, 0.01, 0.1],
                         'elasticnet__l1_ratio': [0.2, 0.5, 0.8]},
        'linear_svr':   {'linearsvr__C': [0.5, 1, 3]},
        'svr':          {'svr__C': [1, 3], 'svr__epsilon': [0.05, 0.1],
                         'svr__kernel': ['rbf'], 'svr__gamma': ['scale']},
        'knn':          {'kneighborsregressor__n_neighbors': [5, 10, 25]},
        'dt':           {'max_depth': [8, 12, 16], 'min_samples_leaf': [5, 10]},
        'rf':           {'n_estimators': [300, 500], 'max_depth': [None, 16],
                         'min_samples_leaf': [2, 5]},
        'extra_trees':  {'n_estimators': [400, 700], 'max_depth': [None, 16],
                         'min_samples_leaf': [2, 5]},
        'grad_boosting':{'n_estimators': [200, 300], 'learning_rate': [0.05, 0.1],
                         'max_depth': [2, 3]},
        'hist_gb':      {'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [31, 63],
                         'min_samples_leaf': [20, 50]},
        'huber':        {'huberregressor__epsilon': [1.2, 1.5, 2.0],
                         'huberregressor__alpha': [1e-4, 1e-3, 1e-2]},
        'sgd':          {'sgdregressor__alpha': [1e-5, 1e-4, 1e-3],
                         'sgdregressor__penalty': ['l2', 'l1', 'elasticnet']},
        'pls':          {'plsregression__n_components': [2, 4, 8, 12]},
        'mlp':          {'mlpregressor__hidden_layer_sizes': [(128,64), (64,64), (64,32)],
                         'mlpregressor__alpha': [1e-4, 1e-3, 1e-2],
                         'mlpregressor__learning_rate_init': [5e-4, 1e-3, 3e-3]},
    }

    # -------- checkpointing setup ----------
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        results_csv = os.path.join(save_dir, "model_results.csv")
        meta_json    = os.path.join(save_dir, "meta.json")
        done_set = set()
        if resume and os.path.exists(results_csv):
            try:
                prev = pd.read_csv(results_csv)
                done_set = set(prev['model'].astype(str).tolist())
                if verbose:
                    print(f"[resume] Found {len(done_set)} completed models; will skip them.")
            except Exception:
                pass
        if not os.path.exists(meta_json):
            with open(meta_json, "w") as f:
                json.dump({
                    "created_at": datetime.now().isoformat(timespec='seconds'),
                    "sklearn_version": sklearn.__version__,
                    "cv_splits": cv_splits, "use_cv": use_cv, "random_state": random_state
                }, f, indent=2)

    # -------- training loop ----------
    y_train_1d = np.ravel(y_train)
    y_test_1d  = np.ravel(y_test)

    rows = []
    best_model = None
    best_name = None
    best_rmse = np.inf

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    if verbose:
        print(f"Fitting {len(model_names)} models on 15,000x20-ish data (use_cv={use_cv})")

    for i, name in enumerate(model_names, start=1):
        if name not in models:
            if verbose: print(f"[{i}/{len(model_names)}] SKIP: unknown model '{name}'")
            continue
        if save_dir and resume and name in done_set:
            if verbose: print(f"[{i}/{len(model_names)}] SKIP (resume): {name}")
            continue

        t0 = time.time()
        base_model = models[name]
        if verbose: print(f"\n[{i}/{len(model_names)}] Training: {name}")

        try:
            if use_cv and name in param_grids:
                grid = param_grids[name]
                if verbose:
                    total_candidates = int(np.prod([len(v) for v in grid.values()])) if len(grid) else 0
                    print(f"  - GridSearchCV: {total_candidates} candidates")
                model, cv_rmse, cv_params, used_scoring = _fit_grid(
                    base_model, grid, cv, X_train, y_train_1d
                )
                if verbose:
                    print(f"  - Best CV RMSE: {cv_rmse:.4f}")
                    print(f"  - Params: {cv_params}")
            else:
                model = base_model
                model.fit(X_train, y_train_1d)
                cv_rmse = None
                cv_params = None
                if verbose: print("  - Fit done (no CV)")

            y_pred = model.predict(X_test)
            rmse = _rmse(y_test_1d, y_pred)
            r2   = r2_score(y_test_1d, y_pred)
            dt   = time.time() - t0

            row = {
                'model': name,
                'rmse': float(rmse),
                'r2': float(r2),
                'cv_rmse': float(cv_rmse) if cv_rmse is not None else None,
                'time_sec': float(dt),
                'best_params': json.dumps(cv_params) if cv_params else None
            }
            rows.append(row)

            if verbose:
                print(f"  - Test RMSE: {rmse:.4f} | R^2: {r2:.4f} | time: {dt:.2f}s")

            # checkpoint this model
            if save_dir:
                try:
                    if os.path.exists(results_csv):
                        existing = pd.read_csv(results_csv)
                        existing = existing[existing['model'] != name]
                        df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
                    else:
                        df = pd.DataFrame([row])
                    df.sort_values('rmse', ascending=True).reset_index(drop=True).to_csv(results_csv, index=False)
                except Exception as e:
                    if verbose: print(f"  ! Failed to write results CSV: {e}")

                # save estimator
                try:
                    est_path = os.path.join(save_dir, f"{name}.joblib")
                    joblib.dump(model, est_path)
                except Exception as e:
                    if verbose: print(f"  ! Failed to save estimator: {e}")

            # best tracker
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_name = name
                if save_dir:
                    try:
                        joblib.dump(best_model, os.path.join(save_dir, "best_so_far.joblib"))
                    except Exception as e:
                        if verbose: print(f"  ! Failed to update best_so_far: {e}")

        except Exception as e:
            if verbose: print(f"  ! ERROR in '{name}': {e}")
            row = {'model': name, 'rmse': np.inf, 'r2': -np.inf, 'cv_rmse': None,
                   'time_sec': time.time() - t0, 'best_params': None}
            rows.append(row)
            if save_dir:
                try:
                    if os.path.exists(results_csv):
                        existing = pd.read_csv(results_csv)
                        existing = existing[existing['model'] != name]
                        df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
                    else:
                        df = pd.DataFrame([row])
                    df.to_csv(results_csv, index=False)
                except Exception:
                    pass
            continue

    model_comparison_df = pd.DataFrame(rows).sort_values('rmse', ascending=True).reset_index(drop=True)
    chosen_model_info = {
        'model_name': best_name,
        'estimator': best_model,
        'test_rmse': float(best_rmse) if np.isfinite(best_rmse) else None,
        'test_r2': float(model_comparison_df.loc[0, 'r2']) if len(model_comparison_df) else None
    }

    if verbose:
        print("\n=== Final Results ===")
        if len(model_comparison_df):
            print(model_comparison_df[['model','rmse','r2','cv_rmse','time_sec']].to_string(
                index=False, float_format=lambda x: f"{x:.4f}"
            ))
            print(f"\nBest: {chosen_model_info['model_name']} | RMSE {chosen_model_info['test_rmse']:.4f} | R^2 {chosen_model_info['test_r2']:.4f}")
        else:
            print("No successful fits.")

    return model_comparison_df, chosen_model_info
