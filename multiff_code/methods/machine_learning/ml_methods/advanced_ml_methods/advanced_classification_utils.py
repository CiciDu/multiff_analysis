import os, json, time, joblib, traceback
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.base import clone

# --- Your imports (keep as-is) ---
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Optional libs (guarded)
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


# ------------- Helpers -------------
def _wrap_with_scaler_if_beneficial(name: str, est):
    """
    Wrap in a Pipeline with fixed step names so hyperparam grids are stable:
      ('scaler', StandardScaler()) + ('model', est)
    """
    needs_scaling = name in {"logreg", "ridge", "sgd", "svm", "knn", "mlp"}
    if needs_scaling:
        return Pipeline([("scaler", StandardScaler()), ("model", est)])
    else:
        return Pipeline([("model", est)])


def _proba_or_score(mdl, X, n_classes: int):
    """Return scores suitable for roc_auc_score (binary or multiclass)."""
    # If wrapped in Pipeline, fetch final estimator if needed
    try:
        final_est = mdl.named_steps.get("model", mdl)
    except Exception:
        final_est = mdl

    if hasattr(final_est, "predict_proba"):
        proba = mdl.predict_proba(X)
        if proba is None:
            return None
        if n_classes == 2:
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            if proba.ndim == 1:
                return proba
            return None
        else:
            if proba.ndim == 2 and proba.shape[1] == n_classes:
                return proba
            return None

    if hasattr(final_est, "decision_function"):
        dec = mdl.decision_function(X)
        if dec is None:
            return None
        if n_classes == 2:
            if dec.ndim == 1:
                return dec
            if dec.ndim == 2 and dec.shape[1] == 1:
                return dec.ravel()
            if dec.ndim == 2 and dec.shape[1] == 2:
                return dec[:, 1]
            return None
        else:
            if dec.ndim == 2 and dec.shape[1] == n_classes:
                return dec
            return None
    return None


def _param_spaces(name, n_classes):
    """
    Param grids/distributions keyed against Pipeline step 'model'.
    Lists (of dicts) work with RandomizedSearchCV and allow conditional spaces.
    """
    spaces = {
        # ---- Logistic Regression: valid solver–penalty combos only; no string 'none' ----
        "logreg": [
            # L2 with common solvers
            {
                "model__solver": ["lbfgs", "liblinear", "saga"],
                "model__penalty": ["l2"],
                "model__C": np.logspace(-3, 2, 10),
                "model__max_iter": [600, 1000],
                "model__tol": [1e-4, 3e-4],
            },
            # L1 requires liblinear or saga
            {
                "model__solver": ["liblinear", "saga"],
                "model__penalty": ["l1"],
                "model__C": np.logspace(-3, 2, 10),
                "model__max_iter": [800, 1200],
                "model__tol": [1e-4, 3e-4],
            },
            # ElasticNet requires saga (+ l1_ratio)
            {
                "model__solver": ["saga"],
                "model__penalty": ["elasticnet"],
                "model__l1_ratio": [0.15, 0.5, 0.8],
                "model__C": np.logspace(-3, 2, 10),
                "model__max_iter": [1200],
                "model__tol": [1e-4, 3e-4],
            },
        ],

        "ridge": {
            "model__alpha": np.logspace(-3, 3, 13),
        },

        # ---- SGD: only set eta0 for learning rates that use it; keep optimal clean ----
        "sgd": [
            # 'optimal' ignores eta0; safer combos here
            {
                "model__loss": ["log_loss", "hinge"],
                "model__penalty": ["l2"],
                "model__learning_rate": ["optimal"],
                "model__alpha": [1e-4, 1e-3, 1e-2],
                "model__max_iter": [2000],
                "model__early_stopping": [True],
                "model__n_iter_no_change": [5, 10],
            },
            # constant/invscaling/adaptive require eta0 > 0
            {
                "model__loss": ["log_loss", "hinge"],
                "model__penalty": ["l2"],
                "model__learning_rate": ["constant", "invscaling", "adaptive"],
                "model__eta0": [0.001, 0.01, 0.1],
                "model__alpha": [1e-4, 1e-3],
                "model__max_iter": [2000],
                "model__early_stopping": [True],
                "model__n_iter_no_change": [5, 10],
            },
        ],

        "dt": {
            "model__max_depth": [None, 3, 5, 7, 9, 12],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
        },

        "bagging": {
            "model__n_estimators": [100, 200, 300],
            "model__max_features": [0.6, 0.8, 1.0],
            "model__bootstrap": [True, False],
        },

        "rf": {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 10, 20, 30],
            "model__max_features": ["sqrt", "log2", None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },

        "extra_trees": {
            "model__n_estimators": [300, 500, 800],
            "model__max_depth": [None, 10, 20, 30],
            "model__max_features": ["sqrt", "log2", None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },

        "boosting": {
            "model__n_estimators": [200, 400, 800],
            "model__learning_rate": [0.01, 0.05, 0.1],
        },

        "grad_boosting": {
            "model__n_estimators": [200, 400, 800],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5, 7],
            "model__subsample": [0.5, 0.8, 1.0],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },

        "knn": {
            "model__n_neighbors": list(range(3, 21, 2)),
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },

        "mlp": {
            "model__hidden_layer_sizes": [(64,), (128,), (64, 64), (128, 64)],
            "model__alpha": np.logspace(-6, -2, 9),
            "model__activation": ["relu", "tanh"],
            "model__learning_rate_init": np.logspace(-4, -2, 7),
            "model__max_iter": [500, 800, 1200],
        },

        "svm": {
            "model__C": np.logspace(-2, 2, 9),
            "model__kernel": ["rbf", "linear"],
            "model__gamma": ["scale", "auto"],
            "model__class_weight": [None, "balanced"],
            "model__probability": [True],
        },

        "xgb": {} if XGBClassifier is None else {
            "model__n_estimators": [200, 400, 800],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_lambda": np.logspace(-3, 2, 10),
        },

        "lgbm": {} if LGBMClassifier is None else {
            "model__n_estimators": [300, 600, 1000],
            "model__num_leaves": [31, 63, 127],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_lambda": np.logspace(-3, 2, 10),
        },

        "catboost": {} if CatBoostClassifier is None else {
            "model__depth": [4, 6, 8],
            "model__iterations": [400, 800, 1200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__l2_leaf_reg": np.logspace(-3, 2, 10),
        },

        "gnb": {},  # usually not tuned
    }
    return spaces.get(name, {})


def _load_state(checkpoint_dir):
    metrics_path = os.path.join(checkpoint_dir, "metrics.json")
    state = {"metrics": {}, "completed": set()}
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r") as f:
            state["metrics"] = json.load(f)
        state["completed"] = set(state["metrics"].keys())
    return state


def _save_metric(checkpoint_dir, name, row_dict):
    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics_path = os.path.join(checkpoint_dir, "metrics.json")
    data = {}
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
    data[name] = row_dict
    with open(metrics_path, "w") as f:
        json.dump(data, f, indent=2)


# ------------- Main -------------
def use_advanced_model_for_classification(
    X_train, y_train, X_test, y_test,
    model_names=None,
    kfold_cv=None,
    random_state=42,
    verbose=True,
    # NEW:
    tune=True,
    n_iter=30,                 # good default for ~3k rows
    tune_scoring="balanced_accuracy",
    checkpoint_dir=None,       # e.g., "checkpoints_cls"
    resume=True,               # skip finished models if metrics exist
    n_jobs=-1,
):
    """
    Train/evaluate multiple classifiers with optional hyperparameter tuning
    and per-model checkpointing. Returns: (best_model, y_pred_best, model_comparison_df)
    """

    # Flatten labels if needed
    if isinstance(y_train, (pd.DataFrame, np.ndarray)) and getattr(y_train, "ndim", 1) > 1:
        y_train = np.asarray(y_train).ravel()
    if isinstance(y_test, (pd.DataFrame, np.ndarray)) and getattr(y_test, "ndim", 1) > 1:
        y_test = np.asarray(y_test).ravel()

    # Label encoding on UNION to keep mapping consistent
    le = LabelEncoder()
    le.fit(np.concatenate([np.asarray(y_train).ravel(), np.asarray(y_test).ravel()]))
    y_train_enc = le.transform(np.asarray(y_train).ravel())
    y_test_enc  = le.transform(np.asarray(y_test).ravel())
    classes = le.classes_
    n_classes = classes.size
    if n_classes < 2:
        raise ValueError("Need at least 2 classes for classification.")

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        joblib.dump(le, os.path.join(checkpoint_dir, "label_encoder.joblib"))

    # Base models (then wrapped)
    base_models = [
        ("gnb", GaussianNB()),
        ("logreg", LogisticRegression(max_iter=400, random_state=random_state)),  # slightly higher default
        ("ridge", RidgeClassifier(random_state=random_state)),
        ("sgd", SGDClassifier(max_iter=1000, tol=1e-3, random_state=random_state)),
        ("dt", DecisionTreeClassifier(random_state=random_state)),
        ("bagging", BaggingClassifier(
            n_estimators=200, max_features=0.9, bootstrap_features=True, bootstrap=True, random_state=random_state
        )),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=random_state)),
        ("extra_trees", ExtraTreesClassifier(n_estimators=500, random_state=random_state)),
        ("boosting", AdaBoostClassifier(n_estimators=500, learning_rate=0.05, random_state=random_state)),
        ("grad_boosting", GradientBoostingClassifier(
            learning_rate=0.05, max_depth=7, n_estimators=500, subsample=0.5,
            max_features="sqrt", min_samples_split=7, min_samples_leaf=2, random_state=random_state
        )),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=random_state)),
    ]
    if XGBClassifier is not None:
        base_models.append((
            "xgb",
            XGBClassifier(
                objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                num_class=n_classes if n_classes > 2 else None,
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
                random_state=random_state,
                tree_method="hist",
            )
        ))
    if LGBMClassifier is not None:
        base_models.append((
            "lgbm",
            LGBMClassifier(
                objective="multiclass" if n_classes > 2 else "binary",
                num_class=n_classes if n_classes > 2 else None,
                random_state=random_state
            )
        ))
    if CatBoostClassifier is not None:
        base_models.append((
            "catboost",
            CatBoostClassifier(
                loss_function="MultiClass" if n_classes > 2 else "Logloss",
                verbose=0, random_state=random_state
            )
        ))
    if len(X_train) < 10_000:
        base_models.append(("svm", SVC(probability=True, decision_function_shape="ovr", random_state=random_state)))

    # Wrap with scaler when beneficial; use fixed step name 'model' inside Pipeline
    all_models = [(n, _wrap_with_scaler_if_beneficial(n, est)) for n, est in base_models]

    # Subselect by names if provided
    if model_names is not None:
        name_set = set(model_names)
        all_models = [(n, m) for n, m in all_models if n in name_set]

    # CV setup (stratified)
    if kfold_cv:
        cv = StratifiedKFold(n_splits=kfold_cv, shuffle=True, random_state=random_state) \
             if isinstance(kfold_cv, int) else kfold_cv
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Load checkpoint state if resuming
    state = {"metrics": {}, "completed": set()}
    if checkpoint_dir and resume:
        state = _load_state(checkpoint_dir)

    metrics_list = []

    for name, base_mdl in all_models:
        try:
            if checkpoint_dir and resume and name in state["completed"]:
                if verbose: print(f"[resume] Skipping '{name}' (already completed).")
                row = state["metrics"][name]
                metrics_list.append(row)
                continue

            if verbose:
                print(f"\n=== Model: {name} ===")

            mdl = clone(base_mdl)

            # TUNING
            if tune:
                space = _param_spaces(name, n_classes)
                if isinstance(space, list):
                    has_space = len(space) > 0
                else:
                    has_space = len(space.keys()) > 0
                if has_space:
                    search = RandomizedSearchCV(
                        estimator=mdl,
                        param_distributions=space,
                        n_iter=n_iter,
                        scoring=tune_scoring,
                        cv=cv,
                        n_jobs=n_jobs,
                        random_state=random_state,
                        refit=True,
                        verbose=0
                    )
                    search.fit(X_train, y_train_enc)
                    mdl = search.best_estimator_
                    if verbose:
                        print(f"Best params for {name}: {search.best_params_}")
                        print(f"Best CV {tune_scoring}: {search.best_score_:.4f}")
                else:
                    mdl.fit(X_train, y_train_enc)
            else:
                mdl.fit(X_train, y_train_enc)

            # Save the fitted model immediately (checkpoint)
            if checkpoint_dir:
                joblib.dump(mdl, os.path.join(checkpoint_dir, f"{name}.joblib"))

            # Evaluate on test
            y_pred_enc = mdl.predict(X_test)

            roc_auc = None
            try:
                scores = _proba_or_score(mdl, X_test, n_classes)
                if scores is not None:
                    roc_auc = roc_auc_score(y_test_enc, scores, multi_class="ovr") if n_classes > 2 \
                              else roc_auc_score(y_test_enc, scores)
            except Exception:
                roc_auc = None

            row = {
                "model": name,
                "accuracy": float(accuracy_score(y_test_enc, y_pred_enc)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test_enc, y_pred_enc)),
                "precision_macro": float(precision_score(y_test_enc, y_pred_enc, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y_test_enc, y_pred_enc, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(y_test_enc, y_pred_enc, average="macro", zero_division=0)),
                "mcc": float(matthews_corrcoef(y_test_enc, y_pred_enc)),
                "roc_auc": None if roc_auc is None else float(roc_auc),
            }

            # Persist metrics (checkpoint)
            if checkpoint_dir:
                _save_metric(checkpoint_dir, name, row)

            metrics_list.append(row)

            if verbose:
                print(f"Test accuracy: {row['accuracy']:.4f} | balanced_acc: {row['balanced_accuracy']:.4f}")
        except Exception as e:
            if verbose:
                print(f"[warn] '{name}' failed: {e}")
                traceback.print_exc()
            # Record failure in metrics so resume won’t get stuck
            fail_row = {"model": name, "accuracy": -1, "balanced_accuracy": -1,
                        "precision_macro": -1, "recall_macro": -1, "f1_macro": -1, "mcc": -1, "roc_auc": None}
            if checkpoint_dir:
                _save_metric(checkpoint_dir, name, fail_row)
            metrics_list.append(fail_row)

    # Summary table
    model_comparison_df = pd.DataFrame(metrics_list).sort_values(
        by="accuracy", ascending=False
    ).reset_index(drop=True)

    if verbose:
        print("\nModel comparison:\n", model_comparison_df)

    # Choose best, then clone fresh and refit on train (to avoid stale internals)
    best_name = model_comparison_df.loc[0, "model"]
    if verbose:
        print(f"\nBest model by accuracy: {best_name}")

    # Load tuned model from checkpoint if available (avoids re-tuning)
    if checkpoint_dir and os.path.isfile(os.path.join(checkpoint_dir, f"{best_name}.joblib")):
        best_model = joblib.load(os.path.join(checkpoint_dir, f"{best_name}.joblib"))
    else:
        best_model = clone(dict(all_models)[best_name])
        if tune and len(_param_spaces(best_name, n_classes)) > 0:
            best_model.fit(X_train, y_train_enc)
        else:
            best_model.fit(X_train, y_train_enc)

    y_pred_best_enc = best_model.predict(X_test)
    y_pred_best = le.inverse_transform(y_pred_best_enc)

    if verbose:
        print("\nChosen model accuracy:", accuracy_score(y_test_enc, y_pred_best_enc))
        print("\nClassification Report (encoded labels):\n",
              classification_report(y_test_enc, y_pred_best_enc, zero_division=0, target_names=[str(c) for c in classes]))
        cm = confusion_matrix(y_test_enc, y_pred_best_enc, labels=np.arange(n_classes))
        cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in classes],
                                 columns=[f"Predicted {c}" for c in classes])
        print("\nConfusion Matrix (original class names):\n", cm_df)

    return best_model, y_pred_best, model_comparison_df
