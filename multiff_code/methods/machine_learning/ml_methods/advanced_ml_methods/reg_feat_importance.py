import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import is_regressor

# ---------------------------
# Helpers for feature names
# ---------------------------
def _safe_get_feature_names(transformer, input_feature_names):
    # Try to ask the transformer for the names it outputs
    try:
        return transformer.get_feature_names_out(input_feature_names)
    except Exception:
        try:
            return transformer.get_feature_names_out()
        except Exception:
            # Fallback: keep input names length if possible
            return np.array(input_feature_names)

def _expand_column_transformer(ct: ColumnTransformer, raw_feature_names):
    names = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if trans == "drop":
            continue

        # Resolve column indices to names if needed
        if isinstance(cols, slice):
            cols = raw_feature_names[cols]
        elif isinstance(cols, (list, np.ndarray)):
            cols = [raw_feature_names[c] if isinstance(c, int) else c for c in cols]
        elif cols is None:
            cols = raw_feature_names

        if hasattr(trans, "transformers_"):  # nested ColumnTransformer
            names.extend(_expand_column_transformer(trans, np.array(cols)))
        else:
            out = _safe_get_feature_names(trans, np.array(cols))
            # Ensure 1D array of strings
            out = np.array([str(x) for x in np.ravel(out)])
            names.extend(out.tolist())
    return np.array(names)

def get_feature_names_from_pipeline(model, X):
    """
    Try to retrieve the post-preprocessing feature names for a Pipeline.
    If not a Pipeline or we can't resolve, fall back to X's columns (if DataFrame),
    else generate generic f0..f{n-1}.
    """
    if isinstance(X, pd.DataFrame):
        base_names = np.array(X.columns, dtype=str)
    else:
        base_names = np.array([f"f{i}" for i in range(X.shape[1])], dtype=str)

    if isinstance(model, Pipeline):
        # Find the last ColumnTransformer in the pipeline (if any)
        ct = None
        for _, step in model.steps:
            if isinstance(step, ColumnTransformer):
                ct = step
        if ct is not None:
            try:
                return _expand_column_transformer(ct, base_names)
            except Exception:
                pass

    # Try global get_feature_names_out on the whole pipeline
    try:
        return model.get_feature_names_out()
    except Exception:
        return base_names

# ---------------------------
# Core importance extractor
# ---------------------------
def get_regressor(import_maybe_pipeline):
    """Return the fitted final regressor (unwrapped from Pipeline if needed)."""
    model = import_maybe_pipeline
    if isinstance(model, Pipeline):
        model = model[-1]  # final step
    return model

def _tree_importances(est):
    try:
        imp = getattr(est, "feature_importances_", None)
        if imp is not None:
            return np.asarray(imp, dtype=float)
    except Exception:
        pass
    return None

def _linear_importances(est):
    # Use absolute coefficient magnitude
    try:
        coef = getattr(est, "coef_", None)
        if coef is not None:
            coef = np.asarray(coef, dtype=float)
            # For multi-output, sum magnitudes across targets
            if coef.ndim > 1:
                coef = np.sum(np.abs(coef), axis=0)
            else:
                coef = np.abs(coef)
            return coef
    except Exception:
        pass
    return None

def feature_importance_table(model, X, y=None, n_repeats=10, random_state=0):
    """
    Returns a DataFrame with columns ['feature', 'importance', 'method'] sorted descending.

    Priority:
      1) .feature_importances_ (trees/boosting)
      2) |coef_| (linear models)
      3) permutation importance (requires y)
    """
    if not hasattr(model, "fit"):
        raise ValueError("Model must be a fitted estimator or Pipeline.")

    # Feature names post-preprocessing if possible
    feat_names = get_feature_names_from_pipeline(model, X)

    # Unwrap final regressor
    est = get_regressor(model)

    # (1) Tree-based importances
    imp = _tree_importances(est)
    if imp is not None and len(imp) == len(feat_names):
        df = pd.DataFrame({"feature": feat_names, "importance": imp})
        df["method"] = "model_feature_importances_"
        return df.sort_values("importance", ascending=False, ignore_index=True)

    # (2) Linear coefficients
    imp = _linear_importances(est)
    if imp is not None and len(imp) == len(feat_names):
        df = pd.DataFrame({"feature": feat_names, "importance": imp})
        df["method"] = "abs(coef_)"
        return df.sort_values("importance", ascending=False, ignore_index=True)

    # (3) Permutation importance (model-agnostic)
    if y is None:
        raise ValueError(
            "This model doesn't expose importances. Provide y to compute permutation importance."
        )
    # Use the pipeline directly so preprocessing is included
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    imp = r.importances_mean
    df = pd.DataFrame({"feature": feat_names, "importance": imp})
    df["method"] = "permutation_importance(mean)"
    return df.sort_values("importance", ascending=False, ignore_index=True)

