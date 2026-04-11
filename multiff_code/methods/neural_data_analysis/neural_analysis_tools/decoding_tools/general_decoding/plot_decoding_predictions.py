import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler

from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import (
    cv_decoding,
)


class ConfigRegressionEstimator:
    """Sklearn-style regressor built from :class:`cv_decoding.DecodingRunConfig`.

    Intended as ``runner_class`` for :func:`get_cv_predictions` (``fit`` / ``predict`` on neural ``X``).
    """

    def __init__(self, config=None, model_name=None):
        self.config = config if config is not None else cv_decoding.DecodingRunConfig()
        self.model_name = model_name
        self._scaler = None
        self._est = None

    def fit(self, X, y):
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(np.asarray(X, dtype=float))
        model_class = self.config.regression_model_class or CatBoostRegressor
        kw = self.config.regression_model_kwargs or dict(verbose=False)
        self._est = model_class(**kw)
        self._est.fit(Xs, np.asarray(y, dtype=float).ravel())
        return self

    def predict(self, X):
        return self._est.predict(self._scaler.transform(np.asarray(X, dtype=float)))


def get_cv_predictions(
    X,
    y,
    groups,
    runner_class,
    config,
    n_splits=5,
    cv_mode="kfold",
    model_name=None,
):
    if config is None:
        config = cv_decoding.DecodingRunConfig()
    if groups is None:
        groups = np.zeros(len(X), dtype=int)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    groups = np.asarray(groups)

    buffer_samples = getattr(config, "buffer_samples", 20)
    splits = cv_decoding._build_folds(
        len(X),
        n_splits=n_splits,
        groups=groups,
        cv_splitter=cv_mode,
        buffer_samples=buffer_samples,
        random_state=0,
    )

    y_true_all = []
    y_pred_all = []
    fold_ids = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        model = runner_class(config=config, model_name=model_name)

        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])

        y_true_all.append(y[test_idx])
        y_pred_all.append(y_pred)
        fold_ids.append(np.full(len(test_idx), fold_idx))

    return (
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
        np.concatenate(fold_ids),
    )

def plot_true_vs_pred(y_true, y_pred, fold_ids, pct=None):
    plt.figure()

    scatter = plt.scatter(
        y_true,
        y_pred,
        c=fold_ids,
        alpha=0.3,
        s=2,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label('Fold')

    # axis limits
    if pct is not None:
        low = min(
            np.percentile(y_true, pct[0]),
            np.percentile(y_pred, pct[0]),
        )
        high = max(
            np.percentile(y_true, pct[1]),
            np.percentile(y_pred, pct[1]),
        )
    else:
        low = min(y_true.min(), y_pred.min())
        high = max(y_true.max(), y_pred.max())

    plt.xlim(low, high)
    plt.ylim(low, high)

    # identity line
    plt.plot([low, high], [low, high], linestyle='--')

    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('True vs Predicted (colored by fold)')
    plt.show()