from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
# import HuberRegressor
from sklearn.linear_model import HuberRegressor

MODEL_SPECS = {

    # ------------------
    # Linear baselines
    # ------------------
    'ridge': {
        'model_class': Ridge,
        'model_kwargs': dict(
            alpha=1.0,
            fit_intercept=True,
        ),
    },

    'ridge_strong': {
        'model_class': Ridge,
        'model_kwargs': dict(
            alpha=10.0,
            fit_intercept=True,
        ),
    },

    # ------------------
    # Sparse models
    # ------------------
    'lasso': {
        'model_class': Lasso,
        'model_kwargs': dict(
            alpha=0.01,
            max_iter=5000,
        ),
    },

    'lasso_weak': {
        'model_class': Lasso,
        'model_kwargs': dict(
            alpha=0.005,
            max_iter=5000,
        ),
    },

    # ------------------
    # Elastic Net (best default sparse model)
    # ------------------
    'elastic_net': {
        'model_class': ElasticNet,
        'model_kwargs': dict(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=5000,
        ),
    },

    'elastic_net_l1': {
        'model_class': ElasticNet,
        'model_kwargs': dict(
            alpha=0.01,
            l1_ratio=0.8,
            max_iter=5000,
        ),
    },

    # ------------------
    # Robust regression (outlier-resistant)
    # ------------------
    'huber': {
        'model_class': HuberRegressor,
        'model_kwargs': dict(
            epsilon=1.35,
            alpha=0.0001,
            
        ),
    },

    # ------------------
    # Nonlinear baseline (strong but controlled)
    # ------------------
    'catboost_shallow': {
        'model_class': CatBoostRegressor,
        'model_kwargs': dict(
            loss_function='RMSE',
            iterations=300,
            depth=4,
            learning_rate=0.05,
            subsample=0.8,
            verbose=False,
            random_seed=0,
        ),
    },

    'catboost_medium': {
        'model_class': CatBoostRegressor,
        'model_kwargs': dict(
            loss_function='RMSE',
            iterations=500,
            depth=6,
            learning_rate=0.05,
            subsample=0.8,
            verbose=False,
            random_seed=0,
        ),
    },
}
