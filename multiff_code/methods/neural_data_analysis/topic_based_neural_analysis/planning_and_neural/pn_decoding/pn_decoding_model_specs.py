from sklearn.linear_model import (
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
    HuberRegressor,
)
from catboost import CatBoostRegressor, CatBoostClassifier


MODEL_SPECS = {

    # ==================================================
    # Ridge (L2) — regression + classification
    # ==================================================

    'ridge': {
        'cache_tag': 'ridge_a1',

        'regression_model_class': Ridge,
        'regression_model_kwargs': dict(
            alpha=1.0,
            fit_intercept=True,
        ),

        'classification_model_class': LogisticRegression,
        'classification_model_kwargs': dict(
            penalty='l2',
            C=1.0,                     # C = 1 / alpha
            solver='lbfgs',
            max_iter=2000,
            class_weight='balanced',
        ),
    },

    'ridge_strong': {
        'cache_tag': 'ridge_a10',

        'regression_model_class': Ridge,
        'regression_model_kwargs': dict(
            alpha=10.0,
            fit_intercept=True,
        ),

        'classification_model_class': LogisticRegression,
        'classification_model_kwargs': dict(
            penalty='l2',
            C=0.1,                     # C = 1 / alpha
            solver='lbfgs',
            max_iter=2000,
            class_weight='balanced',
        ),
    },

    # ==================================================
    # Lasso (L1) — regression + classification
    # ==================================================

    'lasso': {
        'cache_tag': 'lasso_a001',

        'regression_model_class': Lasso,
        'regression_model_kwargs': dict(
            alpha=0.01,
            max_iter=5000,
        ),

        'classification_model_class': LogisticRegression,
        'classification_model_kwargs': dict(
            penalty='l1',
            C=100.0,                   # ~ 1 / alpha
            solver='saga',
            max_iter=5000,
            class_weight='balanced',
        ),
    },

    'lasso_weak': {
        'cache_tag': 'lasso_a0005',

        'regression_model_class': Lasso,
        'regression_model_kwargs': dict(
            alpha=0.005,
            max_iter=5000,
        ),

        'classification_model_class': LogisticRegression,
        'classification_model_kwargs': dict(
            penalty='l1',
            C=200.0,
            solver='saga',
            max_iter=5000,
            class_weight='balanced',
        ),
    },

    # ==================================================
    # Elastic Net — regression + classification
    # ==================================================

    'elastic_net': {
        'cache_tag': 'enet_a001_l05',

        'regression_model_class': ElasticNet,
        'regression_model_kwargs': dict(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=5000,
        ),

        'classification_model_class': LogisticRegression,
        'classification_model_kwargs': dict(
            penalty='elasticnet',
            C=100.0,
            l1_ratio=0.5,
            solver='saga',
            max_iter=5000,
            class_weight='balanced',
        ),
    },

    'elastic_net_l1': {
        'cache_tag': 'enet_a001_l08',

        'regression_model_class': ElasticNet,
        'regression_model_kwargs': dict(
            alpha=0.01,
            l1_ratio=0.8,
            max_iter=5000,
        ),

        'classification_model_class': LogisticRegression,
        'classification_model_kwargs': dict(
            penalty='elasticnet',
            C=100.0,
            l1_ratio=0.8,
            solver='saga',
            max_iter=5000,
            class_weight='balanced',
        ),
    },

    # ==================================================
    # Huber — regression only
    # ==================================================

    'huber': {
        'cache_tag': 'huber_eps135',

        'regression_model_class': HuberRegressor,
        'regression_model_kwargs': dict(
            epsilon=1.35,
            alpha=0.0001,
        ),

        'classification_model_class': None,
        'classification_model_kwargs': {},
    },

    # ==================================================
    # CatBoost — regression + classification
    # ==================================================

    'catboost_shallow': {
        'cache_tag': 'cb_d4_i300',

        'regression_model_class': CatBoostRegressor,
        'regression_model_kwargs': dict(
            loss_function='RMSE',
            iterations=300,
            depth=4,
            learning_rate=0.05,
            subsample=0.8,
            verbose=False,
            random_seed=0,
        ),

        'classification_model_class': CatBoostClassifier,
        'classification_model_kwargs': dict(
            loss_function='Logloss',
            iterations=300,
            depth=4,
            learning_rate=0.05,
            subsample=0.8,
            verbose=False,
            random_seed=0,
        ),
    },

    'catboost_medium': {
        'cache_tag': 'cb_d6_i500',

        'regression_model_class': CatBoostRegressor,
        'regression_model_kwargs': dict(
            loss_function='RMSE',
            iterations=500,
            depth=6,
            learning_rate=0.05,
            subsample=0.8,
            verbose=False,
            random_seed=0,
        ),

        'classification_model_class': CatBoostClassifier,
        'classification_model_kwargs': dict(
            loss_function='Logloss',
            iterations=500,
            depth=6,
            learning_rate=0.05,
            subsample=0.8,
            verbose=False,
            random_seed=0,
        ),
    },
}
