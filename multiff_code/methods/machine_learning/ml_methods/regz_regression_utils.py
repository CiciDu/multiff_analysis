# Standard library imports
import os

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

# Scientific computing
from scipy import stats

# Machine learning - sklearn
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
    max_error
)
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
)

# Statistics
import statsmodels.api as sm

# Deep learning

# Custom imports

# Set matplotlib parameters
plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def regularized_regression(X_train, y_train, X_test, y_test, method='ridge', alpha=0.01, verbose=False, show_plots=False):
    """
    Example of how to use regularized regression instead of statsmodels OLS.

    Parameters:
    - method: 'ridge', 'lasso', 'elastic_net', or 'cv' (for cross-validated)
    - alpha: regularization strength (higher = more regularization)
    - verbose: print metrics if True
    - show_plots: show diagnostic show_plots if True
    """

    if method == 'ridge':
        model = Ridge(alpha=alpha)
    elif method == 'lasso':
        model = Lasso(alpha=alpha)
    elif method == 'elastic_net':
        model = ElasticNet(alpha=alpha, l1_ratio=0.5)
    elif method == 'cv':
        # Use cross-validation to find optimal alpha
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    else:
        raise ValueError(
            "method must be 'ridge', 'lasso', 'elastic_net', or 'cv'")

    # Fit model
    model.fit(X_train, y_train)

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Get coefficients
    coefficients = model.coef_
    intercept = model.intercept_

    # Calculate comprehensive metrics
    train_metrics = regression_metrics_report(
        y_train, y_pred_train, verbose=verbose, show_plots=show_plots, model_name=f"{method} (train)")
    test_metrics = regression_metrics_report(
        y_test, y_pred_test, verbose=verbose, show_plots=show_plots, model_name=f"{method} (test)")

    results = {
        'model': model,
        'coefficients': coefficients,
        'intercept': intercept,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
    }

    # Build a one-row DataFrame for this model
    metrics_to_include = [
        'r2', 'pearson_corr', 'explained_variance'  # , 'rmse', 'mae', 'mape'
    ]
    row = {'Model': method}
    for split, metrics in [('train', train_metrics), ('test', test_metrics)]:
        for metric in metrics_to_include:
            col = f"{split.capitalize()} {metric.replace('_', ' ').title()}"
            val = metrics.get(metric, None)
            if isinstance(val, float):
                row[col] = f"{val:.4f}"
            else:
                row[col] = val
    results_df = pd.DataFrame([row])

    return results, results_df, y_pred_train, y_pred_test


def compare_regularized_models(X_train, y_train, X_test, y_test, verbose=False, show_plots=False):
    """
    Compare different regularization methods with comprehensive metrics (train and test).
    """
    methods = {
        'OLS (no regularization)': None,
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01),
        'Elastic Net': ElasticNet(alpha=0.01, l1_ratio=0.5),
        'Ridge CV': RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]),
        'Lasso CV': LassoCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0])
    }
    results = {}
    for name, model in methods.items():
        if name == 'OLS (no regularization)':
            X_train_const = sm.add_constant(X_train)
            X_test_const = sm.add_constant(X_test)
            ols_model = sm.OLS(y_train, X_train_const).fit()
            y_pred_train = ols_model.predict(X_train_const)
            y_pred_test = ols_model.predict(X_test_const)
            train_metrics = regression_metrics_report(
                y_train, y_pred_train, verbose=verbose, show_plots=show_plots, model_name=f"{name} (train)")
            test_metrics = regression_metrics_report(
                y_test, y_pred_test, verbose=verbose, show_plots=show_plots, model_name=f"{name} (test)")
            coefficients = ols_model.params[1:]
            intercept = ols_model.params[0]
        else:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            train_metrics = regression_metrics_report(
                y_train, y_pred_train, verbose=verbose, show_plots=show_plots, model_name=f"{name} (train)")
            test_metrics = regression_metrics_report(
                y_test, y_pred_test, verbose=verbose, show_plots=show_plots, model_name=f"{name} (test)")
            coefficients = model.coef_
            intercept = model.intercept_
        results[name] = {
            'model': model if name != 'OLS (no regularization)' else ols_model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'coefficients': coefficients,
            'intercept': intercept,
            'train_predictions': y_pred_train,
            'test_predictions': y_pred_test,
        }

        results_df = model_comparison_dataframe(results)

    return results, results_df


def regression_metrics_report(y_true, y_pred, verbose=True, show_plots=False, model_name=None):
    """
    Calculate and optionally print and plot comprehensive regression metrics.
    Returns a metrics dictionary.
    """
    # Metrics
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    explained_variance = explained_variance_score(y_true, y_pred)
    max_error_val = max_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) /
                   np.where(y_true != 0, y_true, 1))) * 100
    pearson_corr = np.corrcoef(y_true, y_pred)[0, 1]
    median_ae = np.median(np.abs(y_true - y_pred))
    if np.all(y_true > 0) and np.all(y_pred > 0):
        msle = np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)
    else:
        msle = np.nan
    metrics = {
        'r2': r2,
        'pearson_corr': pearson_corr,
        'explained_variance': explained_variance,
        # 'mse': mse,
        # 'rmse': rmse,
        # 'mae': mae,
        # 'mape': mape,
        # 'max_error': max_error_val,
        # 'median_ae': median_ae,
        # 'msle': msle
    }
    if verbose:
        print(f"{'='*60}")
        if model_name:
            print(f"REGRESSION REPORT: {model_name}")
        print(f"R²: {r2:.4f}")
        print(f"Explained Variance: {explained_variance:.4f}")
        print(f"Pearson Correlation on Test Data: {pearson_corr:.4f}")
        # print(f"MSE: {mse:.4f}")
        # print(f"RMSE: {rmse:.4f}")
        # print(f"MAE: {mae:.4f}")
        # print(f"MAPE: {mape:.2f}%")
        # print(f"Max Error: {max_error_val:.4f}")
        # print(f"Median Absolute Error: {median_ae:.4f}")
        # if not np.isnan(msle):
        #     print(f"Mean Squared Log Error: {msle:.4f}")
        # Residual analysis
        # residuals = y_true - y_pred
        # print(f"Residual Mean: {np.mean(residuals):.4f}")
        # print(f"Residual Std: {np.std(residuals):.4f}")
        # print(f"Residual Skewness: {stats.skew(residuals):.4f}")
        # print(f"Residual Kurtosis: {stats.kurtosis(residuals):.4f}")
    if show_plots:
        create_regression_diagnostic_plots(
            y_true, y_pred, y_true - y_pred, model_name or "Model")
    return metrics


def create_regression_diagnostic_plots(y_true, y_pred, residuals, model_name):
    """
    Create comprehensive diagnostic plots for regression analysis.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Regression Diagnostics: {model_name}', fontsize=16)
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
    axes[0, 0].plot([y_true.min(), y_true.max()], [
                    y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    # 2. Residuals vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    # 3. Residuals histogram
    axes[0, 2].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Residuals')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Residuals Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    # 4. Q-Q plot for residuals
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Residuals)')
    axes[1, 0].grid(True, alpha=0.3)
    # 5. Residuals vs Index
    axes[1, 1].plot(residuals, alpha=0.6)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Index')
    axes[1, 1].grid(True, alpha=0.3)
    # 6. Prediction error distribution
    prediction_errors = np.abs(y_true - y_pred)
    axes[1, 2].hist(prediction_errors, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 2].set_xlabel('Absolute Prediction Error')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Prediction Error Distribution')
    axes[1, 2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_model_comparison_summary(results, show_train=True, show_test=True):
    """
    Print a formatted summary of model comparison results for train and test.
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    summary_data = []
    for name, result in results.items():
        row = {'Model': name}
        if show_train:
            tm = result['train_metrics']
            row.update({
                'Train R²': f"{tm['r2']:.4f}",
                # 'Train RMSE': f"{tm['rmse']:.4f}",
                # 'Train MAE': f"{tm['mae']:.4f}",
                # 'Train MAPE (%)': f"{tm['mape']:.2f}",
                'Train Pearson R': f"{tm['pearson_corr']:.4f}",
                'Train EV': f"{tm['explained_variance']:.4f}"
            })
        if show_test:
            tm = result['test_metrics']
            row.update({
                'Test R²': f"{tm['r2']:.4f}",
                # 'Test RMSE': f"{tm['rmse']:.4f}",
                # 'Test MAE': f"{tm['mae']:.4f}",
                # 'Test MAPE (%)': f"{tm['mape']:.2f}",
                'Test Pearson R': f"{tm['pearson_corr']:.4f}",
                'Test EV': f"{tm['explained_variance']:.4f}"
            })
        summary_data.append(row)
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    # Find best model by test R²
    if show_test:
        best_model = max(
            results.items(), key=lambda x: x[1]['test_metrics']['r2'])
        print(
            f"\nBest model by Test R²: {best_model[0]} (R² = {best_model[1]['test_metrics']['r2']:.4f})")
    elif show_train:
        best_model = max(
            results.items(), key=lambda x: x[1]['train_metrics']['r2'])
        print(
            f"\nBest model by Train R²: {best_model[0]} (R² = {best_model[1]['train_metrics']['r2']:.4f})")
    return summary_df


def model_comparison_dataframe(results, metrics_to_include=None):
    """
    Create a DataFrame summarizing key metrics for each model (train and test).

    Parameters:
        results: dict
            Output from compare_regularized_models.
        metrics_to_include: list of str or None
            Which metrics to include (default: common ones).
    Returns:
        pd.DataFrame
    """
    if metrics_to_include is None:
        metrics_to_include = [
            'r2', 'pearson_corr', 'explained_variance', 'rmse', 'mae', 'mape'
        ]
    rows = []
    for model_name, res in results.items():
        row = {'Model': model_name}
        for split in ['train', 'test']:
            metrics = res[f'{split}_metrics']
            for metric in metrics_to_include:
                col = f"{split.capitalize()} {metric.replace('_', ' ').title()}"
                val = metrics.get(metric, None)
                if isinstance(val, float):
                    row[col] = f"{val:.4f}"
                else:
                    row[col] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    return df
