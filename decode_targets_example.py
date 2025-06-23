#!/usr/bin/env python3
"""
Example: Decode Monkey's Target Representation from Neural Data

This script demonstrates how to use the multiff_analysis codebase to decode
a monkey's representation of targets from neural activity data.

The script performs:
1. Data loading and preprocessing
2. Canonical Correlation Analysis (CCA) 
3. Machine learning-based decoding
4. Visualization of results

Author: AI Assistant
Date: 2024
"""

from non_behavioral_analysis.neural_data_analysis.model_neural_data.cca_methods import cca_class
from non_behavioral_analysis.neural_data_analysis.decode_targets.decode_target_class import DecodeTargetClass
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Setup paths
project_folder = '/Users/dusiyi/Documents/Multifirefly-Project'
os.chdir(project_folder)
sys.path.append(os.path.join(project_folder, 'multiff_analysis', 'methods'))

# Import required classes


def main():
    """Main function to demonstrate target decoding."""

    print("="*60)
    print("MONKEY TARGET REPRESENTATION DECODING")
    print("="*60)
    print("This script demonstrates how to decode a monkey's internal")
    print("representation of targets from neural activity patterns.\n")

    # Step 1: Initialize the decoder
    print("Step 1: Initializing decoder...")
    raw_data_path = "all_monkey_data/raw_monkey_data/monkey_Bruno/data_0328"

    decoder = DecodeTargetClass(
        raw_data_folder_path=raw_data_path,
        bin_width=0.02,      # 20ms time bins
        window_width=0.05    # 50ms sliding window
    )
    print("✓ Decoder initialized")

    # Step 2: Load and prepare data
    print("\nStep 2: Loading neural and behavioral data...")
    try:
        # Load behavioral and neural data
        decoder.streamline_making_behav_and_neural_data(exists_ok=True)

        # Get X (neural) and Y (behavioral) variables with temporal lags
        decoder.get_x_and_y_var(exists_ok=True)

        # Prepare data
        neural_data = decoder.x_var_lags.drop(
            columns=['bin']) if 'bin' in decoder.x_var_lags.columns else decoder.x_var_lags
        target_data = decoder.y_var_lags_reduced

        # Handle missing values
        neural_data = neural_data.fillna(0)
        target_data = target_data.fillna(
            method='ffill').fillna(method='bfill').fillna(0)

        print(f"✓ Data loaded successfully")
        print(f"  Neural data shape: {neural_data.shape}")
        print(f"  Target data shape: {target_data.shape}")
        print(f"  Time points: {neural_data.shape[0]:,}")
        print(f"  Neurons: {neural_data.shape[1]}")
        print(
            f"  Recording duration: ~{neural_data.shape[0] * 0.02 / 60:.1f} minutes")

    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return

    # Step 3: Canonical Correlation Analysis
    print("\nStep 3: Performing Canonical Correlation Analysis...")
    try:
        # Initialize and run CCA
        cca = cca_class.CCAclass(
            X1=neural_data, X2=target_data, lagging_included=True)
        cca.conduct_cca(n_components=10, plot_correlations=False)

        print(f"✓ CCA completed")
        print(f"  Top 5 canonical correlations: {cca.canon_corr[:5]}")
        print(
            f"  Strongest neural-target correlation: {cca.canon_corr[0]:.3f}")
        print(
            f"  Variance explained by 1st component: {cca.canon_corr[0]**2:.1%}")

        # Plot CCA results
        plot_cca_results(cca)

    except Exception as e:
        print(f"✗ CCA failed: {e}")
        cca = None

    # Step 4: Machine Learning Decoding
    print("\nStep 4: Machine learning-based target decoding...")

    # Find target variables to decode
    target_vars = [col for col in target_data.columns
                   if any(keyword in col.lower() for keyword in
                          ['target_distance', 'target_angle', 'target_x', 'target_y'])]

    if len(target_vars) == 0:
        print("✗ No suitable target variables found")
        return

    print(f"  Available target variables: {len(target_vars)}")

    # Try to decode the first available target variable
    target_var = target_vars[0]
    print(f"  Attempting to decode: {target_var}")

    try:
        results = decode_target_with_ml(neural_data, target_data, target_var)
        if results:
            print(f"✓ Successfully decoded {target_var}")
            print(
                f"  Cross-validation R²: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
            print(f"  Test R²: {results['test_r2']:.3f}")

            # Plot results
            plot_ml_results(results, target_var)

            # Show feature importance
            plot_feature_importance(results, neural_data.columns, target_var)
        else:
            print(f"✗ Failed to decode {target_var}")

    except Exception as e:
        print(f"✗ ML decoding failed: {e}")

    print("\n" + "="*60)
    print("TARGET DECODING ANALYSIS COMPLETED")
    print("="*60)

    # Summary
    print("\nSUMMARY:")
    if cca is not None:
        print(f"• CCA found {len(cca.canon_corr)} canonical components")
        print(f"• Strongest correlation: {cca.canon_corr[0]:.3f}")
        print(f"• Components with r > 0.1: {sum(cca.canon_corr > 0.1)}")

    if 'results' in locals() and results:
        print(f"• ML decoding R²: {results['test_r2']:.3f}")
        if results['test_r2'] > 0.3:
            print("• Strong decoding performance achieved")
        elif results['test_r2'] > 0.1:
            print("• Moderate decoding performance achieved")
        else:
            print("• Limited decoding performance")

    print("\nInterpretation:")
    print("• Higher correlations indicate stronger neural representation")
    print("• R² > 0.3 suggests good predictive power")
    print("• Feature importance shows which neurons encode targets")


def decode_target_with_ml(neural_data, target_data, target_var):
    """Decode a target variable using Random Forest."""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error

    # Prepare data
    X = neural_data.values
    y = target_data[target_var].values

    # Remove invalid values
    finite_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X = X[finite_mask]
    y = y[finite_mask]

    if len(X) < 100:
        print(f"Insufficient valid data points: {len(X)}")
        return None

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=5, scoring='r2')

    # Train and evaluate
    model.fit(X_train_scaled, y_train)
    y_pred_test = model.predict(X_test_scaled)

    # Calculate metrics
    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)

    return {
        'model': model,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_r2': test_r2,
        'test_mse': test_mse,
        'predictions': y_pred_test,
        'true_values': y_test,
        'scaler': scaler
    }


def plot_cca_results(cca, max_components=5):
    """Plot CCA results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Canonical correlations
    axes[0, 0].bar(range(len(cca.canon_corr)), cca.canon_corr)
    axes[0, 0].set_title('Canonical Correlations')
    axes[0, 0].set_xlabel('Component')
    axes[0, 0].set_ylabel('Correlation')
    axes[0, 0].grid(True, alpha=0.3)

    # Neural loadings heatmap
    neural_loadings = cca.X1_loading[:, :max_components]
    im1 = axes[0, 1].imshow(neural_loadings.T, aspect='auto', cmap='RdBu_r')
    axes[0, 1].set_title(
        f'Neural Loadings (First {max_components} Components)')
    axes[0, 1].set_xlabel('Neurons')
    axes[0, 1].set_ylabel('Components')
    plt.colorbar(im1, ax=axes[0, 1])

    # Target loadings heatmap
    target_loadings = cca.X2_loading[:, :max_components]
    im2 = axes[1, 0].imshow(target_loadings.T, aspect='auto', cmap='RdBu_r')
    axes[1, 0].set_title(
        f'Target Loadings (First {max_components} Components)')
    axes[1, 0].set_xlabel('Target Features')
    axes[1, 0].set_ylabel('Components')
    plt.colorbar(im2, ax=axes[1, 0])

    # First canonical variables scatter
    axes[1, 1].scatter(cca.X1_c[:, 0], cca.X2_c[:, 0], alpha=0.5, s=1)
    axes[1, 1].set_title(
        f'First Canonical Variables (r={cca.canon_corr[0]:.3f})')
    axes[1, 1].set_xlabel('Neural CV1')
    axes[1, 1].set_ylabel('Target CV1')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cca_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  CCA results saved as 'cca_results.png'")


def plot_ml_results(results, target_var):
    """Plot ML decoding results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Predicted vs True
    axes[0].scatter(results['true_values'],
                    results['predictions'], alpha=0.6, s=10)
    min_val = min(results['true_values'].min(), results['predictions'].min())
    max_val = max(results['true_values'].max(), results['predictions'].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title(
        f'Predicted vs True: {target_var}\nR² = {results["test_r2"]:.3f}')
    axes[0].grid(True, alpha=0.3)

    # Residuals
    residuals = results['true_values'] - results['predictions']
    axes[1].scatter(results['predictions'], residuals, alpha=0.6, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residuals Plot')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'ml_results_{target_var}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ML results saved as 'ml_results_{target_var}.png'")


def plot_feature_importance(results, feature_names, target_var, top_n=15):
    """Plot feature importance from Random Forest."""
    if not hasattr(results['model'], 'feature_importances_'):
        print("  Feature importance not available")
        return

    importances = results['model'].feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Neural Features for Decoding {target_var}')
    plt.bar(range(top_n), importances[indices[:top_n]])
    plt.xticks(range(top_n), [feature_names[i]
               for i in indices[:top_n]], rotation=45)
    plt.xlabel('Neural Features (Neurons)')
    plt.ylabel('Feature Importance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f'feature_importance_{target_var}.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(
        f"  Feature importance plot saved as 'feature_importance_{target_var}.png'")
    print(f"  Top 5 most important neurons:")
    for i in range(min(5, len(indices))):
        neuron_idx = indices[i]
        importance = importances[neuron_idx]
        print(f"    {i+1}. {feature_names[neuron_idx]}: {importance:.4f}")


if __name__ == "__main__":
    main()
