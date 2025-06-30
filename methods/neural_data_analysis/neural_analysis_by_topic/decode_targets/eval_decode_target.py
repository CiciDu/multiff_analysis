import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compare_models(successful_decodings):
    # --- Usage Example ---
    model_comparison = extract_model_comparison(successful_decodings)
    comparison_df = build_comparison_dataframe(model_comparison)
    print_comparison_table(comparison_df)
    plot_model_comparison(comparison_df)
    return comparison_df

def extract_model_comparison(successful_decodings):
    """Extracts model performance metrics into a nested dictionary."""
    model_comparison = {}
    for target_var, models in successful_decodings.items():
        model_comparison[target_var] = {}
        for model_name, results in models.items():
            cv_score = results['cv_mean']
            cv_std = results['cv_std']
            if 'test_r2' in results:
                test_score = results['test_r2']
                score_type = 'R²'
            elif 'test_accuracy' in results:
                test_score = results['test_accuracy']
                score_type = 'Accuracy'
            else:
                test_score = cv_score
                score_type = 'CV Score'
            model_comparison[target_var][model_name] = {
                'cv_score': cv_score,
                'cv_std': cv_std,
                'test_score': test_score,
                'score_type': score_type
            }
    return model_comparison

def build_comparison_dataframe(model_comparison):
    """Builds a DataFrame for model comparison."""
    comparison_data = [
        {
            'Target Variable': target_var,
            'Model': model_name.upper(),
            'CV Score': f"{data['cv_score']:.4f} ± {data['cv_std']:.4f}",
            'Test Score': f"{data['test_score']:.4f}",
            'Score Type': data['score_type']
        }
        for target_var, models in model_comparison.items()
        for model_name, data in models.items()
    ]
    return pd.DataFrame(comparison_data)

def print_comparison_table(comparison_df):
    """Prints the model comparison table."""
    print("Model Performance Comparison:")
    print("="*60)
    print(comparison_df.to_string(index=False))

def plot_model_comparison(comparison_df):
    """Plots the model performance comparison."""
    if comparison_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    target_vars = comparison_df['Target Variable'].unique()
    models = comparison_df['Model'].unique()
    x = np.arange(len(target_vars))
    width = 0.8 / len(models)  # Adjust width for number of models

    for i, model in enumerate(models):
        model_scores = [
            float(comparison_df[
                (comparison_df['Target Variable'] == target_var) & 
                (comparison_df['Model'] == model)
            ]['Test Score'].iloc[0]) if not comparison_df[
                (comparison_df['Target Variable'] == target_var) & 
                (comparison_df['Model'] == model)
            ].empty else 0
            for target_var in target_vars
        ]
        ax.bar(x + i*width, model_scores, width, label=model)

    ax.set_xlabel('Target Variables')
    ax.set_ylabel('Test Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width*(len(models)-1)/2)
    ax.set_xticklabels(target_vars, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.show()

