import pandas as pd

def collect_global_summary(all_sessions_outs, reg_key):
    rows = []

    for session_id, outs_reg in all_sessions_outs.items():
        df = outs_reg[reg_key]['global_summary'].copy()

        # Average across CV folds within session
        df_sess = (
            df
            .groupby('model', as_index=False)
            .agg(
                r2=('r2', 'mean'),
                mse=('mse', 'mean')
            )
        )

        df_sess['session_id'] = session_id
        rows.append(df_sess)

    return pd.concat(rows, ignore_index=True)

def collect_cond_delta_summary(all_sessions_outs, reg_key):
    rows = []

    for session_id, outs_reg in all_sessions_outs.items():
        df = outs_reg[reg_key]['cond_delta_summary'].copy()
        df['session_id'] = session_id
        rows.append(df)

    return pd.concat(rows, ignore_index=True)

import matplotlib.pyplot as plt
import numpy as np

def plot_cond_delta_across_sessions(df_cond_group):
    fig, ax = plt.subplots(figsize=(6, 4))

    conditions = df_cond_group['condition_value'].unique()
    models = df_cond_group['model'].unique()

    x = np.arange(len(conditions))
    width = 0.35

    for i, model in enumerate(models):
        df_m = df_cond_group[df_cond_group['model'] == model]

        ax.bar(
            x + i * width,
            df_m['mean_effect'],
            yerr=df_m['sem_effect'],
            width=width,
            label=model,
            capsize=4
        )

    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(conditions)
    ax.set_ylabel('Δ score (mean across sessions)')
    ax.set_title('Condition effects across sessions')
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_session_scatter(df_cond_all):
    fig, ax = plt.subplots(figsize=(6, 4))

    for (cond, model), df_sub in df_cond_all.groupby(['condition_value', 'model']):
        ax.scatter(
            df_sub['session_id'],
            df_sub['mean_delta_score'],
            label=f'{cond} – {model}',
            alpha=0.7
        )

    ax.axhline(0, linestyle='--')
    ax.set_ylabel('Δ score')
    ax.set_title('Session-level condition effects')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
