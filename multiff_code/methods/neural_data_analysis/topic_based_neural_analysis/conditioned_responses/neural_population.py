import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier

def build_population_matrix(trial_rates, seg_ids, labels, neuron_ids, t_bins):
    T = len(t_bins)
    N = len(neuron_ids)
    trial_list, label_list = [], []
    for seg_id, lbl in zip(seg_ids, labels):
        trial = np.array([
            trial_rates[(seg_id, nid)]
            for nid in neuron_ids
            if (seg_id, nid) in trial_rates
        ])
        if trial.shape[0] == N:
            trial_list.append(trial)
            label_list.append(lbl)
        else:
            print(f"  Warning: seg {seg_id} missing neurons, skipping.")
    if len(trial_list) == 0:
        raise ValueError("No valid trials found in build_population_matrix.")
    X_3d = np.array(trial_list)
    y = np.array(label_list)
    return X_3d, y

def _make_clf(method="svm", n_features=None):
    if method == "svm":
        return make_pipeline(StandardScaler(),
                             LinearSVC(max_iter=2000, random_state=0, class_weight="balanced"))
    elif method == "lda":
        n_components = min(10, n_features - 1) if n_features else 10
        return make_pipeline(StandardScaler(),
                             PCA(n_components=n_components),
                             LinearDiscriminantAnalysis())
        
        
def _effective_n_splits(y, requested_splits):
    y = np.asarray(y)
    if y.size == 0:
        return None
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2:
        return None
    max_splits = int(np.min(counts))
    if max_splits < 2:
        return None
    return min(int(requested_splits), max_splits)


def decode_per_timepoint(X_3d, y, n_splits=5, method="svm"):
    """
    Decode bin label from population snapshot at each timepoint.
    Returns acc (T,) and chance level.
    """

    min_count = np.bincount(y).min()
    if min_count < n_splits:
        print(f"Warning: min bin count {min_count} < n_splits {n_splits}, reducing to {min_count}")
        n_splits = min_count

    T = X_3d.shape[2]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    chance_clf = DummyClassifier(strategy="most_frequent")
    acc = np.zeros(T)
    chance = np.zeros(T)

    for t in range(T):
        X_t = X_3d[:, :, t]   # (n_trials, N)
        fold_accs, fold_chance = [], []
        for train_idx, test_idx in cv.split(X_t, y):
            clf = _make_clf(method)
            clf.fit(X_t[train_idx], y[train_idx])
            acc_fold = clf.score(X_t[test_idx], y[test_idx])
            fold_accs.append(acc_fold)

            chance_clf.fit(X_t[train_idx], y[train_idx])
            fold_chance.append(chance_clf.score(X_t[test_idx], y[test_idx]))

        acc[t] = np.mean(fold_accs)
        chance[t] = np.mean(fold_chance)

    return acc, chance


def decode_full_trajectory(X_3d, y, n_splits=5, method="svm"):
    min_count = np.bincount(y).min()
    if min_count < n_splits:
        print(f"Warning: min bin count {min_count} < n_splits {n_splits}, reducing to {min_count}")
        n_splits = min_count

    n_trials, N, T = X_3d.shape
    X_flat = X_3d.reshape(n_trials, N * T)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    
    chance_clf = DummyClassifier(strategy="most_frequent")
    fold_accs, fold_chance = [], []

    for train_idx, test_idx in cv.split(X_flat, y):
        clf = _make_clf(method)
        clf.fit(X_flat[train_idx], y[train_idx])
        fold_accs.append(clf.score(X_flat[test_idx], y[test_idx]))

        chance_clf.fit(X_flat[train_idx], y[train_idx])
        fold_chance.append(chance_clf.score(X_flat[test_idx], y[test_idx]))

    return np.mean(fold_accs), np.mean(fold_chance)


def pca_population_trajectories(X_3d, y, t_bins, n_components=3):
    """
    PCA fit on all individual trials (n_trials*T, N), then mean trajectory
    per bin plotted in PC space. Returns X_proj (n_trials, T, n_components),
    y, explained variance ratio.
    """
    n_trials, N, T = X_3d.shape

    # fit PCA on all trials and timepoints
    X_flat = X_3d.transpose(0, 2, 1).reshape(-1, N)   # (n_trials*T, N)
    n_components_eff = min(int(n_components), X_flat.shape[0], X_flat.shape[1])
    if n_components_eff < 1:
        raise ValueError("Not enough data to fit PCA.")
    pca = PCA(n_components=n_components_eff)
    X_proj = pca.fit_transform(X_flat)                 # (n_trials*T, n_components)
    X_proj = X_proj.reshape(n_trials, T, n_components_eff) # (n_trials, T, n_components)

    return X_proj, pca.explained_variance_ratio_


def plot_population_analysis(psth, trial_rates, seg_ids, labels, t_bins,
                              n_splits=5, method="svm"):
    neuron_ids = sorted(set(k[1] for k in psth))
    unique_labels_arr = sorted(np.unique(labels))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(unique_labels_arr)))

    X_3d, y = build_population_matrix(trial_rates, seg_ids, labels, neuron_ids, t_bins)
    if X_3d.size == 0 or y.size == 0:
        raise ValueError("No complete trials available for population analysis.")

    print(f"Population matrix: {X_3d.shape} (trials x neurons x time)")
    print(f"Trials per bin: { {label: int((y == label).sum()) for label in unique_labels_arr} }")

    # compute all analyses
    acc_t, chance_t = decode_per_timepoint(X_3d, y, n_splits=n_splits, method=method)
    acc_full, chance_full = decode_full_trajectory(X_3d, y, n_splits=n_splits, method=method)
    X_proj, var_ratio = pca_population_trajectories(X_3d, y, t_bins)

    fig = plt.figure(figsize=(14, 12))
    gs = plt.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # --- top left: per-timepoint decoding ---
    ax0 = fig.add_subplot(gs[0, 0])
    print('acc_t', acc_t)
    ax0.plot(t_bins, acc_t, color="steelblue", lw=2, label="per-timepoint")
    ax0.fill_between(t_bins, chance_t, acc_t,
                     where=acc_t > chance_t, alpha=0.15, color="steelblue")
    ax0.plot(t_bins, chance_t, color="k", lw=1, ls="--", label="chance")
    ax0.axvline(0, color="k", lw=0.8, ls="--")
    ax0.set_xlabel("normalized time")
    ax0.set_ylabel("decoding accuracy")
    ax0.set_ylim(0, 1)
    ax0.set_title(f"per-timepoint decoding ({method.upper()})")
    ax0.legend(fontsize=8)

    # --- top right: full trajectory decoding (scalar, shown as hline) ---
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axhline(acc_full, color="steelblue", lw=2,
                label=f"full trajectory acc={acc_full:.2f}")
    ax1.axhline(chance_full, color="k", lw=1, ls="--",
                label=f"chance={chance_full:.2f}")
    ax1.set_xlim(t_bins[0], t_bins[-1])
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("normalized time")
    ax1.set_ylabel("decoding accuracy")
    ax1.set_title(f"full trajectory decoding ({method.upper()})")
    ax1.legend(fontsize=8)

    # --- bottom left: PC1 vs PC2 ---
    ax2 = fig.add_subplot(gs[1, 0])
    for lbl, color in zip(unique_labels_arr, cmap):
        mask = y == lbl
        mean_traj = X_proj[mask].mean(axis=0)   # (T, 3)
        if mean_traj.shape[1] < 2:
            continue
        ax2.plot(mean_traj[:, 0], mean_traj[:, 1], color=color, lw=1.5, label=f"bin {lbl}")
        ax2.scatter(mean_traj[0, 0], mean_traj[0, 1], color=color, s=40, zorder=5)
        ax2.scatter(mean_traj[-1, 0], mean_traj[-1, 1], color=color,
                    marker="x", s=40, zorder=5)
    ax2.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    if len(var_ratio) >= 2:
        ax2.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    else:
        ax2.set_ylabel("PC2")
    ax2.set_title("population trajectories PC1 vs PC2")
    ax2.legend(fontsize=7, ncol=2)

    # --- bottom right: PC1 vs PC3 ---
    ax3 = fig.add_subplot(gs[1, 1])
    if X_proj.shape[2] >= 3 and len(var_ratio) >= 3:
        for lbl, color in zip(unique_labels_arr, cmap):
            mask = y == lbl
            mean_traj = X_proj[mask].mean(axis=0)
            ax3.plot(mean_traj[:, 0], mean_traj[:, 2], color=color, lw=1.5, label=f"bin {lbl}")
            ax3.scatter(mean_traj[0, 0], mean_traj[0, 2], color=color, s=40, zorder=5)
            ax3.scatter(mean_traj[-1, 0], mean_traj[-1, 2], color=color,
                        marker="x", s=40, zorder=5)
        ax3.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
        ax3.set_ylabel(f"PC3 ({var_ratio[2]*100:.1f}%)")
        ax3.set_title("population trajectories PC1 vs PC3")
        ax3.legend(fontsize=7, ncol=2)
    else:
        ax3.text(0.5, 0.5, "PC3 unavailable\n(insufficient dimensionality)",
                 ha="center", va="center", transform=ax3.transAxes)
        ax3.set_axis_off()

    return fig