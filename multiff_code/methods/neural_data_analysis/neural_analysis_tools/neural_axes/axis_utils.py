from scipy.stats import mannwhitneyu
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

# -----------------------------------------------------------
# Continuous FR construction
# -----------------------------------------------------------
def build_continuous_fr(spike_times, spike_codes, n_neurons, bin_width_ms, smoothing_sigma_ms):
    bin_width_s = bin_width_ms / 1000.0

    start_t = float(spike_times.min())
    end_t = float(spike_times.max())
    total_dur = end_t - start_t

    n_bins = int(np.ceil(total_dur / bin_width_s))
    FR = np.zeros((n_bins, n_neurons), float)

    bin_idx = ((spike_times - start_t) / bin_width_s).astype(int)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)

    np.add.at(FR, (bin_idx[valid], spike_codes[valid]), 1)
    FR /= bin_width_s

    if smoothing_sigma_ms > 0:
        sigma_bins = smoothing_sigma_ms / bin_width_ms
        FR = gaussian_filter1d(FR, sigma=sigma_bins, axis=0, mode="nearest")

    return FR, start_t


# -----------------------------------------------------------
# Event → bin mapping
# -----------------------------------------------------------
def events_to_bins(event_times, start_time, bin_width_s):
    return np.floor((event_times - start_time) / bin_width_s).astype(int)


# -----------------------------------------------------------
# Extract event windows (vectorized) with safe range handling
# -----------------------------------------------------------
def extract_event_windows(fr_mat, event_bins, window_ms, bin_width_ms, require_full_window=True):
    """
    Vectorized extraction of peri-event windows from a continuous FR matrix.
    By default, drops events whose requested window extends outside the FR range.
    If require_full_window is False, indices are clipped to valid bounds.
    """
    start_ms, end_ms = window_ms
    so = int(start_ms / bin_width_ms)
    eo = int(end_ms / bin_width_ms)

    offsets = np.arange(so, eo)
    idx = event_bins[:, None] + offsets[None, :]

    T = fr_mat.shape[0]
    if require_full_window:
        # Keep only events whose window lies entirely within [0, T-1]
        valid = (idx.min(axis=1) >= 0) & (idx.max(axis=1) < T)
        if not np.any(valid):
            return np.zeros((0, len(offsets), fr_mat.shape[1]), dtype=fr_mat.dtype)
        idx = idx[valid]
        return fr_mat[idx]
    else:
        # Fallback: clip to bounds (may introduce edge bias)
        idx = np.clip(idx, 0, T - 1)
        return fr_mat[idx]


# -----------------------------------------------------------
# Fit behavior axis
# -----------------------------------------------------------
def fit_behavior_axis(X, y, model='logreg', **kwargs):
    """
    Returns:
        readout : fitted model
        axis_vec : np.ndarray or None
    """

    if model == 'lda':
        clf = LinearDiscriminantAnalysis(priors=None)
        clf.fit(X, y)
        axis = clf.coef_.ravel()
        axis /= np.linalg.norm(axis)
        return clf, axis

    elif model == 'logreg':
        clf = LogisticRegression(
            penalty='l2',
            solver='liblinear',
            max_iter=2000,
            class_weight="balanced"
        )
        clf.fit(X, y)
        axis = clf.coef_.ravel()
        axis /= np.linalg.norm(axis)
        return clf, axis

    elif model == 'poly_logreg':
        degree = kwargs.get('degree', 2)
        clf = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('logreg', LogisticRegression(
                penalty='l2',
                solver='liblinear',
                max_iter=2000,
                class_weight="balanced"
            ))
        ])
        clf.fit(X, y)
        return clf, None

    elif model == 'rbf_svm':
        clf = SVC(
            kernel='rbf',
            C=kwargs.get('C', 1.0),
            gamma=kwargs.get('gamma', 'scale'),
            probability=False,
            class_weight="balanced"
        )
        clf.fit(X, y)
        return clf, None

    elif model == 'mlp':
        sample_weight = compute_sample_weight(
            class_weight='balanced',
            y=y
        )
        clf = MLPClassifier(
            hidden_layer_sizes=kwargs.get('hidden', (32,)),
            activation='relu',
            alpha=kwargs.get('alpha', 1e-3),
            early_stopping=True,
            max_iter=500
        )

        clf.fit(X, y, sample_weight=sample_weight)

        return clf, None

    elif model == 'ridge_regression':
        clf = Ridge(alpha=kwargs.get('alpha', 1.0))
        clf.fit(X, y)
        axis = clf.coef_.ravel()
        axis /= np.linalg.norm(axis)
        return clf, axis

    else:
        raise ValueError(f'Unknown model: {model}')


# -----------------------------------------------------------
# Cross-validation
# -----------------------------------------------------------
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold

# -----------------------------------------------------------
def find_best_accuracy_threshold(prob, y_true):
    """
    Find threshold t that maximizes raw accuracy on y_true.
    prob: predicted probabilities of class 1.
    y_true: binary labels (0/1).
    """
    prob = np.asarray(prob)
    y_true = np.asarray(y_true).astype(int)

    # Candidate thresholds are all unique probability values
    thresholds = np.unique(prob)

    # Accuracy for each threshold
    accuracies = [
        accuracy_score(y_true, (prob >= t).astype(int))
        for t in thresholds
    ]

    best_idx = np.argmax(accuracies)
    return thresholds[best_idx], accuracies[best_idx]
# -----------------------------------------------------------


from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

def _fit_preprocess_fr(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    return mu, sd


def _apply_preprocess_fr(X, mu, sd):
    return (X - mu) / sd


def cross_validate_axis(X, y, model='logreg', n_splits=5, **kwargs):
    """
    Cross-validate behavioral readout.
    Fully CV-safe: preprocessing is fit on training folds only.
    """

    y_int = y.astype(int)
    unique_classes = np.unique(y_int)

    if len(unique_classes) < 2:
        return {
            'mean_accuracy': np.nan,
            'mean_auc': np.nan,
            'axis_cosine_similarity': np.nan,
            'all_axes': [],
            'all_accuracies': [],
            'all_aucs': []
        }

    class_counts = np.bincount(y_int)
    if class_counts.min() >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(X, y_int)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(X)

    axes = []
    accs = []
    aucs = []

    for tr, te in split_iter:
        Xtr_raw, Xte_raw = X[tr], X[te]
        ytr, yte = y_int[tr], y_int[te]

        # ---- FIT PREPROCESSING ON TRAIN ONLY ----
        mu, sd = _fit_preprocess_fr(Xtr_raw)
        Xtr = _apply_preprocess_fr(Xtr_raw, mu, sd)
        Xte = _apply_preprocess_fr(Xte_raw, mu, sd)

        # ---- Fit model ----
        readout, axis = fit_behavior_axis(Xtr, ytr, model=model, **kwargs)

        # ---- Compute scores ----
        if axis is not None:
            # Linear model
            if hasattr(readout, 'decision_function'):
                score_te = readout.decision_function(Xte)
                score_tr = readout.decision_function(Xtr)
            else:
                score_te = Xte @ axis
                score_tr = Xtr @ axis
        else:
            # Nonlinear model
            if hasattr(readout, 'decision_function'):
                score_tr = readout.decision_function(Xtr)
                score_te = readout.decision_function(Xte)
            elif hasattr(readout, 'predict_proba'):
                score_tr = readout.predict_proba(Xtr)[:, 1]
                score_te = readout.predict_proba(Xte)[:, 1]
            else:
                score_tr = readout.predict(Xtr)
                score_te = readout.predict(Xte)

        # ---- Threshold selection (score space) ----
        t_star, _ = find_best_accuracy_threshold(score_tr, ytr)

        y_pred_te = (score_te >= t_star).astype(int)
        accs.append(accuracy_score(yte, y_pred_te))

        try:
            aucs.append(roc_auc_score(yte, score_te))
        except Exception:
            aucs.append(np.nan)

        axes.append(axis)

    # ---- Axis cosine similarity (linear models only) ----
    valid_axes = [a for a in axes if a is not None]

    if len(valid_axes) >= 2:
        ref = valid_axes[0]
        aligned_axes = []
        for a in valid_axes:
            if np.dot(a, ref) < 0:
                a = -a
            aligned_axes.append(a)

        cosines = []
        for i in range(len(aligned_axes)):
            for j in range(i + 1, len(aligned_axes)):
                cosines.append(
                    np.dot(aligned_axes[i], aligned_axes[j]) /
                    (np.linalg.norm(aligned_axes[i]) *
                     np.linalg.norm(aligned_axes[j]) + 1e-12)
                )

        axis_cosine_similarity = float(np.nanmean(cosines))
    else:
        axis_cosine_similarity = np.nan

    return {
        'mean_accuracy': float(np.nanmean(accs)),
        'mean_auc': float(np.nanmean(aucs)),
        'axis_cosine_similarity': axis_cosine_similarity,
        'all_axes': axes,
        'all_accuracies': accs,
        'all_aucs': aucs
    }


# -----------------------------------------------------------
# Bootstrap CI utility
# -----------------------------------------------------------
def bootstrap_ci(aligned_data, ci=95, n_boot=500):
    n_events, T = aligned_data.shape
    boot_means = np.zeros((n_boot, T))

    for i in range(n_boot):
        idx = np.random.randint(0, n_events, size=n_events)
        boot_means[i] = aligned_data[idx].mean(axis=0)

    alpha = (100-ci)/2
    lower = np.percentile(boot_means, alpha, axis=0)
    upper = np.percentile(boot_means, 100-alpha, axis=0)
    return lower, upper


# -----------------------------------------------------------
# Axis orthogonalization & angle
# -----------------------------------------------------------
def orthogonalize_axes(axes):
    out = []
    for v in axes:
        w = v.copy()
        for u in out:
            w -= np.dot(w, u)*u
        w /= np.linalg.norm(w)
        out.append(w)
    return out


def axis_angle(a, b):
    x = np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))
    x = np.clip(x, -1, 1)
    return float(np.degrees(np.arccos(x)))


def compute_timepoint_pvals(aligned_A, aligned_B):
    """
    Compute Mann–Whitney U p-values at each timepoint.
    Returns array shape (T,)
    """
    _, T = aligned_A.shape
    pvals = np.zeros(T)

    for t in range(T):
        try:
            _, p = mannwhitneyu(
                aligned_A[:, t], aligned_B[:, t], alternative="two-sided")
        except:
            p = 1.0
        pvals[t] = p

    return pvals


def find_clusters(sig_mask):
    """
    Given boolean vector (T,), return list of clusters.
    Each cluster is a list/array of time indices.
    """
    clusters = []
    current = []

    for i, sig in enumerate(sig_mask):
        if sig:
            current.append(i)
        else:
            if current:
                clusters.append(np.array(current))
                current = []
    if current:
        clusters.append(np.array(current))

    return clusters


def cluster_mass(cluster_idx, pvals):
    """
    Compute cluster mass = ∑ -log(pvals)
    """
    return np.sum(-np.log(pvals[cluster_idx]))


def cluster_permutation_test(
    aligned_A,
    aligned_B,
    alpha=0.05,
    cluster_alpha=0.05,
    n_perm=1000
):
    """
    Fast cluster-based permutation test using vectorized label shuffling.

    Parameters
    ----------
    aligned_A : (nA, T)
    aligned_B : (nB, T)

    Returns
    -------
    corrected_mask : (T,) boolean
    """

    # ----------------------------------------------------------
    # 0. Real p-values and clusters
    # ----------------------------------------------------------
    pvals = compute_timepoint_pvals(aligned_A, aligned_B)
    sig_mask = pvals < alpha
    clusters = find_clusters(sig_mask)

    if len(clusters) == 0:
        return np.zeros_like(sig_mask, dtype=bool)

    real_masses = np.array([cluster_mass(c, pvals) for c in clusters])

    # ----------------------------------------------------------
    # 1. Batch data for permutations
    # ----------------------------------------------------------
    nA, T = aligned_A.shape
    nB = aligned_B.shape[0]

    all_data = np.vstack([aligned_A, aligned_B])     # shape (nA+nB, T)
    N = nA + nB

    # ----------------------------------------------------------
    # 2. Generate all permutation labelings at once
    # ----------------------------------------------------------
    # Each perm is a shuffled index array shape (n_perm, N)
    perm_indices = np.array([np.random.permutation(N) for _ in range(n_perm)])

    # First nA rows → A; remaining rows → B
    perm_A_idx = perm_indices[:, :nA]     # shape (n_perm, nA)
    perm_B_idx = perm_indices[:, nA:]     # shape (n_perm, nB)

    # ----------------------------------------------------------
    # 3. Build all permuted datasets (batched)
    # ----------------------------------------------------------
    # perm_A : (n_perm, nA, T)
    # perm_B : (n_perm, nB, T)
    perm_A = all_data[perm_A_idx]   # fancy indexing batches properly
    perm_B = all_data[perm_B_idx]

    # ----------------------------------------------------------
    # 4. Compute permutation p-values (vectorized over perms)
    # ----------------------------------------------------------
    # We loop over timepoints (T), but **not** over permutations.
    # This reduces Python overhead by ~N_perm fold.
    p_perm_all = np.zeros((n_perm, T), dtype=float)

    for t in range(T):
        # column vector for all perms
        Avals = perm_A[:, :, t]          # shape (n_perm, nA)
        Bvals = perm_B[:, :, t]          # shape (n_perm, nB)

        # vectorized Mann-Whitney approximation:
        # fallback: loop over permutations but not over T
        p_perm = np.zeros(n_perm)
        for i in range(n_perm):
            try:
                _, p = mannwhitneyu(
                    Avals[i], Bvals[i], alternative='two-sided')
            except:
                p = 1.0
            p_perm[i] = p

        p_perm_all[:, t] = p_perm

    # ----------------------------------------------------------
    # 5. Get clusters & masses for each perm
    # ----------------------------------------------------------
    null_masses = np.zeros(n_perm, float)

    for i in range(n_perm):
        mask = p_perm_all[i] < alpha
        clusters_perm = find_clusters(mask)
        if len(clusters_perm) == 0:
            null_masses[i] = 0
        else:
            masses = [cluster_mass(c, p_perm_all[i]) for c in clusters_perm]
            null_masses[i] = np.max(masses)

    # ----------------------------------------------------------
    # 6. Threshold for corrected significance
    # ----------------------------------------------------------
    thresh = np.percentile(null_masses, 100 * (1 - cluster_alpha))

    corrected_mask = np.zeros_like(sig_mask, dtype=bool)
    for c, mass in zip(clusters, real_masses):
        if mass > thresh:
            corrected_mask[c] = True

    return corrected_mask
