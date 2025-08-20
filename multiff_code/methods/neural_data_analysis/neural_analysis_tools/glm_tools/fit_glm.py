import numpy as np
from scipy.signal import fftconvolve
from scipy.special import gammaln
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold

# -------------------- Bases & design helpers --------------------

def raised_cosine_basis(n_basis, t_max, dt, t_min=0.0, log_spaced=True, eps=1e-3):
    """
    Causal raised-cosine basis functions that tile [t_min, t_max].
    Columns are normalized to unit area (sum * dt = 1).
    Returns: lags (L,), B (L x K)
    """
    lags = np.arange(0, t_max + dt, dt)

    def warp(x):  # denser near 0 if log_spaced
        return np.log(x + eps) if log_spaced else x

    W = warp(lags)
    c = np.linspace(warp(t_min + eps), warp(t_max), n_basis)
    w = (c[1] - c[0]) if n_basis > 1 else (warp(t_max) - warp(t_min) + 1e-6)

    B = []
    for ci in c:
        arg = (W - ci) * np.pi / w
        b = 0.5 * (1 + np.cos(np.clip(arg, -np.pi, np.pi)))
        b[(W < ci - w) | (W > ci + w)] = 0.0
        B.append(b)
    B = np.stack(B, axis=1)
    B /= (B.sum(axis=0, keepdims=True) * dt + 1e-12)  # unit area
    return lags, B

def convolve_causal(x, k):
    """
    Causal convolution of time series x with kernel k (defined on nonnegative lags).
    Output length matches x (truncate 'full' at T).
    """
    return fftconvolve(x, k, mode='full')[:len(x)]

def build_design_for_trial(trial, dt, bases_cfg):
    """
    trial: dict with keys:
      - 'y': counts per bin (T,)
      - optional continuous series: e.g., 'speed', 'dist', ... each (T,)
      - optional events as binary series: e.g., 'go', 'flash', ... each (T,)
      - optional angle series in radians: e.g., 'heading' (T,)
    bases_cfg: dict describing which features get which bases.
      Example:
        bases_cfg = {
           'speed':  {'kind':'cont',  't_max':0.30, 'K':6, 'log':True},
           'go':     {'kind':'event', 't_max':0.50, 'K':6, 'log':True},
           'hist':   {'kind':'hist',  't_max_short':0.025, 'K_short':3,
                                    't_max_long':0.250, 'K_long':5},
           'heading':{'kind':'angle'}  # instant sin/cos (no lags)
        }
    Returns: X_trial (T x P), colnames (list of str)
    """
    y = trial['y']
    T = len(y)
    X_cols = []
    names = []

    # 1) Continuous covariates with temporal filters (project onto basis)
    for key, cfg in bases_cfg.items():
        if cfg.get('kind') == 'cont' and key in trial:
            lags, B = raised_cosine_basis(cfg['K'], cfg['t_max'], dt,
                                          t_min=0.0, log_spaced=cfg.get('log', True))
            for k in range(B.shape[1]):
                X_cols.append(convolve_causal(trial[key], B[:, k]))
                names.append(f"{key}_rc{k+1}")
    # 2) Event covariates (binary impulses convolved with basis)
    for key, cfg in bases_cfg.items():
        if cfg.get('kind') == 'event' and key in trial:
            lags, B = raised_cosine_basis(cfg['K'], cfg['t_max'], dt,
                                          t_min=0.0, log_spaced=cfg.get('log', True))
            for k in range(B.shape[1]):
                X_cols.append(convolve_causal(trial[key], B[:, k]))
                names.append(f"{key}_rc{k+1}")

    # 3) Spike history: fine 0–t_max_short and coarse t_max_short–t_max_long
    if 'hist' in bases_cfg:
        cfg = bases_cfg['hist']

        # Full lag grid up to the long window (for padding)
        l_full = np.arange(0, cfg['t_max_long'] + dt, dt)
        L_full = len(l_full)

        # Short window bases (0 .. t_max_short), linear spacing
        l1, B1 = raised_cosine_basis(
            n_basis=cfg['K_short'],
            t_max=cfg['t_max_short'],
            dt=dt,
            t_min=0.0,
            log_spaced=False
        )

        # Long window bases (t_max_short .. t_max_long), log spacing
        l2, B2 = raised_cosine_basis(
            n_basis=cfg['K_long'],
            t_max=cfg['t_max_long'],
            dt=dt,
            t_min=cfg['t_max_short'],
            log_spaced=True
        )
        # B2 is already defined on [0 .. t_max_long] (zeros before t_min),
        # so its number of rows should be L_full. B1 has fewer rows; pad it.

        # Zero-pad B1 to the full length
        pad_B1 = np.zeros((L_full, B1.shape[1]))
        pad_B1[:len(l1), :] = B1

        # Make sure B2 has the same number of rows (guard small rounding diffs)
        if len(l2) != L_full:
            pad_B2 = np.zeros((L_full, B2.shape[1]))
            L = min(L_full, len(l2))
            pad_B2[:L, :] = B2[:L, :]
            Bhist = np.hstack([pad_B1, pad_B2])  # columns = K_short + K_long
        else:
            Bhist = np.hstack([pad_B1, B2])

        # Convolve trial spikes with each history basis column (causal)
        for k in range(Bhist.shape[1]):
            X_cols.append(convolve_causal(y, Bhist[:, k]))
            names.append(f"hist_rc{k+1}")


    # 4) Instantaneous angle features (no lags): sin/cos for circular variables
    for key, cfg in bases_cfg.items():
        if cfg.get('kind') == 'angle' and key in trial:
            ang = trial[key]
            X_cols.append(np.sin(ang)); names.append(f"{key}_sin")
            X_cols.append(np.cos(ang)); names.append(f"{key}_cos")

    if not X_cols:
        X = np.zeros((T, 0))
    else:
        X = np.column_stack(X_cols)
    return X, names

# -------------------- Metrics --------------------

def loglik_poisson(y, mu):
    """Sum log-likelihood for Poisson counts with mean mu (counts/bin)."""
    mu = np.clip(mu, 1e-12, None)
    return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))

def bits_per_spike(y, mu, mu0=None):
    """
    Bits/spike relative to a baseline (mu0 = homogeneous mean if None).
    y, mu, mu0 are in counts/bin.
    """
    if mu0 is None:
        mu0 = np.full_like(y, y.sum() / len(y))
    LLm = loglik_poisson(y, mu)
    LL0 = loglik_poisson(y, mu0)
    return (LLm - LL0) / (y.sum() * np.log(2) + 1e-12)

def psth_from_trials(trials_counts, dt, smooth_kernel=None):
    """
    Average counts across trials, optional smoothing on counts,
    then convert to Hz (divide by dt).
    """
    M = np.mean(np.stack(trials_counts), axis=0)  # counts/bin
    if smooth_kernel is not None:
        M = np.convolve(M, smooth_kernel, mode='same')
    return M / dt  # Hz

def split_half_sb(test_trials_counts, dt, n_splits=200, rng=0, center=True, smooth_kernel=None):
    rng = np.random.default_rng(rng)
    counts = list(test_trials_counts)
    rsb = []
    for _ in range(n_splits):
        idx = rng.permutation(len(counts))
        A = [counts[i] for i in idx[::2]]
        B = [counts[i] for i in idx[1::2]]
        if len(A) == 0 or len(B) == 0:
            continue
        pA = psth_from_trials(A, dt, smooth_kernel)  # << same smoothing as eval
        pB = psth_from_trials(B, dt, smooth_kernel)
        if center:
            pA -= pA.mean(); pB -= pB.mean()
        r = np.dot(pA, pB) / (np.linalg.norm(pA)*np.linalg.norm(pB) + 1e-12)
        rsb.append(2*r / (1 + r))  # Spearman–Brown
    return float(np.mean(rsb)) if rsb else np.nan

# -------------------- Cross-validated GLM fit --------------------

def fit_glm_cv(trials, dt, bases_cfg, alphas=np.logspace(-4, 1, 8),
               n_splits=5, shuffle_trials=True, random_state=0):
    """
    trials: list of per-trial dicts. Each must contain:
        'y'  (counts, shape T,)
      Optional per-trial arrays of same length T:
        e.g., 'speed', 'heading', 'go', ...
    Returns dict with model, scalers, column names, and CV metrics.
    """
    # Build per-trial design matrices
    X_trials, y_trials = [], []
    for tr in trials:
        Xtr, _names = build_design_for_trial(tr, dt, bases_cfg)
        X_trials.append(Xtr); y_trials.append(tr['y'])

    # CV split by trials (NOT by time-bins)
    kf = KFold(n_splits=n_splits, shuffle=shuffle_trials, random_state=random_state)

    # Hyperparameter search (ridge strength)
    best_alpha, best_ll = None, -np.inf
    for a in alphas:
        ll_sum = 0.0
        for tr_idx, te_idx in kf.split(X_trials):
            # Assemble train arrays
            Xtr = np.vstack([X_trials[i] for i in tr_idx])
            ytr = np.concatenate([y_trials[i] for i in tr_idx])

            # Fit scaler on TRAIN ONLY
            scaler = StandardScaler(with_mean=True, with_std=True)
            Xtrz = scaler.fit_transform(Xtr)

            # Fit Poisson GLM (ridge)
            model = PoissonRegressor(alpha=a, max_iter=5000, fit_intercept=True)
            model.fit(Xtrz, ytr)

            # Evaluate on TEST trials
            ll_fold = 0.0
            for i in te_idx:
                Xte = X_trials[i]; yte = y_trials[i]
                Xtez = scaler.transform(Xte)
                mu = model.predict(Xtez)  # mean counts/bin
                ll_fold += loglik_poisson(yte, mu)
            ll_sum += ll_fold
        if ll_sum > best_ll:
            best_ll, best_alpha = ll_sum, a

    # Refit on ALL trials with best alpha
    X_all = np.vstack(X_trials)
    y_all = np.concatenate(y_trials)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_all_z = scaler.fit_transform(X_all)
    final_model = PoissonRegressor(alpha=best_alpha, max_iter=5000, fit_intercept=True)
    final_model.fit(X_all_z, y_all)

    return {
        "model": final_model,
        "scaler": scaler,
        "X_trials": X_trials,
        "y_trials": y_trials,
        "colnames": _names,  # from the last built trial (same across trials)
        "dt": dt,
        "best_alpha": best_alpha
    }

# -------------------- Evaluation on held-out trials --------------------

def evaluate_on_test(train_trials, test_trials, dt, bases_cfg, model_bundle,
                     smooth_sigma_ms=None):
    """
    Build PSTHs & metrics on held-out TEST trials.
    Optionally smooth counts (both data & model) with Gaussian kernel (σ in ms).
    """
    # Gaussian kernel (on counts) that preserves units (sum*dt = 1)
    smooth_kernel = None
    if smooth_sigma_ms is not None:
        sigma_s = smooth_sigma_ms / 1000.0
        sigma_bins = max(1, int(round(sigma_s / dt)))
        n = np.arange(-5*sigma_bins, 5*sigma_bins + 1)
        g = np.exp(-0.5 * (n / sigma_bins)**2)
        g /= (g.sum() * dt)
        smooth_kernel = g

    # Build per-trial designs for TEST
    X_te, y_te = [], []
    for tr in test_trials:
        X, _ = build_design_for_trial(tr, dt, bases_cfg)
        X_te.append(X); y_te.append(tr['y'])

    # Predict per test trial
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    mu_te = []
    for X, y in zip(X_te, y_te):
        mu = model.predict(scaler.transform(X))  # counts/bin
        mu_te.append(mu)

    # Metrics: held-out LL and bits/spike (concatenated across test trials)
    y_cat  = np.concatenate(y_te)
    mu_cat = np.concatenate(mu_te)
    bps = bits_per_spike(y_cat, mu_cat)  # vs homogeneous baseline on test
    LL  = loglik_poisson(y_cat, mu_cat)

    # PSTH correlation and ceiling-normalized score on TEST
    psth_data  = psth_from_trials(y_te, dt, smooth_kernel)
    psth_model = psth_from_trials(mu_te, dt, smooth_kernel)
    # mean-center before correlation if you care about *shape*
    r_model = np.corrcoef(psth_model - psth_model.mean(),
                          psth_data  - psth_data.mean())[0, 1]
    r_sb = split_half_sb(y_te, dt, n_splits=200, rng=1, center=True)
    r_ceiling = np.sqrt(max(r_sb, 0.0))
    r_norm = np.clip(r_model / (r_ceiling + 1e-12), 0, 1) if np.isfinite(r_ceiling) else np.nan

    return {
        "LL_test": LL,
        "bits_per_spike_test": bps,
        "psth_corr_test": r_model,
        "psth_ceiling_rsb": r_sb,
        "psth_ceiling_sqrt": r_ceiling,
        "psth_corr_normalized": r_norm,
        "psth_model_Hz": psth_model,
        "psth_data_Hz": psth_data
    }
