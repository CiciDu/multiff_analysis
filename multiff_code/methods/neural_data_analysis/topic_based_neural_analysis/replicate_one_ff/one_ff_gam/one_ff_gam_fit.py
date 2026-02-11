import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import ttest_1samp


@dataclass
class GroupSpec:
    name: str
    cols: List[str]
    vartype: str               # '1D', 'event', '1Dcirc', '2D', '0D'
    lam: float                 # penalty strength (lambda)


@dataclass
class FitResult:
    coef: pd.Series            # beta indexed by design_df columns
    success: bool
    message: str
    n_iter: int
    fun: float
    grad_norm: float


# -----------------------------
# Smooth helper functions
# -----------------------------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _softplus(x: np.ndarray, k: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    softplus(x) = (1/k) * log(1 + exp(kx))
    Returns (sp, sp', sp'') w.r.t x.
    """
    kx = np.clip(k * x, -60.0, 60.0)
    sp = (1.0 / k) * np.log1p(np.exp(kx))
    s = _sigmoid(kx)                 # sp' = sigmoid(kx)
    sp1 = s
    sp2 = k * s * (1.0 - s)          # d/dx sigmoid(kx)
    return sp, sp1, sp2


def _softclip(u: np.ndarray, lo: float, hi: float, k: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smoothly clip u into [lo, hi] with smoothness controlled by k.

    Returns:
      u_tilde, du_tilde/du, d2u_tilde/du2  (elementwise)
    """
    # Lower smooth bound: v = lo + softplus(u - lo)
    sp_lo, sp1_lo, sp2_lo = _softplus(u - lo, k=k)
    v = lo + sp_lo
    dv = sp1_lo
    d2v = sp2_lo

    # Upper smooth bound: w = hi - softplus(hi - v)
    sp_hi, sp1_hi, sp2_hi = _softplus(hi - v, k=k)
    w = hi - sp_hi

    # dw/dv and d2w/dv2
    dw_dv = sp1_hi
    d2w_dv2 = -sp2_hi

    # Chain to u
    dw = dw_dv * dv
    d2w = d2w_dv2 * (dv ** 2) + dw_dv * d2v

    return w, dw, d2w


# -----------------------------
# Penalty matrices (match MATLAB)
# -----------------------------
def _dd1_matrix(n: int, circ: bool) -> sparse.csr_matrix:
    """
    Construct DD1 = D1.T @ D1 where D1 is first-difference operator.
    Matches MATLAB: D1 = spdiags([-1, 1], [0, 1], n-1, n)

    For circ=True, emulate wrap correction by adjusting first/last rows.
    """
    if n <= 1:
        return sparse.csr_matrix((n, n))

    diags = np.vstack([-np.ones(n - 1), np.ones(n - 1)])
    D1 = sparse.diags(diagonals=diags, offsets=[0, 1], shape=(n - 1, n), format='csr')
    DD1 = (D1.T @ D1).tocsr()

    if circ and n >= 2:
        DD1 = DD1.tolil()
        row2 = DD1[1, :].toarray().ravel()
        row_endm1 = DD1[-2, :].toarray().ravel()
        DD1[0, :] = np.roll(row2, -1)
        DD1[-1, :] = np.roll(row_endm1, 1)
        DD1 = DD1.tocsr()

    return DD1


def _laplacian_2d_matrix(n: int) -> sparse.csr_matrix:
    """
    Match MATLAB 2D penalty:
      let s = sqrt(n)
      DD1 = D1.T@D1 for 1D bins of length s
      M = kron(I, DD1) + kron(DD1, I)

    Requires n to be a perfect square.
    """
    s = int(round(np.sqrt(n)))
    if s * s != n:
        raise ValueError(f'2D penalty requires perfect square n, got n={n}')

    diags = np.vstack([-np.ones(s - 1), np.ones(s - 1)])
    D1 = sparse.diags(diagonals=diags, offsets=[0, 1], shape=(s - 1, s), format='csr')
    DD1 = (D1.T @ D1).tocsr()

    I = sparse.eye(s, format='csr')
    M = sparse.kron(I, DD1, format='csr') + sparse.kron(DD1, I, format='csr')
    return M.tocsr()


# -----------------------------
# Poisson log-link GAM (MAP)
# -----------------------------
def load_fit_results(save_path: str) -> Dict:
    """
    Load saved GAM fit results from pickle file.
    
    Parameters
    ----------
    save_path : str
        Path to the saved pickle file
    
    Returns
    -------
    Dict
        Dictionary containing:
        - fit_result: dict with coef, success, message, n_iter, fun, grad_norm
        - groups: list of group specifications
        - l1_groups: list of L1 group specifications (if applicable)
        - hyperparameters: dict with l1_smooth_eps, max_iter, tol
        - design_df: design matrix (if save_design=True was used)
        - y: response variable (if save_design=True was used)
        - metadata: additional metadata (if provided during fitting)
    """
    with open(save_path, 'rb') as f:
        return pickle.load(f)


def load_elimination_results(save_path: str) -> Dict:
    """
    Load saved backward elimination results from pickle file.
    
    Parameters
    ----------
    save_path : str
        Path to the saved pickle file
    
    Returns
    -------
    Dict
        Dictionary containing:
        - kept_groups: List of dicts with group specifications that were retained
        - history: List of elimination steps with statistics
        - completed: Boolean indicating if elimination finished
        - current_step: Number of steps completed
        - lambda_config: Dict with 4 main lambda parameters (lam_f, lam_g, lam_h, lam_p)
        - metadata: Additional metadata (if provided during elimination)
        
    Note
    ----
    When using backward_elimination_gam with load_if_exists=True, the function
    automatically validates that lambda values match. This standalone loader
    does not perform validation.
    """
    with open(save_path, 'rb') as f:
        return pickle.load(f)


def build_penalty_blocks(
    groups: List[GroupSpec],
    col_index: Dict[str, int],
    ridge: float = 1e-6,               # tiny ridge to anchor DC mode
) -> Tuple[sparse.csr_matrix, List[np.ndarray]]:
    n_total = len(col_index)

    all_rows: List[np.ndarray] = []
    all_cols: List[np.ndarray] = []
    all_data: List[np.ndarray] = []
    group_indices: List[np.ndarray] = []

    for g in groups:
        if g.lam is None or float(g.lam) == 0.0:
            continue

        idx = np.array([col_index[c] for c in g.cols if c in col_index], dtype=int)
        n = idx.size
        if n == 0:
            continue

        if g.vartype == '2D':
            Pg = float(g.lam) * _laplacian_2d_matrix(n)
        elif g.vartype == '1Dcirc':
            Pg = float(g.lam) * _dd1_matrix(n, circ=True)
        elif g.vartype in ['1D', 'event']:
            Pg = float(g.lam) * _dd1_matrix(n, circ=False)
        elif g.vartype == '0D':
            continue
        else:
            raise ValueError(f'Unknown vartype {g.vartype!r} for group {g.name!r}')

        # add tiny ridge to anchor constant mode (DC)
        if ridge > 0:
            Pg = Pg.tocsr() + ridge * sparse.eye(n, format='csr')

        Pg = Pg.tocoo()
        all_rows.append(idx[Pg.row])
        all_cols.append(idx[Pg.col])
        all_data.append(Pg.data)
        group_indices.append(idx)

    if not all_data:
        return sparse.csr_matrix((n_total, n_total)), group_indices

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    data = np.concatenate(all_data)

    P_full = sparse.coo_matrix((data, (rows, cols)), shape=(n_total, n_total)).tocsr()
    return P_full, group_indices



def poisson_loglik(y, rate):
    """
    Average negative log-likelihood per spike (same normalization as MATLAB).
    """
    eps = 1e-12
    rate = np.clip(rate, eps, None)
    return np.sum(rate - y * np.log(rate) + gammaln(y + 1)) / np.sum(y)


def compute_likelihoods(design_df, beta, y):
    X = design_df.to_numpy()
    u = X @ beta.to_numpy()
    rate = np.exp(u)

    # model
    ll_model = poisson_loglik(y, rate)

    # mean-rate model
    mean_rate = np.mean(y)
    ll_mean = poisson_loglik(y, np.full_like(y, mean_rate))

    # saturated model (best possible)
    ll_sat = poisson_loglik(y, y)

    return ll_model, ll_mean, ll_sat


def pseudo_r2(design_df, beta, y):
    ll_model, ll_mean, ll_sat = compute_likelihoods(design_df, beta, y)
    return (ll_mean - ll_model) / (ll_mean - ll_sat)

def bits_per_spike(design_df, beta, y):
    X = design_df.to_numpy()
    u = X @ beta.to_numpy()
    rate = np.exp(u)

    ll_model = poisson_loglik(y, rate)
    ll_mean = poisson_loglik(y, np.full_like(y, np.mean(y)))

    # convert nats → bits
    return (ll_mean - ll_model) / np.log(2)

def cross_validated_ll(
    design_df,
    y,
    groups,
    n_folds=10,
    random_state=0,
):
    """
    Returns:
        ll_per_fold : (n_folds,) array of log-likelihoods (nats/spike)
    """
    rng = np.random.default_rng(random_state)
    T = len(y)
    indices = np.arange(T)
    rng.shuffle(indices)

    folds = np.array_split(indices, n_folds)
    ll_folds = []

    for k in range(n_folds):
        test_idx = folds[k]
        train_idx = np.setdiff1d(indices, test_idx)

        X_train = design_df.iloc[train_idx]
        y_train = y[train_idx]

        X_test = design_df.iloc[test_idx]
        y_test = y[test_idx]

        fit = fit_poisson_gam_map(
            design_df=X_train,
            y=y_train,
            groups=groups,
            verbose=False,
        )

        ll_test, _, _ = compute_likelihoods(
            X_test, fit.coef, y_test
        )
        ll_folds.append(ll_test)

    return np.array(ll_folds)


def _format_lambda(lam):
    """Format lambda value concisely for display."""
    if lam == 0:
        return '0'
    elif lam >= 1000:
        return f'{lam:.2e}'
    elif lam >= 1:
        return f'{lam:.1f}'
    else:
        return f'{lam:.2e}'


def generate_lambda_suffix(groups, delimiter='_'):
    """
    Generate a filename suffix based on the 4 main lambda parameters.
    
    Parameters
    ----------
    groups : List[GroupSpec]
        Groups with lambda values
    delimiter : str, optional
        Delimiter between lambda pairs, by default '_'
    
    Returns
    -------
    str
        Filename suffix like 'lamF-100_lamG-10_lamH-10_lamP-10'
        
    Examples
    --------
    >>> groups = [
    ...     GroupSpec('t_targ', cols, 'event', lam=10.0),
    ...     GroupSpec('spike_hist', cols, 'event', lam=10.0),
    ...     GroupSpec('position', cols, '1D', lam=100.0)
    ... ]
    >>> generate_lambda_suffix(groups)
    'lamF-100_lamG-10_lamH-10'
    
    Notes
    -----
    Extracts the 4 main lambda parameters:
    - lam_f: firefly features (tuning curves)
    - lam_g: temporal event kernels (t_*)
    - lam_h: spike history
    - lam_p: coupling (cpl_*)
    """
    lambda_config = _extract_lambda_config(groups)
    
    parts = []
    # Order: F, G, H, P for consistency
    for key in ['lam_f', 'lam_g', 'lam_h', 'lam_p']:
        if key in lambda_config:
            lam_str = _format_lambda(lambda_config[key]).replace('.', 'p')
            # Use capitalized short names: lamF, lamG, lamH, lamP
            short_name = key.replace('lam_', 'lam').upper()
            parts.append(f'{short_name}-{lam_str}')
    
    return delimiter.join(parts)


def _extract_lambda_config(groups):
    """
    Extract the 4 main lambda parameters from groups for validation.
    
    Returns:
        dict: {'lam_f': float, 'lam_g': float, 'lam_h': float, 'lam_p': float}
    """
    lambda_config = {}
    
    for g in groups:
        # Determine which lambda type this group uses
        if g.name.startswith('t_'):
            # lam_g: temporal event kernels
            key = 'lam_g'
        elif g.name == 'spike_hist':
            # lam_h: spike history
            key = 'lam_h'
        elif g.name.startswith('cpl_'):
            # lam_p: coupling
            key = 'lam_p'
        else:
            # lam_f: tuning curves (firefly features)
            key = 'lam_f'
        
        # Store or verify consistency
        if key in lambda_config:
            if abs(lambda_config[key] - g.lam) > 1e-10:
                raise ValueError(
                    f"Inconsistent lambda values for {key}: "
                    f"found {lambda_config[key]} and {g.lam}"
                )
        else:
            lambda_config[key] = g.lam
    
    return lambda_config


def _maybe_load_saved_fit(save_path, load_if_exists, verbose):
    if save_path is None or not load_if_exists:
        return None

    save_path = Path(save_path)
    if not save_path.exists():
        return None

    if verbose:
        print('=' * 80)
        print(f'Loading existing results from: {save_path}')
        print('=' * 80)

    saved = load_fit_results(save_path)['fit_result']
    fit_result = FitResult(**saved)

    if verbose:
        print(f'✓ Loaded saved fit result:')
        print(f'  Success: {fit_result.success}')
        print(f'  Iterations: {fit_result.n_iter}')
        print(f'  Final objective: {fit_result.fun:.6e}')
        print(f'  Gradient norm: {fit_result.grad_norm:.3e}')
        print('=' * 80)

    return fit_result


def _build_l1_terms(l1_groups, col_index):
    terms = []
    for g in l1_groups:
        if g.lam is None or float(g.lam) == 0.0:
            continue
        idx = np.array([col_index[c] for c in g.cols if c in col_index], dtype=int)
        if idx.size > 0:
            terms.append((idx, float(g.lam)))
    return terms


def _make_l1_smooth_fns(l1_terms, eps, p):
    def f(beta):
        if not l1_terms:
            return 0.0
        return sum(lam * np.sum(np.sqrt(beta[idx] ** 2 + eps)) for idx, lam in l1_terms)

    def grad(beta):
        g = np.zeros(p)
        for idx, lam in l1_terms:
            b = beta[idx]
            s = np.sqrt(b * b + eps)
            g[idx] += lam * (b / s)
        return g

    def hess_diag(beta):
        h = np.zeros(p)
        for idx, lam in l1_terms:
            b = beta[idx]
            s = np.sqrt(b * b + eps)
            h[idx] += lam * (eps / (s ** 3))
        return h

    return f, grad, hess_diag


def _make_poisson_objective(X, y, P, l1_fun, l1_grad, l1_hess_diag):
    def fun(beta):
        u = X @ beta
        rate = np.exp(u)
        return (
            np.sum(rate - y * u)
            + 0.5 * beta @ (P @ beta)
            + l1_fun(beta)
        )

    def jac(beta):
        u = X @ beta
        rate = np.exp(u)
        return (
            X.T @ (rate - y)
            + P @ beta
            + l1_grad(beta)
        )

    def hessp(beta, v):
        u = X @ beta
        rate = np.exp(u)
        return (
            X.T @ (rate * (X @ v))
            + P @ v
            + l1_hess_diag(beta) * v
        )

    return fun, jac, hessp


def _init_beta(y, p, col_index):
    rng = np.random.default_rng(0)
    beta0 = 1e-3 * rng.standard_normal(p)
    if 'const' in col_index:
        beta0[col_index['const']] = np.log(max(y.mean(), 1e-8))
    return beta0


def _make_callback(fun, jac, beta0):
    state = {'iter': 0, 'prev_beta': beta0.copy()}

    def callback(xk):
        state['iter'] += 1
        f = float(fun(xk))
        gnorm = float(np.linalg.norm(jac(xk)))
        step = float(np.linalg.norm(xk - state['prev_beta']))
        state['prev_beta'] = xk.copy()
        print(f'[iter {state["iter"]:4d}] fun={f: .6e} |grad|={gnorm: .3e} |step|={step: .3e}')

    return callback


def _save_fit_results(
    *,
    save_path,
    fit_result,
    groups,
    l1_groups,
    design_df,
    y,
    save_metadata,
    hyperparameters,
    verbose,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'fit_result': asdict(fit_result),
        'groups': [asdict(g) for g in groups],
        'l1_groups': [asdict(g) for g in l1_groups] if l1_groups else None,
        'hyperparameters': hyperparameters,
    }

    if design_df is not None:
        save_dict['design_df'] = design_df
        save_dict['y'] = y

    if save_metadata is not None:
        save_dict['metadata'] = save_metadata

    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    if verbose:
        print(f'\nResults saved to: {save_path}')
        

def fit_poisson_gam_map(
    design_df: pd.DataFrame,
    y: np.ndarray,
    *,
    groups: List[GroupSpec],
    l1_groups: Optional[List[GroupSpec]] = None,
    l1_smooth_eps: float = 1e-6,
    max_iter: int = 1000,
    tol: float = 1e-6,
    optimizer: str = 'L-BFGS-B',
    verbose: bool = True,
    save_path: Optional[str] = None,
    save_design: bool = False,
    save_metadata: Optional[Dict] = None,
    load_if_exists: bool = True,
) -> FitResult:
    # ------------------------------------------------------------------
    # 1) Load cached result if requested
    # ------------------------------------------------------------------
    maybe_loaded = _maybe_load_saved_fit(
        save_path=save_path,
        load_if_exists=load_if_exists,
        verbose=verbose,
    )
    if maybe_loaded is not None:
        return maybe_loaded

    # ------------------------------------------------------------------
    # 2) Prepare data and penalties
    # ------------------------------------------------------------------
    X = design_df.to_numpy(dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape

    col_index = {c: i for i, c in enumerate(design_df.columns)}
    P, _ = build_penalty_blocks(groups, col_index, ridge=1e-6)

    l1_terms = _build_l1_terms(l1_groups or [], col_index)

    l1_fun, l1_grad, l1_hess_diag = _make_l1_smooth_fns(
        l1_terms=l1_terms,
        eps=l1_smooth_eps,
        p=p,
    )

    # ------------------------------------------------------------------
    # 3) Objective, gradient, Hessian-vector product
    # ------------------------------------------------------------------
    fun, jac, hessp = _make_poisson_objective(
        X=X,
        y=y,
        P=P,
        l1_fun=l1_fun,
        l1_grad=l1_grad,
        l1_hess_diag=l1_hess_diag,
    )

    # ------------------------------------------------------------------
    # 4) Initialization and callback
    # ------------------------------------------------------------------
    beta0 = _init_beta(y=y, p=p, col_index=col_index)

    callback = _make_callback(fun, jac, beta0) if verbose else None

    # ------------------------------------------------------------------
    # 4.5) Diagnostics
    # ------------------------------------------------------------------
    if verbose:
        u0 = X @ beta0
        print('\n' + '=' * 80)
        print('PRE-OPTIMIZATION DIAGNOSTICS')
        print('=' * 80)
        print(f"Design matrix shape: {X.shape}")
        print(f"X range: [{X.min():.2e}, {X.max():.2e}]")
        print(f"y range: [{y.min():.2e}, {y.max():.2e}]")
        print(f"y mean: {y.mean():.2e}, y sum: {y.sum():.2e}")
        print(f"Initial beta range: [{beta0.min():.2e}, {beta0.max():.2e}]")
        print(f"Initial u = X @ beta0 range: [{u0.min():.2e}, {u0.max():.2e}]")
        print(f"Initial rate = exp(u) range: [{np.exp(u0).min():.2e}, {np.exp(u0).max():.2e}]")
        print(f"Initial objective: {fun(beta0):.6e}")
        print(f"Initial gradient norm: {np.linalg.norm(jac(beta0)):.3e}")
        print('=' * 80 + '\n')

    # ------------------------------------------------------------------
    # 5) Optimize
    # ------------------------------------------------------------------
    if optimizer.upper() == 'L-BFGS-B':
        res = minimize(
            fun=fun,
            x0=beta0,
            method='L-BFGS-B',
            jac=jac,
            callback=callback,
            options={'maxiter': int(max_iter), 'ftol': float(tol), 'disp': bool(verbose)},
        )
    elif optimizer.lower() == 'trust-ncg':
        res = minimize(
            fun=fun,
            x0=beta0,
            method='trust-ncg',
            jac=jac,
            hessp=hessp,
            callback=callback,
            options={'maxiter': int(max_iter), 'gtol': float(tol), 'disp': bool(verbose)},
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}. Choose 'L-BFGS-B' or 'trust-ncg'.")

    beta_hat = res.x
    grad_norm = float(np.linalg.norm(jac(beta_hat)))

    fit_result = FitResult(
        coef=pd.Series(beta_hat, index=design_df.columns, name='beta'),
        success=bool(res.success),
        message=str(res.message),
        n_iter=int(getattr(res, 'nit', -1)),
        fun=float(res.fun),
        grad_norm=grad_norm,
    )
    
    # print message
    print(f'fit_result.message: {fit_result.message}')

    # ------------------------------------------------------------------
    # 6) Save results
    # ------------------------------------------------------------------
    if save_path is not None:
        _save_fit_results(
            save_path=save_path,
            fit_result=fit_result,
            groups=groups,
            l1_groups=l1_groups,
            design_df=design_df if save_design else None,
            y=y if save_design else None,
            save_metadata=save_metadata,
            hyperparameters={
                'l1_smooth_eps': l1_smooth_eps,
                'max_iter': max_iter,
                'tol': tol,
                'optimizer': optimizer,
            },
            verbose=verbose,
        )

    return fit_result
