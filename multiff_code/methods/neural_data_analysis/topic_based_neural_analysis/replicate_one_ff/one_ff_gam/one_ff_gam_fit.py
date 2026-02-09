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
def fit_poisson_gam_map(
    design_df: pd.DataFrame,
    y: np.ndarray,
    *,
    groups: List[GroupSpec],
    l1_groups: Optional[List[GroupSpec]] = None,
    l1_smooth_eps: float = 1e-6,
    max_iter: int = 200,
    tol: float = 1e-6,
    verbose: bool = True,
    save_path: Optional[str] = None,
    save_design: bool = False,
    save_metadata: Optional[Dict] = None,
    # softclip_lo: float = -50.0,
    # softclip_hi: float = 50.0,
    # softclip_k: float = 5.0,
) -> FitResult:
    """
    Fit Poisson log-link GAM via MAP:
      f(beta) = sum(exp(u_tilde) - y*u_tilde) + 0.5*beta^T P beta + L1_terms

    This version FIXES SciPy trust-ncg sparse Hessian issues by providing hessp
    (Hessian-vector product) instead of returning a sparse Hessian.

    Numerics:
    - Smooth soft-clipping of u to keep exp stable while keeping derivatives consistent.
    - Smooth L1: |b| ≈ sqrt(b^2 + eps) so Newton/trust-region remains valid.
    
    Parameters
    ----------
    design_df : pd.DataFrame
        Design matrix with columns corresponding to features
    y : np.ndarray
        Response variable (spike counts)
    groups : List[GroupSpec]
        List of GroupSpec objects defining penalty groups
    l1_groups : Optional[List[GroupSpec]], optional
        List of GroupSpec objects for L1 penalty, by default None
    l1_smooth_eps : float, optional
        Smoothing parameter for L1 penalty, by default 1e-6
    max_iter : int, optional
        Maximum number of optimization iterations, by default 200
    tol : float, optional
        Tolerance for optimization convergence, by default 1e-6
    verbose : bool, optional
        Print optimization progress, by default True
    save_path : Optional[str], optional
        Path to save fit results (as pickle file). If None, results are not saved, by default None
    save_design : bool, optional
        If True, save design_df and y along with fit results, by default False
    save_metadata : Optional[Dict], optional
        Additional metadata to save (e.g., neuron ID, session info), by default None
    
    Returns
    -------
    FitResult
        Fitted model results including coefficients and convergence information
    """
    X = design_df.to_numpy(dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape

    col_index = {c: i for i, c in enumerate(design_df.columns)}
    P, _ = build_penalty_blocks(groups, col_index, ridge=1e-6)

    # L1 groups (smooth |b| with sqrt(b^2 + eps))
    l1_groups = l1_groups or []
    l1_terms: List[Tuple[np.ndarray, float]] = []
    for g in l1_groups:
        if g.lam is None or float(g.lam) == 0.0:
            continue
        idx = np.array([col_index[c] for c in g.cols if c in col_index], dtype=int)
        if idx.size == 0:
            continue
        l1_terms.append((idx, float(g.lam)))

    def _l1_smooth_diag(beta: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Smooth L1:
          phi(b) = sqrt(b^2 + eps)
        Accumulates:
          f_l1 (scalar), grad_l1 (p,), hess_l1_diag (p,)
        """
        if not l1_terms:
            return 0.0, np.zeros_like(beta), np.zeros_like(beta)

        eps = float(l1_smooth_eps)
        f_l1 = 0.0
        grad_l1 = np.zeros_like(beta)
        hess_l1_diag = np.zeros_like(beta)

        for idx, lam in l1_terms:
            b = beta[idx]
            s = np.sqrt(b * b + eps)
            f_l1 += lam * float(np.sum(s))
            grad_l1[idx] += lam * (b / s)
            hess_l1_diag[idx] += lam * (eps / (s ** 3))

        return f_l1, grad_l1, hess_l1_diag

    def _core_terms(beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Exact Poisson terms (no soft clipping) so Hessian is PSD and matches MATLAB.
        Returns: u, du (ones), d2u (zeros), rate, w2 (rate)
        """
        u = X @ beta
        rate = np.exp(u)     # exact
        # For exact Poisson with u = Xb:
        # du = 1, d2u = 0 so
        # w2 = rate (since rate * du^2 + (rate - y)*d2u = rate)
        du = np.ones_like(u)
        d2u = np.zeros_like(u)
        w2 = rate
        return u, du, d2u, rate, w2

    def fun(beta: np.ndarray) -> float:
        u_tilde, _, _, rate, _ = _core_terms(beta)

        f_data = float(np.sum(rate - y * u_tilde))

        Pb = P @ beta
        f_pen = 0.5 * float(beta @ Pb)

        f_l1, _, _ = _l1_smooth_diag(beta)

        return f_data + f_pen + f_l1

    def jac(beta: np.ndarray) -> np.ndarray:
        u_tilde, du, _, rate, _ = _core_terms(beta)

        # df/dbeta from data term via chain: X.T @ ((rate - y) * du)
        w1 = (rate - y) * du
        grad_data = X.T @ w1

        grad_pen = P @ beta

        _, grad_l1, _ = _l1_smooth_diag(beta)

        return grad_data + grad_pen + grad_l1

    def hessp(beta: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Hessian-vector product:
          (X.T diag(w2) X) v + P v + diag(hess_l1_diag) v
        """
        _, _, _, _, w2 = _core_terms(beta)

        Xv = X @ v
        Hv_data = X.T @ (w2 * Xv)

        Hv_pen = P @ v

        _, _, hess_l1_diag = _l1_smooth_diag(beta)
        Hv_l1 = hess_l1_diag * v

        return Hv_data + Hv_pen + Hv_l1

    rng = np.random.default_rng(0)
    beta0 = 1e-3 * rng.standard_normal(p)

    # sensible intercept init if column exists
    if 'const' in col_index:
        idx0 = col_index['const']
        eps = 1e-8
        beta0[idx0] = np.log(np.maximum(y.mean(), eps))
        
    # --- INSERT THIS BLOCK right before:  res = minimize(...)
    progress_state = {'iter': 0, 'prev_beta': beta0.copy()}

    def _callback(xk: np.ndarray) -> None:
        # Called once per iteration by scipy.optimize.minimize
        progress_state['iter'] += 1

        f = float(fun(xk))
        gnorm = float(np.linalg.norm(jac(xk)))
        step_norm = float(np.linalg.norm(xk - progress_state['prev_beta']))
        progress_state['prev_beta'] = xk.copy()

        print(f'[iter {progress_state["iter"]:4d}] fun={f: .6e} |grad|={gnorm: .3e} |step|={step_norm: .3e}')
        
           
    res = minimize(
        fun=fun,
        x0=beta0,
        method='trust-ncg',
        jac=jac,
        hessp=hessp,               # <-- key fix (no sparse Hessian returned)
        callback=_callback if verbose else None,        # <-- ADD THIS LINE
        options={'maxiter': int(max_iter), 'gtol': float(tol), 'disp': bool(verbose)},
    )

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
    
    # Save results if save_path is provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'fit_result': {
                'coef': fit_result.coef,
                'success': fit_result.success,
                'message': fit_result.message,
                'n_iter': fit_result.n_iter,
                'fun': fit_result.fun,
                'grad_norm': fit_result.grad_norm,
            },
            'groups': [asdict(g) for g in groups],
            'l1_groups': [asdict(g) for g in l1_groups] if l1_groups else None,
            'hyperparameters': {
                'l1_smooth_eps': l1_smooth_eps,
                'max_iter': max_iter,
                'tol': tol,
            },
        }
        
        if save_design:
            save_dict['design_df'] = design_df
            save_dict['y'] = y
        
        if save_metadata is not None:
            save_dict['metadata'] = save_metadata
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        if verbose:
            print(f'\nResults saved to: {save_path}')

    return fit_result
    
    
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
        - lambda_config: Dict mapping group names to lambda values used
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
    Generate a filename suffix based on lambda configuration.
    
    Parameters
    ----------
    groups : List[GroupSpec]
        Groups with lambda values
    delimiter : str, optional
        Delimiter between group-lambda pairs, by default '_'
    
    Returns
    -------
    str
        Filename suffix like 'pos-100_vel-100_spike-1'
        
    Examples
    --------
    >>> groups = [GroupSpec('position', cols, '2D', lam=100.0),
    ...           GroupSpec('velocity', cols, '2D', lam=100.0)]
    >>> generate_lambda_suffix(groups)
    'position-100_velocity-100'
    """
    parts = []
    for g in groups:
        lam_str = _format_lambda(g.lam).replace('.', 'p')
        parts.append(f'{g.name}-{lam_str}')
    return delimiter.join(parts)


def _extract_lambda_config(groups):
    """Extract lambda configuration from groups for validation."""
    return {g.name: g.lam for g in groups}


def _validate_lambda_match(saved_lambdas, current_lambdas):
    """Check if lambda configurations match."""
    if saved_lambdas.keys() != current_lambdas.keys():
        return False, "Group names don't match"
    
    mismatches = []
    for name in saved_lambdas:
        if abs(saved_lambdas[name] - current_lambdas[name]) > 1e-10:
            mismatches.append(
                f"{name}: saved={saved_lambdas[name]:.6e}, current={current_lambdas[name]:.6e}"
            )
    
    if mismatches:
        return False, "Lambda values don't match:\n  " + "\n  ".join(mismatches)
    
    return True, "Lambda values match"


def _load_elimination_results(save_path, current_groups):
    """Load existing backward elimination results and validate lambda values."""
    save_path_obj = Path(save_path)
    if not save_path_obj.exists():
        return None
    
    print('='*80)
    print(f'Loading existing results from: {save_path}')
    
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
    
    # Validate lambda configuration
    if 'lambda_config' in saved_data:
        current_lambdas = _extract_lambda_config(current_groups)
        matches, msg = _validate_lambda_match(saved_data['lambda_config'], current_lambdas)
        
        if not matches:
            print('⚠ WARNING: Lambda mismatch detected!')
            print(f'⚠ {msg}')
            print('⚠ Ignoring cached results due to lambda mismatch.')
            print('='*80)
            return None
        else:
            print('✓ Lambda values validated - match saved configuration')
    else:
        print('⚠ Warning: No lambda information in saved file (old format)')
    
    if saved_data.get('completed', False):
        print('✓ Backward elimination already complete.')
        print(f'✓ Final model has {len(saved_data["kept_groups"])} groups')
        print(f'✓ Groups retained: {[g["name"] for g in saved_data["kept_groups"]]}')
        print('='*80)
        return saved_data
    
    print(f'⚠ Found partial results: {saved_data["current_step"]} steps completed')
    print(f'⚠ Current model has {len(saved_data["kept_groups"])} groups')
    print('⚠ Resuming elimination...')
    print('='*80)
    return saved_data


def _save_elimination_state(save_path_obj, kept, history, completed, current_step, 
                            save_metadata, initial_groups):
    """Save the current backward elimination state with lambda configuration."""
    # Convert GroupSpec objects to dicts for serialization
    kept_dicts = [
        {
            'name': g.name,
            'cols': g.cols,
            'vartype': g.vartype,
            'lam': g.lam,
        }
        for g in kept
    ]
    
    # Extract lambda configuration from initial groups for validation
    lambda_config = _extract_lambda_config(initial_groups)
    
    save_dict = {
        'kept_groups': kept_dicts,
        'history': history,
        'completed': completed,
        'current_step': current_step,
        'lambda_config': lambda_config,  # Save lambda values for validation
    }
    
    if save_metadata is not None:
        save_dict['metadata'] = save_metadata
    
    with open(save_path_obj, 'wb') as f:
        pickle.dump(save_dict, f)


def _print_elimination_header(groups, n_folds, alpha, ll_initial):
    """Print header for backward elimination."""
    print('='*80)
    print('BACKWARD ELIMINATION - GAM')
    print('='*80)
    print(f'Initial groups: {len(groups)}')
    print(f'CV folds: {n_folds}')
    print(f'Significance level (α): {alpha}')
    print(f'Initial model LL: {ll_initial:.6f}')
    print('\nPenalty (λ) configuration:')
    for g in groups:
        print(f'  {g.name}: λ={_format_lambda(g.lam)}')
    print('='*80)


def _print_elimination_step_header(step, n_kept):
    """Print header for elimination step."""
    print(f'\n{"="*80}')
    print(f'STEP {step}: Evaluating {n_kept} groups for removal')
    print(f'{"="*80}')


def _print_candidate_progress(idx, total, group_name, mean_delta, p_val, elapsed, avg_time):
    """Print progress for candidate evaluation."""
    progress = idx / total
    bar_length = 30
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f'  [{idx}/{total}] {bar} {100*progress:.0f}%')
    print(f'    Group: {group_name}')
    print(f'    ΔLL = {mean_delta:+.6f}, p = {p_val:.4f}')
    print(f'    Time: {elapsed:.1f}s (avg: {avg_time:.1f}s/candidate)')


def _print_elimination_decision(best, removed, kept_names):
    """Print decision after evaluating all candidates."""
    print('\n' + '─'*80)
    print('DECISION:')
    if removed:
        print(f'  ✓ REMOVED: {best["group"].name}')
        print(f'    ΔLL = {best["mean_delta"]:+.6f}, p = {best["p_value"]:.4f}')
        print(f'  → Remaining groups ({len(kept_names)}): {kept_names}')
    else:
        print(f'  ✗ STOP: {best["group"].name} is significant (p = {best["p_value"]:.4f})')
        print('  → All remaining groups are necessary')
        print(f'  → Final groups ({len(kept_names)}): {kept_names}')
    print('─'*80)


def _print_elimination_summary(kept, history, total_time):
    """Print final summary of backward elimination."""
    print('\n' + '='*80)
    print('BACKWARD ELIMINATION COMPLETE')
    print('='*80)
    print(f'✓ Total steps: {len(history)}')
    print(f'✓ Variables removed: {len(history)}')
    print(f'✓ Variables retained: {len(kept)}')
    print(f'✓ Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)')
    
    print(f'\n✓ FINAL MODEL ({len(kept)} groups):')
    for g in kept:
        print(f'    • {g.name}')
    
    if history:
        print('\n✓ ELIMINATION HISTORY:')
        for h in history:
            print(f'    Step {h["step"]}: Removed {h["removed"]} '
                  f'(ΔLL={h["delta_ll"]:+.4f}, p={h["p_value"]:.4f})')
    
    print('='*80)


def backward_elimination_gam(
    design_df,
    y,
    groups,
    *,
    alpha=0.05,
    n_folds=10,
    verbose=True,
    save_path: Optional[str] = None,
    load_if_exists: bool = True,
    save_metadata: Optional[Dict] = None,
):
    """
    Paper-faithful backward elimination for one neuron with incremental saving.

    Parameters
    ----------
    design_df : pd.DataFrame
        Design matrix
    y : np.ndarray
        Response variable
    groups : List[GroupSpec]
        Initial group specifications with penalty (lambda) values
    alpha : float, optional
        Significance level for retention, by default 0.05
    n_folds : int, optional
        Number of CV folds, by default 10
    verbose : bool, optional
        Print progress information, by default True
    save_path : Optional[str], optional
        Path to save elimination results. If provided, results are saved after
        each step, by default None
    load_if_exists : bool, optional
        If True and save_path exists, load and resume from saved state. 
        Lambda values are validated against saved configuration - if they don't
        match, cached results are ignored, by default True
    save_metadata : Optional[Dict], optional
        Additional metadata to save with results, by default None

    Returns
    -------
    kept_groups : list of GroupSpec
        Variables retained in the final model.
    history : list of dict
        Elimination steps and statistics.
        
    Notes
    -----
    Lambda (penalty) values from the input groups are saved with the results.
    When loading cached results, the function validates that lambda values match.
    This ensures that elimination results are only reused when penalty settings
    are identical.
    
    For convenience, use `generate_lambda_suffix(groups)` to create descriptive
    filenames that include lambda configuration, e.g.:
        suffix = generate_lambda_suffix(groups)
        save_path = f'results/elimination_{suffix}.pkl'
    
    Examples
    --------
    >>> # Basic usage
    >>> kept, history = backward_elimination_gam(
    ...     design_df, y, groups,
    ...     save_path='results/elim.pkl'
    ... )
    
    >>> # With lambda-based filename
    >>> suffix = generate_lambda_suffix(groups)
    >>> kept, history = backward_elimination_gam(
    ...     design_df, y, groups,
    ...     save_path=f'results/elim_{suffix}.pkl'
    ... )
    """
    import time
    start_time = time.time()
    
    # Save initial groups for lambda validation throughout
    initial_groups = groups.copy()
    
    # Try to load existing results
    if save_path is not None and load_if_exists:
        saved_data = _load_elimination_results(save_path, initial_groups)
        if saved_data is not None and saved_data.get('completed', False):
            # Reconstruct GroupSpec objects
            kept = [
                GroupSpec(
                    name=g['name'],
                    cols=g['cols'],
                    vartype=g['vartype'],
                    lam=g['lam'],
                )
                for g in saved_data['kept_groups']
            ]
            return kept, saved_data['history']
    
    # Initialize or resume
    kept = groups.copy()
    history = []
    step = 0
    
    # Prepare save path if needed
    save_path_obj = None
    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Compute baseline CV LL for full model
    ll_full = cross_validated_ll(
        design_df, y, kept, n_folds=n_folds
    )
    
    if verbose:
        _print_elimination_header(initial_groups, n_folds, alpha, ll_full.mean())

    improved = True

    while improved and len(kept) > 1:
        step += 1
        improved = False
        candidates = []
        
        if verbose:
            _print_elimination_step_header(step, len(kept))
        
        # Timing for candidate evaluation
        candidate_times = []

        for i, g in enumerate(kept, start=1):
            cand_start = time.time()
            
            reduced = [gg for gg in kept if gg.name != g.name]

            ll_reduced = cross_validated_ll(
                design_df, y, reduced, n_folds=n_folds
            )

            delta = ll_reduced - ll_full  # per fold
            mean_delta = delta.mean()

            # Paired test: is removal NOT harmful?
            t_stat, p_val = ttest_1samp(delta, 0.0)

            candidates.append({
                'group': g,
                'mean_delta': mean_delta,
                'p_value': p_val,
                'll_reduced': ll_reduced,
            })
            
            cand_time = time.time() - cand_start
            candidate_times.append(cand_time)

            if verbose:
                avg_time = np.mean(candidate_times)
                _print_candidate_progress(
                    i, len(kept), g.name, mean_delta, p_val, cand_time, avg_time
                )

        # Choose the variable whose removal hurts least
        candidates.sort(key=lambda x: x['mean_delta'], reverse=True)
        best = candidates[0]

        # Decision rule:
        # If removal does NOT significantly decrease LL → remove it
        will_remove = best['p_value'] > alpha and best['mean_delta'] >= 0
        
        if will_remove:
            removed_group = best['group']
            kept = [g for g in kept if g.name != removed_group.name]
            ll_full = best['ll_reduced']
            improved = True

            history.append({
                'step': step,
                'removed': removed_group.name,
                'delta_ll': best['mean_delta'],
                'p_value': best['p_value'],
            })
        
        if verbose:
            kept_names = [g.name for g in kept]
            _print_elimination_decision(best, will_remove, kept_names)
        
        # Save incrementally after each step
        if save_path_obj is not None:
            _save_elimination_state(
                save_path_obj, kept, history, 
                completed=not improved,  # Mark complete if we're stopping
                current_step=step,
                save_metadata=save_metadata,
                initial_groups=initial_groups
            )

    # Print final summary
    total_time = time.time() - start_time
    if verbose:
        _print_elimination_summary(kept, history, total_time)
    
    # Final save marking completion
    if save_path_obj is not None:
        _save_elimination_state(
            save_path_obj, kept, history,
            completed=True,
            current_step=step,
            save_metadata=save_metadata,
            initial_groups=initial_groups
        )
        if verbose:
            print(f'\n✓ Results saved to: {save_path}')

    return kept, history