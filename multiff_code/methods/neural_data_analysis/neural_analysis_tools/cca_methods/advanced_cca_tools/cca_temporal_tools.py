
"""
cca_temporal_tools.py
---------------------
Practical tools for temporally-aware CCA on neural (X) and behavioral (Y) data.

Includes:
- Ridge-regularized CCA (RCCA)
- Kernel CCA (KCCA) with linear/RBF kernels
- Spectral/temporal CCA via cross-spectral density matrices (Welch)
- Temporal lag reduction using raised-cosine bases

Author: ChatGPT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.linalg import svd
from scipy import signal
from scipy.linalg import cholesky, solve_triangular

# -----------------------------
# Utilities
# -----------------------------


def _center(X) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X)                 # <— force to NumPy
    mu = X.mean(axis=0, keepdims=True)
    return X - mu, mu


def _cov(X: np.ndarray) -> np.ndarray:
    """Empirical covariance of zero-mean X (n x d): (X^T X) / (n-1)."""
    X = np.asarray(X)
    n = X.shape[0]
    return (X.T @ X) / max(1, n - 1)


def _safe_cholesky(A: np.ndarray, jitter: float = 0) -> np.ndarray:
    """Cholesky with optional jitter on the diagonal."""
    A = np.asarray(A)
    d = A.shape[0]
    try:
        return cholesky(A, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return cholesky(A + jitter * np.eye(d), lower=True, check_finite=False)

# -----------------------------
# Ridge-regularized CCA (RCCA)
# -----------------------------


@dataclass
class RCCA:
    """Ridge-regularized CCA using SVD of whitened cross-covariance.

    Args:
        n_components: number of canonical pairs to return
        reg_x: L2 ridge regularization added to Cov(X)
        reg_y: L2 ridge regularization added to Cov(Y)
        scale: if True, z-score each variable (recommended if scales differ)
    """
    n_components: int = 2
    reg_x: float = 1e-3
    reg_y: float = 1e-3
    scale: bool = False

    # Fitted attributes
    x_mean_: Optional[np.ndarray] = None
    y_mean_: Optional[np.ndarray] = None
    chol_x_: Optional[np.ndarray] = None
    chol_y_: Optional[np.ndarray] = None
    Wx_: Optional[np.ndarray] = None  # weights in X-space (d_x x k)
    Wy_: Optional[np.ndarray] = None  # weights in Y-space (d_y x k)
    corrs_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RCCA":
        """Fit RCCA on (n x p) X and (n x q) Y."""
        Xc, self.x_mean_ = _center(X)
        Yc, self.y_mean_ = _center(Y)

        if self.scale:
            Xstd = Xc.std(axis=0, keepdims=True) + 1e-12
            Ystd = Yc.std(axis=0, keepdims=True) + 1e-12
            Xc /= Xstd
            Yc /= Ystd

        Sxx = _cov(Xc) + self.reg_x * np.eye(Xc.shape[1])
        Syy = _cov(Yc) + self.reg_y * np.eye(Yc.shape[1])
        Sxy = (Xc.T @ Yc) / max(1, Xc.shape[0] - 1)

        Lx = _safe_cholesky(Sxx, jitter=self.reg_x)
        Ly = _safe_cholesky(Syy, jitter=self.reg_y)

        # Whitened cross-covariance: R = Lx^{-1} Sxy Ly^{-T}
        R = solve_triangular(Lx, Sxy, lower=True, check_finite=False)
        R = solve_triangular(Ly.T, R.T, lower=False, check_finite=False).T

        U, s, Vt = svd(R, full_matrices=False)
        k = min(self.n_components, U.shape[1])

        # Back to data space: Wx = Lx^{-T} U, Wy = Ly^{-T} V
        Wx = solve_triangular(Lx.T, U[:, :k], lower=False, check_finite=False)
        Wy = solve_triangular(
            Ly.T, Vt[:k, :].T, lower=False, check_finite=False)

        self.Wx_, self.Wy_, self.corrs_ = Wx, Wy, s[:k]
        return self

    def transform(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Project X (and optionally Y) to canonical variates."""
        assert self.Wx_ is not None, "Call fit first."
        Xc = X - self.x_mean_
        U = Xc @ self.Wx_
        if Y is None or self.Wy_ is None:
            return U, None
        Yc = Y - self.y_mean_
        V = Yc @ self.Wy_
        return U, V

# -----------------------------
# Kernel CCA (KCCA)
# -----------------------------


def _linear_kernel(X, Y=None):
    Y = X if Y is None else Y
    return X @ Y.T


def _rbf_kernel(X, Y=None, gamma: float = None):
    Y = X if Y is None else Y
    if gamma is None:
        gamma = 1.0 / (X.shape[1] * (X.var() + 1e-12))
    Xn = np.sum(X**2, axis=1)[:, None]
    Yn = np.sum(Y**2, axis=1)[None, :]
    K = Xn + Yn - 2.0 * (X @ Y.T)
    return np.exp(-gamma * np.maximum(K, 0.0))


def _center_kernel(K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    K = np.asarray(K)
    n = K.shape[0]
    one = np.ones((n, n)) / n
    Kc = K - one @ K - K @ one + one @ K @ one
    return Kc, one  # return centering matrix info if needed


@dataclass
class KCCA:
    """Kernel CCA with centering and ridge regularization on Gram matrices.

    Args:
        n_components: number of canonical pairs
        kernel_x, kernel_y: 'linear' or 'rbf' or callables K(X, Y)
        gamma_x, gamma_y: RBF kernel width parameters
        reg_x, reg_y: ridge terms added to Gram matrices
    """
    n_components: int = 2
    kernel_x: Any = 'rbf'
    kernel_y: Any = 'rbf'
    gamma_x: Optional[float] = None
    gamma_y: Optional[float] = None
    reg_x: float = 1e-3
    reg_y: float = 1e-3

    # Fitted
    X_train_: Optional[np.ndarray] = None
    Y_train_: Optional[np.ndarray] = None
    Kx_c_: Optional[np.ndarray] = None
    Ky_c_: Optional[np.ndarray] = None
    chol_x_: Optional[np.ndarray] = None
    chol_y_: Optional[np.ndarray] = None
    Ax_: Optional[np.ndarray] = None  # dual weights (n x k)
    Ay_: Optional[np.ndarray] = None  # dual weights (n x k)
    corrs_: Optional[np.ndarray] = None
    H_: Optional[np.ndarray] = None  # centering matrix factor (ones/n)

    def _kernel(self, side: str, X, Y=None):
        k = self.kernel_x if side == 'x' else self.kernel_y
        gamma = self.gamma_x if side == 'x' else self.gamma_y
        if callable(k):
            return k(X, Y)
        if k == 'linear':
            return _linear_kernel(X, Y)
        elif k == 'rbf':
            return _rbf_kernel(X, Y, gamma=gamma)
        else:
            raise ValueError(f"Unknown kernel '{k}'")

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "KCCA":
        """Fit Kernel CCA on training X,Y (n x d)."""
        self.X_train_, self.Y_train_ = np.array(X), np.array(Y)

        Kx = self._kernel('x', self.X_train_)
        Ky = self._kernel('y', self.Y_train_)

        Kx_c, H = _center_kernel(Kx)
        Ky_c, _ = _center_kernel(Ky)

        self.Kx_c_, self.Ky_c_, self.H_ = Kx_c, Ky_c, H

        Cxx = Kx_c + self.reg_x * np.eye(Kx_c.shape[0])
        Cyy = Ky_c + self.reg_y * np.eye(Ky_c.shape[0])

        Lx = _safe_cholesky(Cxx, jitter=self.reg_x)
        Ly = _safe_cholesky(Cyy, jitter=self.reg_y)

        # Whitened cross-covariance in kernel space
        R = solve_triangular(Lx, Kx_c @ Ky_c, lower=True, check_finite=False)
        R = solve_triangular(Ly.T, R.T, lower=False, check_finite=False).T

        U, s, Vt = svd(R, full_matrices=False)
        k = min(self.n_components, U.shape[1])

        Ax = solve_triangular(Lx.T, U[:, :k], lower=False, check_finite=False)
        Ay = solve_triangular(
            Ly.T, Vt[:k, :].T, lower=False, check_finite=False)

        self.Ax_, self.Ay_, self.corrs_ = Ax, Ay, s[:k]
        return self

    def _center_against_train(self, K_new: np.ndarray) -> np.ndarray:
        """Center a cross-kernel K(X_new, X_train) using training centering."""
        # K_new: (n_new x n_train)
        n_tr = self.X_train_.shape[0]
        one_tr = np.ones((n_tr, n_tr)) / n_tr
        one_nt = np.ones((K_new.shape[0], n_tr)) / n_tr
        Kx_tr = self._kernel('x', self.X_train_)
        K_tr_c = Kx_tr - one_tr @ Kx_tr - Kx_tr @ one_tr + one_tr @ Kx_tr @ one_tr
        # Center new:
        Kc = K_new - one_nt @ Kx_tr - K_new @ one_tr + one_nt @ Kx_tr @ one_tr
        return Kc

    def transform(self, X: Optional[np.ndarray] = None, Y: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Project training or new data into kernel canonical variates.

        If X or Y is None, returns projections for the side(s) provided.
        For new data, uses K(new, train) and training centering.
        """
        assert self.Ax_ is not None, "Call fit first."
        U = V = None
        if X is not None:
            if X is self.X_train_:
                Kx_c = self.Kx_c_
            else:
                Kx = self._kernel('x', np.array(X), self.X_train_)
                n_tr = self.X_train_.shape[0]
                # Center new rows
                one_tr = np.ones((n_tr, n_tr)) / n_tr
                one_nt = np.ones((Kx.shape[0], n_tr)) / n_tr
                Kx_tr = self._kernel('x', self.X_train_)
                Kx_tr_c = Kx_tr - one_tr @ Kx_tr - Kx_tr @ one_tr + one_tr @ Kx_tr @ one_tr
                Kx_c = Kx - one_nt @ Kx_tr - Kx @ one_tr + one_nt @ Kx_tr @ one_tr
            U = Kx_c @ self.Ax_
        if Y is not None:
            if Y is self.Y_train_:
                Ky_c = self.Ky_c_
            else:
                Ky = self._kernel('y', np.array(Y), self.Y_train_)
                n_tr = self.Y_train_.shape[0]
                one_tr = np.ones((n_tr, n_tr)) / n_tr
                one_nt = np.ones((Ky.shape[0], n_tr)) / n_tr
                Ky_tr = self._kernel('y', self.Y_train_)
                Ky_tr_c = Ky_tr - one_tr @ Ky_tr - Ky_tr @ one_tr + one_tr @ Ky_tr @ one_tr
                Ky_c = Ky - one_nt @ Ky_tr - Ky @ one_tr + one_nt @ Ky_tr @ one_tr
            V = Ky_c @ self.Ay_
        return U, V

# ------------------------------------------
# Spectral / Temporal CCA via cross-spectra
# ------------------------------------------


@dataclass
class SpectralCCAResult:
    freqs: np.ndarray                # (F,)
    # (k, F) canonical coherence spectrum (k up to min(p,q))
    coh: np.ndarray
    # per-frequency weights for X (p x k), or None
    Ux: Optional[List[np.ndarray]]
    # per-frequency weights for Y (q x k), or None
    Uy: Optional[List[np.ndarray]]


def _csd_matrix(X: np.ndarray, fs: float, nperseg: int, noverlap: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cross-spectral density matrix for multivariate X (n x p).
    Returns freqs (F,) and S(f) (F x p x p), Hermitian PSDs at each frequency.
    """
    n, p = X.shape
    # Use Welch; build all pairs
    # We standardize each channel to zero-mean to avoid DC bias
    Xc = X - X.mean(axis=0, keepdims=True)
    # Compute frequencies using the first pair
    freqs, _ = signal.welch(
        Xc[:, 0], fs=fs, nperseg=nperseg, noverlap=noverlap, return_onesided=True)
    F = len(freqs)
    S = np.zeros((F, p, p), dtype=np.complex128)
    for i in range(p):
        fi, Pii = signal.welch(
            Xc[:, i], fs=fs, nperseg=nperseg, noverlap=noverlap, return_onesided=True)
        S[:, i, i] = Pii
        for j in range(i+1, p):
            _, Pij = signal.csd(
                Xc[:, i], Xc[:, j], fs=fs, nperseg=nperseg, noverlap=noverlap, return_onesided=True)
            S[:, i, j] = Pij
            S[:, j, i] = np.conjugate(Pij)
    return freqs, S


def spectral_cca(
    X: np.ndarray, Y: np.ndarray, *, fs: float = 1.0, nperseg: int = 256, noverlap: Optional[int] = None,
    reg_x: float = 1e-6, reg_y: float = 1e-6, n_components: Optional[int] = None, return_weights: bool = False
) -> SpectralCCAResult:
    """Compute canonical coherence spectrum between multivariate X (n x p) and Y (n x q).

    For each frequency f, we solve the CCA generalized eigenproblem on cross-spectral matrices:
        Cxx(f) = S_xx(f) + reg_x I
        Cyy(f) = S_yy(f) + reg_y I
        Cxy(f) = S_xy(f)
    Then SVD of the whitened cross-spectral matrix yields canonical coherences at f.

    Returns canonical coherence (values between 0 and 1) per frequency.
    Optionally returns per-frequency weight matrices (complex).
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."
    n, p = X.shape
    _, q = Y.shape
    kmax = min(p, q) if n_components is None else min(n_components, min(p, q))

    freqs, Sxx = _csd_matrix(X, fs, nperseg, noverlap)
    freqs2, Syy = _csd_matrix(Y, fs, nperseg, noverlap)
    assert np.allclose(freqs, freqs2), "Frequency grids do not match."

    # Cross spectra between each Xi and Yj
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    F = len(freqs)
    Sxy = np.zeros((F, p, q), dtype=np.complex128)
    for i in range(p):
        for j in range(q):
            _, Pxy = signal.csd(
                Xc[:, i], Yc[:, j], fs=fs, nperseg=nperseg, noverlap=noverlap, return_onesided=True)
            Sxy[:, i, j] = Pxy

    coh = np.zeros((kmax, F))
    Ux = [] if return_weights else None
    Uy = [] if return_weights else None

    eye_p = np.eye(p)
    eye_q = np.eye(q)

    for f in range(F):
        Cxx = Sxx[f] + reg_x * eye_p
        Cyy = Syy[f] + reg_y * eye_q
        Cxy = Sxy[f]

        # Whiten with Cholesky
        Lx = _safe_cholesky(Cxx, jitter=reg_x)
        Ly = _safe_cholesky(Cyy, jitter=reg_y)

        # R = Lx^{-1} Cxy Ly^{-H}
        R = solve_triangular(Lx, Cxy, lower=True, check_finite=False)
        R = solve_triangular(Ly.conj().T, R.conj().T,
                             lower=False, check_finite=False).conj().T

        U, s, Vh = svd(R, full_matrices=False)
        r = min(kmax, len(s))
        # canonical coherence (singular values)
        coh[:r, f] = np.clip(s[:r].real, 0, 1)

        if return_weights:
            Wx = solve_triangular(
                Lx.T.conj(), U[:, :r], lower=False, check_finite=False)
            Wy = solve_triangular(
                Ly.T.conj(), Vh[:r, :].T, lower=False, check_finite=False)
            Ux.append(Wx)  # p x r
            Uy.append(Wy)  # q x r

    return SpectralCCAResult(freqs=freqs, coh=coh, Ux=Ux, Uy=Uy)

# ------------------------------------------
# Reduced bases for lags (raised cosines)
# ------------------------------------------


def raised_log_cosine_basis(n_basis: int, t_max: float, dt: float, *, t_min: float = 0.0,
                            log_spaced: bool = True, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Causal raised-cosine basis that tiles [t_min, t_max].
    Returns lags (L,), B (L x K) with unit-area columns (sum * dt = 1).
    """
    lags = np.arange(0.0, t_max + 1e-12, dt)
    K = int(n_basis)

    def warp(x):
        return np.log(x + eps) if log_spaced else x

    W = warp(lags)
    W_min, W_max = warp(t_min), warp(t_max + 1e-12)
    centers = np.linspace(W_min, W_max, K)
    widths = (centers[1] - centers[0]) if K > 1 else (W_max - W_min + 1e-6)

    B = []
    for c in centers:
        arg = (W - c) * np.pi / (2 * widths)
        phi = np.cos(np.clip(arg, -np.pi, np.pi))
        phi = np.where(np.abs(arg) <= np.pi, (phi + 1) /
                       2.0, 0.0)  # raised cosine [0,1]
        B.append(phi)
    B = np.stack(B, axis=1)  # (L x K)
    # Unit area
    B /= (B.sum(axis=0, keepdims=True) + 1e-12)
    return lags, B


def embed_with_basis(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Convolve each channel of X (n x p) with each basis in B (L x K), causal.
    Output has shape (n x (p*K)), with 'same' alignment (current time uses past lags)."""
    n, p = X.shape
    L, K = B.shape
    out = np.zeros((n, p * K), dtype=X.dtype)
    for j in range(p):
        x = X[:, j]
        for k in range(K):
            # Causal FIR: y[t] = sum_{l=0..L-1} B[l,k] * x[t-l]
            # reverse for lfilter convention
            y = signal.lfilter(B[::-1, k], [1.0], x)
            out[:, j*K + k] = y
    return out

# -----------------------------
# Demo
# -----------------------------


def _demo():
    rng = np.random.default_rng(0)
    n = 3000
    p, q = 6, 4
    # Generate latent drivers with lead-lag structure
    t = np.arange(n)
    z = signal.lfilter(
        [1], [1, -0.95], rng.standard_normal((n, 2)))  # AR(1) latent

    X = z @ rng.normal(size=(2, p)) + 0.2 * rng.standard_normal((n, p))
    Y = np.roll(z, 3, axis=0) @ rng.normal(size=(2, q)) + 0.2 * \
        rng.standard_normal((n, q))  # Y lags X by ~3

    # RCCA on raw signals
    rcca = RCCA(n_components=2, reg_x=1e-2, reg_y=1e-2).fit(X, Y)
    U, V = rcca.transform(X, Y)
    print("[RCCA] corr #1 ~", np.corrcoef(U[:, 0], V[:, 0])[0, 1])

    # Basis-embedded RCCA
    lags, B = raised_log_cosine_basis(
        n_basis=6, t_max=10, dt=1.0, log_spaced=True)
    Xb = embed_with_basis(X, B)
    Yb = embed_with_basis(Y, B)
    rcca_b = RCCA(n_components=2, reg_x=1e-2, reg_y=1e-2).fit(Xb, Yb)
    Ub, Vb = rcca_b.transform(Xb, Yb)
    print("[RCCA + basis] corr #1 ~", np.corrcoef(Ub[:, 0], Vb[:, 0])[0, 1])

    # KCCA (RBF)
    kcca = KCCA(n_components=2, kernel_x='rbf', kernel_y='rbf',
                reg_x=1e-2, reg_y=1e-2).fit(X, Y)
    Uk, Vk = kcca.transform(X, Y)
    print("[KCCA-RBF] corr #1 ~",
          np.corrcoef(np.real(Uk[:, 0]), np.real(Vk[:, 0]))[0, 1])

    # Spectral CCA: canonical coherence spectrum
    spec = spectral_cca(X, Y, fs=1.0, nperseg=512, noverlap=256,
                        reg_x=1e-3, reg_y=1e-3, n_components=1)
    # Report peak frequency
    f_peak = spec.freqs[np.argmax(spec.coh[0])]
    print(
        f"[Spectral CCA] Peak canonical coherence at f={f_peak:.3f} cycles/bin, value={spec.coh[0].max():.3f}")


if __name__ == "__main__":
    _demo()
