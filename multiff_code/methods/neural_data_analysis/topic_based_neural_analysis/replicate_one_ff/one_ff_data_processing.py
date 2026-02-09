"""
one_ff_data_processing.py

Navigation covariates for One-FF task.
Faithful Python replication of AnalyseBehaviour.m outputs
used downstream in AnalysePopulation.m.

Implements:
V, W, D, Phi, R_targ, Theta_targ, Eye_ver, Eye_hor, move

All angular quantities are in DEGREES (matches MATLAB).

Author: you
"""

import numpy as np


# =========================
# Core integrals
# =========================

def compute_D(v, ts, dt):
    """
    Integrated distance:
    D(t) = ∫ v(t) dt for ts > 0
    """
    D = np.zeros_like(v)
    mask = ts > 0
    D[mask] = np.cumsum(v[mask]) * dt
    return D


def compute_Phi(w, ts, dt):
    """
    Integrated heading angle (degrees):
    Phi(t) = ∫ w(t) dt for ts > 0

    w is assumed to be in deg/s
    """
    return np.cumsum(w * (ts > 0)) * dt


# =========================
# Coordinate rotation
# =========================
def rotate_xy(x, y, phi_deg):
    """
    Rotate world-frame (x, y) into egocentric frame
    using heading angle phi in DEGREES.

    EXACT match to MATLAB:
    R(phi) = [cosd(phi) -sind(phi);
              sind(phi)  cosd(phi)]
    """
    phi_rad = np.deg2rad(phi_deg)

    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)

    x_rot = cos_phi * x - sin_phi * y
    y_rot = sin_phi * x + cos_phi * y
    return x_rot, y_rot



# =========================
# Target-relative variables
# =========================

def compute_target_relative(tr, dt):
    """
    Compute R_targ and Theta_targ exactly as in AnalyseBehaviour.m

    Key points:
    - target position estimated ONCE per trial (median during movement)
    - rotated into egocentric frame using Phi (degrees)
    - Theta_targ returned in DEGREES
    - NaN after stopping
    """
    ts = tr.continuous.ts
    w = tr.continuous.w

    # Integrated heading (degrees)
    Phi = compute_Phi(w, ts, dt)

    # Indices for movement window
    indx_beg = np.searchsorted(ts, tr.events.t_targ)
    indx_stop = np.searchsorted(ts, tr.events.t_stop)

    if indx_beg >= indx_stop:
        print('Empty movement window:', indx_beg, indx_stop)
        
    
    # Trial-level target position (median during movement)
    x_fly = np.nanmedian(tr.continuous.xfp[indx_beg:indx_stop])
    y_fly = np.nanmedian(tr.continuous.yfp[indx_beg:indx_stop])

    # Relative position (world frame)
    dx = x_fly - tr.continuous.xmp
    dy = y_fly - tr.continuous.ymp

    # Rotate into egocentric frame
    dx_r, dy_r = rotate_xy(dx, dy, Phi)

    # Radial distance
    R_targ = np.sqrt(dx_r**2 + dy_r**2)

    # NOTE: MATLAB uses atan2d(x, y)
    Theta_targ = np.rad2deg(np.arctan2(dx_r, dy_r))

    # NaN after stopping
    R_targ[indx_stop + 1:] = np.nan
    Theta_targ[indx_stop + 1:] = np.nan

    return R_targ, Theta_targ


# =========================
# Eye variables (as used in GAM)
# =========================

def compute_eye_ver(tr):
    """
    Vertical eye signal used in GAM.
    MATLAB logic:
      - if one eye is entirely NaN, use the other
      - else average
    """
    zle = tr.continuous.zle
    zre = tr.continuous.zre

    if np.all(np.isnan(zle)):
        return zre
    elif np.all(np.isnan(zre)):
        return zle
    else:
        return 0.5 * (zle + zre)


def compute_eye_hor(tr):
    """
    Horizontal eye signal used in GAM.
    MATLAB logic:
      - if one eye is entirely NaN, use the other
      - else average
    """
    yle = tr.continuous.yle
    yre = tr.continuous.yre

    if np.all(np.isnan(yle)):
        return yre
    elif np.all(np.isnan(yre)):
        return yle
    else:
        return 0.5 * (yle + yre)


# =========================
# Movement state
# =========================

def compute_move(tr):
    """
    Binary movement indicator.
    1 between t_targ and t_stop, else 0.
    """
    ts = tr.continuous.ts
    move = np.zeros_like(ts)

    t0 = tr.events.t_targ
    t1 = tr.events.t_stop

    move[(ts >= t0) & (ts <= t1)] = 1.0
    return move


# =========================
# Master function
# =========================

def compute_all_covariates(tr, dt):
    """
    Compute all navigation covariates for one trial.

    Returns dict with keys:
    V, W, D, Phi, R_targ, Theta_targ, Eye_ver, Eye_hor, move
    """
    ts = tr.continuous.ts

    V = tr.continuous.v
    W = tr.continuous.w

    D = compute_D(V, ts, dt)
    Phi = compute_Phi(W, ts, dt)
    R_targ, Theta_targ = compute_target_relative(tr, dt)
    Eye_ver = compute_eye_ver(tr)
    Eye_hor = compute_eye_hor(tr)
    move = compute_move(tr)

    return {
        'v': V,
        'w': W,
        'd': D,
        'phi': Phi,
        'r_targ': R_targ,
        'theta_targ': Theta_targ,
        'eye_ver': Eye_ver,
        'eye_hor': Eye_hor,
        'move': move
    }
