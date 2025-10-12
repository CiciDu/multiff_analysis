import os
import numpy as np
import math
from math import pi
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def theta_half_and_delta_abs(theta_center: float, theta_boundary: float):
    import math

    def wrap(a):
        return (a + math.pi) % (2*math.pi) - math.pi
    theta_c = wrap(theta_center)
    theta_b = wrap(theta_boundary)
    options = [
        (theta_c, theta_b),
        (theta_c + 2*math.pi, theta_b),
        (theta_c - 2*math.pi, theta_b),
    ]
    th, tb = min(options, key=lambda ab: abs(ab[0]-ab[1]))
    theta_half = 0.5*(th + tb)
    delta_abs = abs(th - tb)
    return wrap(theta_half), min(delta_abs, math.pi)


def make_ff_flash_from_random_sampling(
    num_alive_ff: int,
    duration: float,
    non_flashing_interval_mean: float = 3.0,
    flash_on_interval: float = 0.3,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    *,
    chunk: int = 16,  # number of intervals generated per active FF per iteration
):
    '''
    Fast NumPy version (chunked & vectorized).

    Each firefly alternates: Poisson(off) then fixed on=flash_on_interval,
    starting at t=-10. Intervals overlapping [0, duration] are kept and clipped.
    If a firefly has no overlapping interval, add a fallback [0, min(on, duration)].

    Args
    ----
    num_alive_ff : int
    duration : float
    non_flashing_interval_mean : float
    flash_on_interval : float
    rng : np.random.Generator | None
    seed : int | None
    chunk : int
        How many intervals to simulate per active firefly per loop iteration.
        Larger = fewer Python iterations (usually faster), but more temporary memory.

    Returns
    -------
    ff_flash : list[np.ndarray]  # each (Mi, 2) float32 array of [start, end]
    '''
    if rng is None:
        rng = np.random.default_rng(seed)

    N = int(num_alive_ff)
    duration = float(duration)
    on = float(flash_on_interval)

    if N <= 0 or duration <= 0.0:
        return [np.zeros((0, 2), dtype=np.float32) for _ in range(max(N, 0))]

    # time cursors per firefly; warmup so initial state at t≈0 feels natural
    t = np.full(N, -10.0, dtype=np.float32)

    # per-FF ragged collectors
    starts_lists = [[] for _ in range(N)]
    ends_lists = [[] for _ in range(N)]

    # precompute k*on for a chunk [0..chunk-1]
    k = np.arange(chunk, dtype=np.float32)
    k_on = k * on

    while True:
        active = t < duration
        if not np.any(active):
            break

        idx = np.nonzero(active)[0]
        n_active = idx.size

        # Sample Poisson gaps in a (n_active, chunk) block
        gaps = rng.poisson(non_flashing_interval_mean, size=(
            n_active, chunk)).astype(np.float32)

        # Cumulative sums along chunk-axis
        csum_gaps = np.cumsum(gaps, axis=1)  # (n_active, chunk)

        # start_k = t0 + csum_gaps_k + k*on
        # end_k   = start_k + on
        t0 = t[idx][:, None]                               # (n_active, 1)
        starts = t0 + csum_gaps + k_on[None, :]           # (n_active, chunk)
        ends = starts + on

        # Keep only intervals overlapping [0, duration]
        # (end > 0) and (start < duration) — clip to horizon
        valid = (ends > 0.0) & (starts < duration)

        if np.any(valid):
            # Clip in vectorized form
            starts_clipped = np.maximum(
                starts, 0.0, where=valid, out=np.zeros_like(starts))
            ends_clipped = np.minimum(
                ends,   duration, where=valid, out=np.zeros_like(ends))

            # Append per-FF (one small Python loop over active FFs)
            for row, i in enumerate(idx):
                m = valid[row]
                if m.any():
                    s_row = starts_clipped[row, m]
                    e_row = ends_clipped[row, m]
                    # extend with small batches (fast)
                    starts_lists[i].extend(s_row.tolist())
                    ends_lists[i].extend(e_row.tolist())

        # Advance time cursors by the final end in this chunk:
        # t_final = t0 + sum(gaps) + chunk*on  (i.e., end of the last interval in the chunk)
        t[idx] = (t0[:, 0] + csum_gaps[:, -1] + chunk * on).astype(np.float32)

    # Finalize ragged outputs with fallback if empty
    ff_flash = []
    fallback_end = float(min(on, duration))
    for s_list, e_list in zip(starts_lists, ends_lists):
        if not s_list:
            s_list, e_list = [0.0], [fallback_end]
        arr = np.column_stack((np.asarray(s_list, dtype=np.float32),
                               np.asarray(e_list, dtype=np.float32)))
        ff_flash.append(arr)
    return ff_flash


def calculate_angles_to_ff(ffxy, agentx, agenty, agentheading, ff_radius, ffdistance=None):
    """
    NumPy version. Returns (angle_to_center, angle_to_boundary), each ∈ (-pi, pi].
    ffxy: array shape (N,2)
    agentx, agenty, agentheading: shape (1,) or scalars
    """
    ffxy = np.asarray(ffxy, dtype=float)
    agentx = float(np.asarray(agentx).reshape(-1)[0])
    agenty = float(np.asarray(agenty).reshape(-1)[0])
    agentheading = float(np.asarray(agentheading).reshape(-1)[0])

    if ffdistance is None:
        dx = ffxy[:, 0] - agentx
        dy = ffxy[:, 1] - agenty
        ffdistance = np.sqrt(dx*dx + dy*dy)
    else:
        ffdistance = np.asarray(ffdistance, dtype=float)

    # angle to centers, then wrap to (-pi, pi]
    angle_to_center = np.arctan2(
        ffxy[:, 1] - agenty, ffxy[:, 0] - agentx) - agentheading
    angle_to_center = np.remainder(angle_to_center, 2*np.pi)
    angle_to_center[angle_to_center > np.pi] -= 2*np.pi

    # boundary-adjusted angle (smallest angle to reward boundary)
    side_opposite = float(ff_radius)
    hyp = np.clip(ffdistance, side_opposite, None)
    theta = np.arcsin(side_opposite / hyp)  # ≥ 0
    angle_to_boundary_abs = np.clip(np.abs(angle_to_center) - theta, 0.0, None)
    angle_to_boundary = np.sign(angle_to_center) * angle_to_boundary_abs

    return angle_to_center, angle_to_boundary


def update_noisy_ffxy(ffx_noisy, ffy_noisy, ffx, ffy, ff_uncertainty_all, visible_ff_indices):
    """
    Adding noise to the positions of the fireflies based on how long ago they were seen and 
    meanwhile restoring the accurate positions of the currently visible fireflies

    Parameters
    ----------
    ffx_noisy: np.ndarray
        containing the x-coordinates of all fireflies with noise
    ffy_noisy: np.ndarray
        containing the y-coordinates of all fireflies with noise
    ffx: np.ndarray
        containing the accurate x-coordinates of all fireflies
    ffy: np.ndarray
        containing the accurate y-coordinates of all fireflies
    ff_uncertainty_all: np.ndarray
        containing the values of uncertainty of all fireflies; scaling is based on a parameter for the environment
    visible_ff_indices: np.ndarray
        containing the indices of the visible fireflies


    Returns
    -------
    ffx_noisy: np.ndarray
        containing the x-coordinates of all fireflies with noise
    ffy_noisy: np.ndarray
        containing the y-coordinates of all fireflies with noise
    ffxy_noisy: np.ndarray
        containing the x-coordinates and the y-coordinates of all fireflies with noise

    """

    # update the positions of fireflies with uncertainties added; note that uncertainties are cummulative across steps
    num_alive_ff = len(ff_uncertainty_all)
    ffx_noisy = ffx_noisy + \
        np.random.normal(0, ff_uncertainty_all)
    ffy_noisy = ffy_noisy + \
        np.random.normal(0, ff_uncertainty_all)
    # for the visible fireflies, their positions are updated to be the real positions
    ffx_noisy[visible_ff_indices] = ffx[visible_ff_indices].copy()
    ffy_noisy[visible_ff_indices] = ffy[visible_ff_indices].copy()
    ffxy_noisy = np.stack((ffx_noisy, ffy_noisy), axis=1)
    return ffx_noisy, ffy_noisy, ffxy_noisy


def find_visible_ff(time, ff_distance_all, ff_angle_all,
                    invisible_distance, invisible_angle, ff_flash):
    """
    Returns indices (relative to inputs) of fireflies visible at `time`.
    Uses NumPy arrays (not torch).
    """
    ff_distance_all = np.asarray(ff_distance_all)
    ff_angle_all = np.asarray(ff_angle_all)

    # distance + angle gates
    visible_ff = np.logical_and(
        ff_distance_all < invisible_distance,
        np.abs(ff_angle_all) < invisible_angle
    )

    if ff_flash is not None:
        # Among currently geometrically visible, keep only those flashing now
        idx = np.where(visible_ff)[0]
        for i in idx:
            intervals = np.asarray(ff_flash[i])
            # Expect shape (M, 2) with [start, end]
            on_now = np.any((intervals[:, 0] <= time)
                            & (intervals[:, 1] >= time))
            if not on_now:
                visible_ff[i] = False

    return np.where(visible_ff)[0]



def _get_placeholder_ff(add_memory, add_ff_t_since_start_seen, invisible_distance):
    if add_memory | add_ff_t_since_start_seen:
        return np.array([[0.], [0.], [invisible_distance], [0.]])
    else:
        return np.array([[0.], [0.], [invisible_distance]])


def _get_topk_indices(ff_indices, ff_distance_all, num_obs_ff):
    # Get the top-k indices with smallest distances (largest negative distances)
    topk_ff = np.argsort(-ff_distance_all[ff_indices])[:num_obs_ff]
    return ff_indices[topk_ff]


def _get_sorted_indices(ff_indices, ff_distance_all):
    sorted_indices = np.argsort(-ff_distance_all[ff_indices])
    return ff_indices[sorted_indices]
