import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import random

# ---------- helpers ----------
def _set_equal_aspect_3d(ax, X, Y, Z):
    xr = np.nanmax(X) - np.nanmin(X)
    yr = np.nanmax(Y) - np.nanmin(Y)
    zr = np.nanmax(Z) - np.nanmin(Z)
    r  = max(xr, yr, zr)
    xm = (np.nanmax(X) + np.nanmin(X)) / 2
    ym = (np.nanmax(Y) + np.nanmin(Y)) / 2
    zm = (np.nanmax(Z) + np.nanmin(Z)) / 2
    ax.set_xlim(xm - r/2, xm + r/2)
    ax.set_ylim(ym - r/2, ym + r/2)
    ax.set_zlim(zm - r/2, zm + r/2)
    try:
        ax.set_box_aspect([1,1,1])
    except Exception:
        pass

def _segments_and_colors_from_pts(pts, t_norm):
    """Build contiguous 3D segments and per-segment color values, skipping NaNs."""
    if pts.shape[0] < 2:
        return None, None
    finite = np.isfinite(pts).all(axis=1)
    mask = finite[:-1] & finite[1:]
    if not np.any(mask):
        return None, None
    segs = np.stack([pts[:-1][mask], pts[1:][mask]], axis=1)
    cols = t_norm[:-1][mask]
    return segs, cols

def _project_variance_for_view(Xyz, elev_deg, azim_deg):
    """Total 2D variance of points when viewed from (elev, azim)."""
    er = np.deg2rad(elev_deg)
    ar = np.deg2rad(azim_deg)
    # viewing direction (unit)
    v = np.array([np.cos(er)*np.cos(ar), np.cos(er)*np.sin(ar), np.sin(er)])
    # screen basis (u1,u2), orthonormal and perpendicular to v
    ref = np.array([0.0, 0.0, 1.0]) if abs(v[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
    u1 = np.cross(v, ref); u1 /= (np.linalg.norm(u1) + 1e-12)
    u2 = np.cross(v, u1);  u2 /= (np.linalg.norm(u2) + 1e-12)
    x = Xyz @ u1
    y = Xyz @ u2
    return np.nanvar(x) + np.nanvar(y)

def _best_view_grid(Xyz, elev_range=(5,85), azim_range=(-180,180), step=5):
    best, be, ba = -np.inf, 25, -60
    for elev in range(elev_range[0], elev_range[1]+1, step):
        for azim in range(azim_range[0], azim_range[1], step):
            val = _project_variance_for_view(Xyz, elev, azim)
            if val > best:
                best, be, ba = val, elev, azim
    return be, ba

def _pca_rotate_points(Xyz):
    """Center and rotate points so columns align with PC1/PC2/PC3. Returns (XcR, mean, R)."""
    mu = np.nanmean(Xyz, axis=0, keepdims=True)
    Xc = Xyz - mu
    # SVD on centered data (NaNs treated as 0 in SVD input)
    U, S, Vt = np.linalg.svd(np.nan_to_num(Xc, nan=0.0), full_matrices=False)
    R = Vt.T  # columns are PCs
    return Xc @ R, mu, R

def _apply_rotation(tr, mu, R):
    """Rotate a (3,T) trajectory to PCA frame using (mu, R)."""
    T = tr.shape[1]
    return ((tr.T - mu) @ R).T  # (3,T)

# ---------- main ----------
def plot_gpfa_traj_3d_timecolored_average(
    trajectories,
    linewidth_single_trial=0.6,
    color_single_trial='C0',
    alpha_single_trial=0.12,
    linewidth_trial_average=3.2,
    cmap_average='viridis',
    show_colorbar=True,
    start_end_markers=True,
    start_marker='o',
    end_marker='X',
    marker_size=36,
    view_azim=-60,
    view_elev=25,
    title='Latent dynamics extracted by GPFA',
    # Presentation controls
    max_single_trials=60,     # max single trials drawn
    single_downsample=2,      # draw every Nth sample for singles
    avg_downsample=1,         # draw every Nth sample for average
    seed=0,
    show_direction_arrow=True,
    # Auto camera / rotation
    auto_view=None,           # None | "pca" | "grid"
    grid_step=5               # step (deg) for "grid" search
):
    """
    trajectories: iterable of arrays shaped (D, T). If D>3, the first 3 dims are used.
    Average is computed with NaN-padding (no truncation).
    auto_view:
        - None: use given view_elev/view_azim
        - "pca": rotate data to PC1/PC2/PC3 and use a nice default view
        - "grid": keep data as-is; search azim/elev maximizing on-screen variance
    """
    # ----- sanitize / keep first 3 dimensions -----
    trajs = []
    for tr in trajectories:
        arr = np.asarray(tr, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"Each trajectory must be 2D (dims x time); got {arr.shape}.")
        if arr.shape[0] < 3:
            raise ValueError(f"Trajectory must have at least 3 dims; got {arr.shape[0]}.")
        trajs.append(arr[:3])  # keep first 3 dims
    if not trajs:
        raise ValueError("No trajectories provided.")

    # ----- (optional) rotate data to PCA frame before plotting -----
    axis_labels = ('Dim 1', 'Dim 2', 'Dim 3')
    if auto_view == "pca":
        # build a big point cloud from all trials (downsampled for speed) to estimate PCs
        cloud = []
        for tr in trajs:
            cloud.append(tr[:, ::max(1, single_downsample)].T)  # (T,3)
        X = np.concatenate(cloud, axis=0)
        Xp, mu, R = _pca_rotate_points(X)  # compute rotation
        # rotate each trajectory and labels
        trajs = [_apply_rotation(tr, mu.squeeze(0), R) for tr in trajs]
        axis_labels = ('PC1', 'PC2', 'PC3')

    # ----- choose subset of single trials for clarity -----
    rng = random.Random(seed)
    if len(trajs) > max_single_trials:
        idxs = rng.sample(range(len(trajs)), max_single_trials)
        plot_trajs = [trajs[i] for i in idxs]
    else:
        plot_trajs = trajs

    # ----- NaN-padding for averaging (no truncation) -----
    max_T = max(tr.shape[1] for tr in trajs)
    padded = np.full((len(trajs), 3, max_T), np.nan)
    for i, tr in enumerate(trajs):
        T = tr.shape[1]
        padded[i, :, :T] = tr
    avg = np.nanmean(padded, axis=0)[:, ::max(1, avg_downsample)]  # (3, T')
    pts = avg.T  # (T', 3)

    # ----- figure / styling -----
    fig = plt.figure(figsize=(6.0, 4.8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, pad=10, fontsize=12)
    ax.set_xlabel(axis_labels[0], labelpad=6)
    ax.set_ylabel(axis_labels[1], labelpad=6)
    ax.set_zlabel(axis_labels[2], labelpad=6)
    # declutter panes/grid
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]['linestyle'] = (0,(1,4))
        axis._axinfo["grid"]['linewidth'] = 0.5
        axis._axinfo["grid"]['color'] = (0,0,0,0.15)
    ax.xaxis.pane.set_edgecolor((0,0,0,0.12))
    ax.yaxis.pane.set_edgecolor((0,0,0,0.12))
    ax.zaxis.pane.set_edgecolor((0,0,0,0.12))
    ax.xaxis.pane.set_facecolor((1,1,1,0))
    ax.yaxis.pane.set_facecolor((1,1,1,0))
    ax.zaxis.pane.set_facecolor((1,1,1,0))

    # ----- draw single trials (downsampled) -----
    for tr in plot_trajs:
        tr_ds = tr[:, ::max(1, single_downsample)]
        ax.plot(tr_ds[0], tr_ds[1], tr_ds[2],
                lw=linewidth_single_trial, c=color_single_trial, alpha=alpha_single_trial)

    # ----- colored average line -----
    t_norm = np.linspace(0.0, 1.0, max(pts.shape[0], 2))
    segs, cols = _segments_and_colors_from_pts(pts, t_norm)
    lc = None
    if segs is not None:
        lc = Line3DCollection(segs, cmap=cmap_average, norm=Normalize(0.0, 1.0))
        lc.set_array(cols)
        lc.set_linewidth(linewidth_trial_average)
        ax.add_collection3d(lc)

    # start/end markers
    if start_end_markers and pts.shape[0] >= 1 and np.all(np.isfinite(pts[0])):
        ax.scatter(*pts[0],  s=marker_size, marker=start_marker, label='start', zorder=5)
    if start_end_markers and pts.shape[0] >= 1 and np.all(np.isfinite(pts[-1])):
        ax.scatter(*pts[-1], s=marker_size, marker=end_marker,  label='end',   zorder=5)

    # small direction arrow
    if show_direction_arrow and pts.shape[0] >= 3:
        i = max(1, pts.shape[0]//3)
        j = min(i+1, pts.shape[0]-1)
        if np.all(np.isfinite(pts[[i,j]])):
            dx, dy, dz = (pts[j] - pts[i])
            ax.quiver(pts[i,0], pts[i,1], pts[i,2], dx, dy, dz,
                      length=1.0, normalize=True, linewidth=1.2, arrow_length_ratio=0.15)

    # ----- auto camera selection -----
    # Use *all* trials (lightly downsampled) + average for the search/rotation
    cloud_for_view = []
    for tr in trajs:
        cloud_for_view.append(tr[:, ::max(1, single_downsample)].T)
    cloud_for_view.append(pts)
    all_xyz = np.concatenate(cloud_for_view, axis=0)

    # if auto_view == "grid":
    #     view_elev, view_azim = _best_view_grid(all_xyz, step=int(grid_step))
    # elif auto_view == "pca":
    #     # already rotated to PCs; a generic pleasing angle
    #     view_elev, view_azim = 25, -60

    # # ----- limits & aspect, colorbar, legend, camera -----
    # ax.view_init(elev=view_elev, azim=view_azim)
    
    ax.view_init(elev=-5, azim=60)

    _set_equal_aspect_3d(ax, all_xyz[:,0], all_xyz[:,1], all_xyz[:,2])

    if show_colorbar and lc is not None:
        cb = fig.colorbar(lc, ax=ax, pad=0.02, shrink=0.75)
        cb.set_label('time (normalized)')

    if start_end_markers:
        ax.legend(frameon=False, fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.show()
    return fig, ax
