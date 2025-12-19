# ============================================================
# parent_assembler_mpl.py  — Matplotlib version
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------
# Helpers to extract grid structure from a Matplotlib figure
# ---------------------------------------------------------------------

def _get_axes_grid(fig):
    """
    Returns a 2D list of axes.
    This works because streamline_making_matplotlib_plot stores:
        fig._axes_grid = axes
    """
    if hasattr(fig, '_axes_grid'):
        return fig._axes_grid
    else:
        # fallback: treat it as 1×1
        return np.array([[fig.axes[0]]], dtype=object)


def _grid_shape(fig):
    axes_grid = _get_axes_grid(fig)
    return axes_grid.shape  # (rows, cols)


# ---------------------------------------------------------------------
# Parent assembly logic
# ---------------------------------------------------------------------

def _adaptive_vspace(n_rows):
    if n_rows <= 2:
        return 0.20
    if n_rows <= 4:
        return 0.12
    if n_rows <= 6:
        return 0.10
    return 0.12



class ParentFigureAssemblerMPL:
    """
    Matplotlib version of ParentFigureAssembler.
    Replicates the interleave/stack logic and title/x/y handling.
    """

    def __init__(self, *, x_title="Optimal Arc Type"):
        self.x_title = x_title

    def assemble(self, main_fig, diff_fig, *,
                main_y_title, diff_y_title, overall_title, x_title=None):

        # Extract child grids
        m_rows, m_cols = _grid_shape(main_fig)
        d_rows, d_cols = _grid_shape(diff_fig)

        grids_match = (m_rows == d_rows) and (m_cols == d_cols)
        expected_parent_rows = 2 * m_rows if grids_match else (m_rows + d_rows)
        vspace = _adaptive_vspace(expected_parent_rows)

        # Panel size
        panel_width = 5
        panel_height = 4.2

        # Setup figure
        parent_rows = 2 * m_rows if grids_match else (m_rows + d_rows)
        parent_cols = m_cols if grids_match else max(m_cols, d_cols)

        fig = plt.figure(
            figsize=(panel_width * parent_cols, panel_height * parent_rows),
            constrained_layout=False
        )
        gs = GridSpec(
            parent_rows, parent_cols,
            figure=fig,
            hspace=vspace,
            wspace=0.4,
        )

        # Build empty axes grid
        parent_axes = np.empty((parent_rows, parent_cols), dtype=object)
        for r in range(parent_rows):
            for c in range(parent_cols):
                parent_axes[r, c] = fig.add_subplot(gs[r, c])

        # Copy child content
        main_axes_grid = _get_axes_grid(main_fig)
        diff_axes_grid = _get_axes_grid(diff_fig)

        if grids_match:
            # Interleave rows: [main, diff, main, diff, ...]
            for r in range(m_rows):
                for c in range(m_cols):
                    _copy_axes_content(main_axes_grid[r, c], parent_axes[2*r, c])
                    _copy_axes_content(diff_axes_grid[r, c], parent_axes[2*r + 1, c])
        else:
            # Stacked mode
            for r in range(m_rows):
                for c in range(m_cols):
                    _copy_axes_content(main_axes_grid[r, c], parent_axes[r, c])
            for r in range(d_rows):
                for c in range(d_cols):
                    _copy_axes_content(diff_axes_grid[r, c], parent_axes[m_rows + r, c])

        # Axis styling
        chosen_x_title = x_title if x_title is not None else self.x_title
        bottom_row = parent_rows - 1

        for r in range(parent_rows):
            for c in range(parent_cols):
                ax = parent_axes[r, c]

                # Apply controlled sizing
                ax.tick_params(labelsize=9)

                if r != bottom_row:
                    ax.set_xticklabels([])
                    ax.set_xlabel("")

        # Y-axis titles
        if grids_match:
            parent_axes[0, 0].set_ylabel(main_y_title, fontsize=11)
            for rr in range(1, parent_rows, 2):
                parent_axes[rr, 0].set_ylabel("")
            for c in range(parent_cols):
                parent_axes[bottom_row, c].set_xlabel(chosen_x_title, fontsize=11)
            for rr in range(1, parent_rows, 2):
                for cc in range(parent_cols):
                    parent_axes[rr, cc].axhline(0, linestyle='--', color='gray', linewidth=1)
        else:
            parent_axes[0, 0].set_ylabel(main_y_title, fontsize=11)
            parent_axes[m_rows, 0].set_ylabel(diff_y_title, fontsize=11)
            for c in range(parent_cols):
                parent_axes[bottom_row, c].set_xlabel(chosen_x_title, fontsize=11)
            for rr in range(m_rows, parent_rows):
                for cc in range(parent_cols):
                    parent_axes[rr, cc].axhline(0, linestyle='--', color='gray', linewidth=1)

        # Supertitle
        fig.suptitle(overall_title, fontsize=16, y=0.98)

        # Adjust final layout
        fig.subplots_adjust(
            top=0.92, bottom=0.08, left=0.08, right=0.97,
            hspace=vspace, wspace=0.25
        )

        return fig


import numpy as np
import matplotlib.colors as mcolors
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.patches import Rectangle

def _copy_axes_content(src_ax, dst_ax):
    """
    Copy lines, scatter points, error ribbons (fill_between), patches,
    titles, labels, limits, and ticks from src_ax → dst_ax *without*
    reusing artists, so Matplotlib never throws an error.

    This version:
        - preserves alpha (opacity)
        - preserves group colors
        - preserves CI ribbons
        - removes dummy legend markers
        - avoids copying background patches
        - prevents ghost points in lower-left
    """

    # ----------------------------------------------------
    # 1. Copy Line2D objects (median lines, diff lines)
    # ----------------------------------------------------
    for line in src_ax.lines:
        dst_ax.plot(
            line.get_xdata(),
            line.get_ydata(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker(),
            markersize=line.get_markersize(),
            color=line.get_color(),
            alpha=(line.get_alpha() if line.get_alpha() is not None else 1.0),
            label=(line.get_label() if line.get_label() != '_child' else None),
        )

    # ----------------------------------------------------
    # 2. Copy Collections (scatter, fill_between)
    # ----------------------------------------------------
    for coll in src_ax.collections:

        # ---------- A) Scatter points (PathCollection) ----------
        if isinstance(coll, PathCollection):
            offsets = coll.get_offsets()

            # Skip dummy legend markers placed at (0,0)
            if offsets.shape[0] == 1 and np.allclose(offsets[0], (0, 0)):
                continue

            # Sizes
            sizes = coll.get_sizes()
            if sizes is None or len(sizes) == 0:
                sizes = 20.0
            elif len(sizes) == 1:
                sizes = float(sizes[0])
            elif len(sizes) != offsets.shape[0]:
                sizes = float(np.nanmean(sizes))

            # Skip zero-size invisible markers
            if (isinstance(sizes, (int, float)) and sizes == 0) or \
               (hasattr(sizes, "__len__") and np.nanmax(sizes) == 0):
                continue

            # Facecolor includes alpha → preserve it
            fc = coll.get_facecolor()
            rgba = fc[0] if len(fc) else (0, 0, 0, 1)

            dst_ax.scatter(
                offsets[:, 0],
                offsets[:, 1],
                s=sizes,
                color=rgba,
                edgecolors=coll.get_edgecolor(),
                alpha=rgba[3],  # preserve original opacity
            )
            continue  # done with scatter

        # ---------- B) CI ribbons (PolyCollection from fill_between) ----------
        if isinstance(coll, PolyCollection):
            # CI ribbons always have real paths
            fc = coll.get_facecolor()
            rgba = fc[0] if len(fc) else (0, 0, 0, 0.2)

            for path in coll.get_paths():
                verts = path.vertices
                xs = verts[:, 0]
                ys = verts[:, 1]

                dst_ax.fill(
                    xs,
                    ys,
                    facecolor=rgba,   # preserve alpha exactly
                    edgecolor='none'
                )
            continue  # done with ribbon

        # ---------- C) Anything else — safe fallback ----------
        # Skip background patch
        if hasattr(coll, 'get_paths'):
            continue

    # ----------------------------------------------------
    # 3. Copy simple polygon patches (rare)
    # ----------------------------------------------------
    for patch in src_ax.patches:
        # Skip background rectangle
        if isinstance(patch, Rectangle) and patch.get_width() == 1 and patch.get_height() == 1:
            continue

        verts = patch.get_path().vertices
        xs, ys = verts[:, 0], verts[:, 1]

        rgba = patch.get_facecolor()

        dst_ax.fill(
            xs,
            ys,
            facecolor=rgba,
            edgecolor=patch.get_edgecolor(),
            alpha=rgba[3] if len(rgba) == 4 else 1.0,
        )

    # ----------------------------------------------------
    # 4. Copy limits, labels, titles, ticks
    # ----------------------------------------------------
    dst_ax.set_xlim(src_ax.get_xlim())
    dst_ax.set_ylim(src_ax.get_ylim())

    dst_ax.set_xlabel(src_ax.get_xlabel())
    dst_ax.set_ylabel(src_ax.get_ylabel())
    dst_ax.set_title(src_ax.get_title())

    dst_ax.set_xticks(src_ax.get_xticks())
    dst_ax.set_xticklabels([t.get_text() for t in src_ax.get_xticklabels()])

    dst_ax.set_yticks(src_ax.get_yticks())
    dst_ax.set_yticklabels([t.get_text() for t in src_ax.get_yticklabels()])

    # ----------------------------------------------------
    # IMPORTANT: lock limits so artists don't expand them
    # ----------------------------------------------------
    dst_ax.autoscale(enable=False)

    return dst_ax
