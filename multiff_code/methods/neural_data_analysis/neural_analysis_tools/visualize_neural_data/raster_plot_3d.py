import numpy as np
import pandas as pd

PLOTLY_EVENT_COLORS = [
    'rgba(66, 135, 245, 0.07)',   # blue
    'rgba(245, 66, 66, 0.07)',    # red
    'rgba(66, 245, 126, 0.07)',   # green
    'rgba(245, 207, 66, 0.07)',   # yellow
    'rgba(179, 66, 245, 0.07)',   # purple
    'rgba(66, 245, 230, 0.07)',   # cyan
    'rgba(245, 133, 66, 0.07)',   # orange
]

def resolve_event_windows_vectorized(new_seg_info, start_cols, end_cols):
    """
    Vectorized resolver for event windows (start, end) per segment.
    Skips rows where:
        - start is NA
        - end is NA
        - end <= start
    Returns:
        List[Dict[int, (float, float)]]
    """

    df = new_seg_info.copy()

    # Create rel_ versions if reference_time is available
    if 'reference_time' in df.columns:
        for col in set(start_cols + end_cols):
            rel_col = f"rel_{col}"
            if rel_col not in df.columns and col in df.columns:
                df[rel_col] = df[col] - df['reference_time']

    segments = df['new_segment'].values
    all_event_windows = []

    for start_col, end_col in zip(start_cols, end_cols):

        # Prefer rel_ columns if they exist
        s_col = f"rel_{start_col}" if f"rel_{start_col}" in df.columns else start_col
        e_col = f"rel_{end_col}"   if f"rel_{end_col}"   in df.columns else end_col

        # Missing required columns → skip this whole event type
        if s_col not in df.columns or e_col not in df.columns:
            all_event_windows.append({})
            continue

        # Convert to numpy float array (pd.NA -> np.nan)
        starts = pd.to_numeric(df[s_col], errors='coerce').to_numpy(dtype=float)
        ends   = pd.to_numeric(df[e_col], errors='coerce').to_numpy(dtype=float)

        valid_mask = (
            (~np.isnan(starts)) &
            (~np.isnan(ends)) &
            (ends > starts)
        )

        # Build dictionary for this event type
        event_dict = {
            int(seg): (float(st), float(en))
            for seg, st, en, valid
            in zip(segments, starts, ends, valid_mask)
            if valid
        }

        all_event_windows.append(event_dict)

    return all_event_windows




import plotly.graph_objects as go

def plot_3d_raster_plotly(
    aligned_spike_trains,
    new_seg_info,
    event_start_cols,
    event_end_cols,
    time_col='rel_spike_time',
    trial_col='new_segment',
    neuron_col='cluster',
    max_segments=None,
    max_neurons=None,
    title='3D Spike Raster'
):

    df = aligned_spike_trains.copy()

    # limit trials / neurons if desired
    if max_segments is not None:
        allowed = np.sort(df[trial_col].unique())[:max_segments]
        df = df[df[trial_col].isin(allowed)]
    if max_neurons is not None:
        allowed = np.sort(df[neuron_col].unique())[:max_neurons]
        df = df[df[neuron_col].isin(allowed)]

    segments = np.sort(df[trial_col].unique())
    neurons = np.sort(df[neuron_col].unique())

    # ---------- SPIKE POINTS (scatter3d) ----------
    fig = go.Figure()

    event_windows_list = resolve_event_windows_vectorized(
        new_seg_info, event_start_cols, event_end_cols
    )

    z_min = neurons.min() - 0.5
    z_max = neurons.max() + 0.5

    for event_idx, event_windows in enumerate(event_windows_list):

        # pick a color for this event type
        color = PLOTLY_EVENT_COLORS[event_idx % len(PLOTLY_EVENT_COLORS)]

        for seg, (t_start, t_end) in event_windows.items():

            if seg not in segments:
                continue

            add_plotly_event_box(
                fig,
                t_start, t_end,
                y_center=seg,
                z_min=z_min,
                z_max=z_max,
                thickness_y=1.0,
                color=color,
            )

    # plot spikes
    fig.add_trace(go.Scatter3d(
        x=df[time_col],
        y=df[trial_col],
        z=df[neuron_col],
        mode='markers',
        marker=dict(size=2, color='black'),
        name='Spikes'
    ))

    # ---------- LAYOUT ----------
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Time',
            yaxis_title='Segment',
            zaxis_title='Neuron',
            aspectmode='cube',
        ),
        height=800
    )

    fig.show()


def add_plotly_event_box(fig, t_start, t_end, y_center, z_min, z_max,
                         thickness_y=1.0, color='rgba(66,135,245,0.25)'):
    """
    Draws a full 3D rectangular prism for an event window.
    """
    y0 = y_center - thickness_y/2
    y1 = y_center + thickness_y/2

    X = [t_start, t_end]
    Y = [y0, y1]
    Z = [z_min, z_max]

    faces = [
        # bottom
        ( [[X[0], X[1]], [X[0], X[1]]],
          [[Y[0], Y[0]], [Y[1], Y[1]]],
          [[Z[0], Z[0]], [Z[0], Z[0]]] ),
        # top
        ( [[X[0], X[1]], [X[0], X[1]]],
          [[Y[0], Y[0]], [Y[1], Y[1]]],
          [[Z[1], Z[1]], [Z[1], Z[1]]] ),
        # front
        ( [[X[0], X[1]], [X[0], X[1]]],
          [[Y[0], Y[0]], [Y[0], Y[0]]],
          [[Z[0], Z[0]], [Z[1], Z[1]]] ),
        # back
        ( [[X[0], X[1]], [X[0], X[1]]],
          [[Y[1], Y[1]], [Y[1], Y[1]]],
          [[Z[0], Z[0]], [Z[1], Z[1]]] ),
        # left
        ( [[X[0], X[0]], [X[0], X[0]]],
          [[Y[0], Y[1]], [Y[0], Y[1]]],
          [[Z[0], Z[0]], [Z[1], Z[1]]] ),
        # right
        ( [[X[1], X[1]], [X[1], X[1]]],
          [[Y[0], Y[1]], [Y[0], Y[1]]],
          [[Z[0], Z[0]], [Z[1], Z[1]]] ),
    ]

    for Xf, Yf, Zf in faces:
        fig.add_trace(go.Surface(
            x=Xf, y=Yf, z=Zf,
            showscale=False,
            opacity=1.0,           # opacity already in RGBA
            surfacecolor=[[0,0],[0,0]],  # needed dummy colormap
            colorscale=[[0, color], [1, color]],
            hoverinfo='skip'
        ))
