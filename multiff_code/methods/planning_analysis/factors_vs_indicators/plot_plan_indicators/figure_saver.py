# ============================================================
# figure_saver.py — Save Plotly or Matplotlib Figures
# ============================================================
import os

def save_figure(fig, filename, backend='plotly', formats=('png', 'pdf', 'svg')):
    """
    Save figures from either backend (Plotly or Matplotlib).
    
    Parameters
    ----------
    fig : figure object
        - Plotly: go.Figure
        - Matplotlib: matplotlib.figure.Figure
    filename : str
        Base filename (no extension)
    backend : str
        'plotly' or 'matplotlib'
    formats : list/tuple of strings
        ('png', 'pdf', 'svg')
    """
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

    for ext in formats:
        out = f"{filename}.{ext}"

        if backend == 'plotly':
            # requires kaleido installed: pip install -U kaleido
            fig.write_image(out)

        elif backend == 'matplotlib':
            fig.savefig(out, bbox_inches='tight')

        else:
            raise ValueError("backend must be 'plotly' or 'matplotlib'")

    return [f"{filename}.{ext}" for ext in formats]
