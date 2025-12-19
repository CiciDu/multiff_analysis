# ============================================================
# plot_styles_mpl.py — Matplotlib publication style utilities
# ============================================================
import matplotlib as mpl
import matplotlib.pyplot as plt

def use_publication_style(
    font_family='sans',            # 'serif' or 'sans'
    base_font_size=11,             # 9–12 good for publication
    line_width=1.5,
    axes_line_width=1.0,
    tick_length=3,
    color_palette='cbf'            # 'cbf' = colorblind-friendly
):
    """
    Apply a high-quality, publication-ready Matplotlib style globally.
    """

    if font_family == 'serif':
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'CMU Serif']
    else:
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    # Core figure settings
    mpl.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'figure.autolayout': False,
        'figure.constrained_layout.use': True,

        'axes.titlesize': base_font_size + 1,
        'axes.labelsize': base_font_size,
        'axes.linewidth': axes_line_width,

        'xtick.labelsize': base_font_size - 1,
        'ytick.labelsize': base_font_size - 1,
        'xtick.major.size': tick_length,
        'ytick.major.size': tick_length,

        'lines.linewidth': line_width,
        'lines.markersize': 5,

        'legend.fontsize': base_font_size - 1,
        'legend.frameon': False,
        'legend.handlelength': 1.5,
    })

    # color palettes
    if color_palette == 'cbf':
        # Paul Tol colorblind-friendly palette
        cbf_colors = [
            '#332288', '#88CCEE', '#44AA99', '#117733', '#999933',
            '#DDCC77', '#CC6677', '#882255', '#AA4499'
        ]
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=cbf_colors)

    elif color_palette == 'matlab':
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(
            color=['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
                   '#77AC30', '#4DBEEE', '#A2142F']
        )


