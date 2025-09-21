# parent_assembler.py
import re
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def _axes_domains(fig):
    lay = (fig.to_dict() or {}).get('layout', {})
    xmap, ymap = {}, {}
    for k, v in lay.items():
        if not isinstance(v, dict):
            continue
        m = re.fullmatch(r'xaxis(\d*)', k)
        if m and 'domain' in v and v['domain'] is not None:
            xmap[m.group(1) or '1'] = tuple(np.round(v['domain'], 6))
        m = re.fullmatch(r'yaxis(\d*)', k)
        if m and 'domain' in v and v['domain'] is not None:
            ymap[m.group(1) or '1'] = tuple(np.round(v['domain'], 6))
    if not xmap:
        xmap['1'] = (0.0, 1.0)
    if not ymap:
        ymap['1'] = (0.0, 1.0)
    return xmap, ymap


def _grid_shape_from_domains(xmap, ymap):
    x_domains = sorted(set(xmap.values()), key=lambda d: d[0])
    y_domains = sorted(set(ymap.values()), key=lambda d: d[0], reverse=True)
    return len(y_domains), len(x_domains), x_domains, y_domains


def _axis_id_of_trace(tr, which):
    ref = getattr(tr, which, None)
    if not ref:
        return '1'
    m = re.fullmatch(r'[xy](\d*)', ref)
    return (m.group(1) or '1') if m else '1'


def _row_col_for_trace(fig, tr):
    xmap, ymap = _axes_domains(fig)
    rows, cols, x_domains, y_domains = _grid_shape_from_domains(xmap, ymap)
    xid = _axis_id_of_trace(tr, 'xaxis')
    yid = _axis_id_of_trace(tr, 'yaxis')
    xdom = xmap.get(xid, (0.0, 1.0))
    ydom = ymap.get(yid, (0.0, 1.0))
    col = x_domains.index(xdom) + 1
    row = y_domains.index(ydom) + 1
    return row, col, rows, cols


def make_parent_figure_from_children(main_fig, diff_fig,
                                     interleave_if_grid_match=True,
                                     vertical_spacing=0.14,
                                     horizontal_spacing=0.06,
                                     top_share=0.68):
    xmap_m, ymap_m = _axes_domains(main_fig)
    m_rows, m_cols, *_ = _grid_shape_from_domains(xmap_m, ymap_m)

    xmap_d, ymap_d = _axes_domains(diff_fig)
    d_rows, d_cols, *_ = _grid_shape_from_domains(xmap_d, ymap_d)

    grids_match = (m_rows == d_rows) and (m_cols == d_cols)

    if interleave_if_grid_match and grids_match:
        parent_rows, parent_cols = 2 * m_rows, m_cols
        bot_share = 1.0 - top_share
        row_heights = sum(
            ([top_share / max(m_rows, 1), bot_share / max(m_rows, 1)] for _ in range(m_rows)), [])
        parent = make_subplots(
            rows=parent_rows, cols=parent_cols,
            shared_xaxes=False,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing,
            row_heights=row_heights
        )
        # paste main → odd rows, diff → even rows
        for tr in main_fig.data:
            r, c, *_ = _row_col_for_trace(main_fig, tr)
            parent.add_trace(tr, row=(r - 1) * 2 + 1, col=c)
        for tr in diff_fig.data:
            r, c, *_ = _row_col_for_trace(diff_fig, tr)
            parent.add_trace(tr, row=(r - 1) * 2 + 2, col=c)
    else:
        parent_rows, parent_cols = m_rows + d_rows, max(m_cols, d_cols)
        top_total, bot_total = top_share, 1.0 - top_share
        top_heights = [top_total / max(m_rows, 1)] * m_rows
        bot_heights = [bot_total / max(d_rows, 1)] * d_rows
        parent = make_subplots(
            rows=parent_rows, cols=parent_cols,
            shared_xaxes=False,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing,
            row_heights=top_heights + bot_heights
        )
        # stacked paste
        for tr in main_fig.data:
            r, c, *_ = _row_col_for_trace(main_fig, tr)
            parent.add_trace(tr, row=r, col=c)
        for tr in diff_fig.data:
            r, c, *_ = _row_col_for_trace(diff_fig, tr)
            parent.add_trace(tr, row=m_rows + r, col=c)

    return parent, parent_rows, parent_cols, m_rows, grids_match


def _adaptive_vspace(n_rows):
    # spacing is fraction of total height between EACH pair of rows
    # keep it small when there are many rows
    if n_rows <= 2:
        return 0.10
    if n_rows <= 4:
        return 0.06
    if n_rows <= 6:
        return 0.05
    return 0.04


class ParentFigureAssembler:
    def __init__(self, *, x_title="Optimal Arc Type"):
        self.x_title = x_title

    def assemble(self, main_fig, diff_fig, *,
                 main_y_title, diff_y_title, overall_title, x_title=None):

        xmap_m, ymap_m = _axes_domains(main_fig)
        m_rows, m_cols, *_ = _grid_shape_from_domains(xmap_m, ymap_m)

        xmap_d, ymap_d = _axes_domains(diff_fig)
        d_rows, d_cols, *_ = _grid_shape_from_domains(xmap_d, ymap_d)

        grids_match = (m_rows == d_rows) and (m_cols == d_cols)
        expected_parent_rows = 2 * m_rows if grids_match else (m_rows + d_rows)
        vspace = _adaptive_vspace(expected_parent_rows)

        # 2) Now create the parent with the chosen spacing
        fig, parent_rows, parent_cols, m_rows, grids_match = make_parent_figure_from_children(
            main_fig, diff_fig,
            interleave_if_grid_match=True,
            vertical_spacing=vspace,      # <- use computed spacing
            horizontal_spacing=0.06,
            top_share=0.68
        )

        # Remove child annotations, clear axis titles
        if getattr(fig.layout, 'annotations', None):
            fig.layout.annotations = ()
        lay = (fig.to_dict() or {}).get('layout', {})
        for k in list(lay.keys()):
            if k.startswith('xaxis') or k.startswith('yaxis'):
                try:
                    fig['layout'][k]['title'] = None
                except Exception:
                    pass

        chosen_x_title = x_title if (x_title is not None) else self.x_title

        if grids_match:
            # Interleaved layout: rows = [main1, diff1, main2, diff2, ...]
            # 1) Put y-title ONLY on the first main row; hide on diff rows
            fig.update_yaxes(title_text=main_y_title, row=1,
                             col=1)  # first "main" row
            # hide on first "diff" row
            fig.update_yaxes(title_text=None, row=2, col=1)

            # Optionally clear any y-titles on later rows (belt & suspenders)
            for r in range(3, parent_rows + 1):
                fig.update_yaxes(title_text=None, row=r, col=1)

            # 2) x-title only on the bottom-most row (unchanged)
            for c in range(1, parent_cols + 1):
                fig.update_xaxes(title_text=chosen_x_title,
                                 row=parent_rows, col=c)

            # 3) Zero reference line on every even row (all diff rows)
            for rr in range(2, parent_rows + 1, 2):
                for c in range(1, parent_cols + 1):
                    fig.add_hline(y=0, line_dash='dash',
                                  line_color='gray', row=rr, col=c)

        else:
            # Stacked layout: top block (main rows), then bottom block (diff rows)
            fig.update_yaxes(title_text=main_y_title, row=1, col=1)
            fig.update_yaxes(title_text=diff_y_title, row=m_rows + 1, col=1)

            for c in range(1, parent_cols + 1):
                fig.update_xaxes(title_text=chosen_x_title,
                                 row=parent_rows, col=c)

            # Zero line across every row in bottom block
            for rr in range(m_rows + 1, parent_rows + 1):
                for c in range(1, parent_cols + 1):
                    fig.add_hline(y=0, line_dash='dash',
                                  line_color='gray', row=rr, col=c)

        fig.update_layout(
            title=overall_title,
            template='plotly_white',
            showlegend=True,
            margin=dict(l=80, r=80, t=100, b=90),
            width=1100,
            height=600 + 200 * (parent_rows - 2),
            legend=dict(orientation="v", y=0.5, x=1.02)
        )

        # Add a legend item for y=0 without drawing another line in panels
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 0], mode='lines',
            line=dict(dash='dash', color='gray'),
            name='y = 0', hoverinfo='skip', visible='legendonly',
            legendrank=100000  # ensure it appears after other legend items
        ))

        # prevent label/tick clipping on all subplots
        for r in range(1, parent_rows + 1):
            for c in range(1, parent_cols + 1):
                # show x tick labels only on 'difference' rows
                is_diff_row = (r % 2 == 0) if grids_match else (r > m_rows)
                fig.update_xaxes(
                    automargin=True, title_standoff=8,
                    showline=True, ticks='outside',
                    showticklabels=is_diff_row,
                    row=r, col=c)
                fig.update_yaxes(
                    automargin=True, title_standoff=8,
                    showline=True, ticks='outside',
                    row=r, col=c)

        # Propagate original diff child x ticks to parent difference subplots
        try:
            diff_lay = (diff_fig.to_dict() or {}).get('layout', {})
            processed = set()
            for tr in diff_fig.data:
                cr, cc, *_ = _row_col_for_trace(diff_fig, tr)
                pr = (cr - 1) * 2 + 2 if grids_match else (m_rows + cr)
                pc = cc
                key = (pr, pc)
                if key in processed:
                    continue
                xid = _axis_id_of_trace(tr, 'xaxis')
                ax_key = 'xaxis' if xid == '1' else f'xaxis{xid}'
                ax_dict = diff_lay.get(ax_key, {})
                tickvals = ax_dict.get('tickvals', None)
                ticktext = ax_dict.get('ticktext', None)
                if tickvals is not None and ticktext is not None:
                    fig.update_xaxes(
                        tickmode='array', tickvals=tickvals, ticktext=ticktext, row=pr, col=pc)
                    processed.add(key)
        except Exception:
            pass

        return fig
