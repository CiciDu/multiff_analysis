import numpy as np
import matplotlib.pyplot as plt

from planning_analysis.factors_vs_indicators import process_variations_utils
from planning_analysis.factors_vs_indicators.plot_plan_indicators import (
    plot_variations_utils,
    plot_variations_utils_mpl,
    parent_assembler,
    parent_assembler_mpl,
    plot_styles_mpl,
)

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


def _prepare_ci_bounds(df, col_median, col_low, col_high, to_degrees=False):
    """Create ci_lower / ci_upper and optionally convert to degrees."""
    df = df.copy()
    if to_degrees:
        df[col_median] = df[col_median] * 180 / np.pi
        df['ci_lower'] = df[col_low] * 180 / np.pi
        df['ci_upper'] = df[col_high] * 180 / np.pi
    else:
        df['ci_lower'] = df[col_low]
        df['ci_upper'] = df[col_high]
    return df


def _filter_difference(df, is_difference):
    df = df.copy()
    if is_difference:
        return df[df['test_or_control'] == 'difference']
    return df[df['test_or_control'] != 'difference']


def _build_child_fig(
    backend,
    data,
    *,
    fixed_variable_values_to_use,
    changeable_variables,
    x_var_column_list,
    y_var_column,
    var_to_determine_x_offset_direction='test_or_control',
    columns_to_find_unique_combinations_for_color=None,
    columns_to_find_unique_combinations_for_line=None,
    use_subplots_based_on_changeable_variables=False,
    is_difference=False,
    constant_marker_size=None,
    show_fig=False,
):
    """Backend-agnostic child figure builder."""
    columns_to_find_unique_combinations_for_color = (
        columns_to_find_unique_combinations_for_color or []
    )
    columns_to_find_unique_combinations_for_line = (
        columns_to_find_unique_combinations_for_line or []
    )

    common = dict(
        fixed_variable_values_to_use=fixed_variable_values_to_use,
        changeable_variables=changeable_variables,
        x_var_column_list=x_var_column_list,
        y_var_column=y_var_column,
        var_to_determine_x_offset_direction=var_to_determine_x_offset_direction,
        columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
        columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
        use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables,
        is_difference=is_difference,
        constant_marker_size=constant_marker_size,
        show_fig=show_fig,
    )

    if backend == 'plotly':
        fig = plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(
            data, **common
        )
    elif backend == 'matplotlib':
        # make sure style is applied once per session
        plot_styles_mpl.use_publication_style()
        fig = plot_variations_utils_mpl.streamline_making_matplotlib_plot_to_compare_two_sets_of_data(  # noqa: E501
            data, **common
        )
    else:
        raise ValueError(
            f"Unknown backend {backend!r} (expected 'plotly' or 'matplotlib').")

    if hasattr(fig, "_combinations"):
        combinations = fig._combinations
    else:
        combinations = []

    if use_subplots_based_on_changeable_variables and combinations:
        _inject_changeable_variable_titles(fig, combinations, backend)

    return fig


def _set_heading_labels_plotly(fig, x_var_column_list, is_difference):
    if len(x_var_column_list) == 1 and 'ref_point_value' in x_var_column_list:
        if is_difference:
            fig.update_layout(
                title=(
                    "Test vs Control Difference in Absolute Angle to Next FF "
                    "(Median ± 95% BCa CI)"
                ),
                xaxis_title="Reference Distance (cm)",
                yaxis_title="Difference in Median Angle(°)",
            )
        else:
            fig.update_layout(
                title="Absolute Angle to Next FF (Median ± 95% BCa CI)",
                xaxis_title="Reference Distance (cm)",
                yaxis_title="Angle (°)",
            )


def _set_heading_labels_mpl(fig, x_var_column_list, is_difference):
    # global title
    if len(x_var_column_list) == 1 and 'ref_point_value' in x_var_column_list:
        if is_difference:
            title = (
                "Test vs Control Difference in Absolute Angle to Next FF "
                "(Median ± 95% BCa CI)"
            )
            y_label = "Difference in Median Angle (°)"
        else:
            title = "Absolute Angle to Next FF (Median ± 95% BCa CI)"
            y_label = "Angle (°)"
    else:
        title = "Absolute Angle to Next FF"
        y_label = "Angle (°)"

    fig.suptitle(title, fontsize=14, y=0.98)
    axes = fig.axes
    if not axes:
        return
    # leftmost axis: y label
    axes[0].set_ylabel(y_label)
    # bottom row: x label
    bottom_y = max(ax.get_subplotspec().rowspan.stop for ax in axes)
    for ax in axes:
        if ax.get_subplotspec().rowspan.stop == bottom_y:
            ax.set_xlabel("Reference Distance (cm)")


def _apply_same_side_ylim_plotly(fig, y_min, y_max):
    fig.update_yaxes(range=[y_min, y_max])


def _apply_same_side_ylim_mpl(fig, y_min, y_max):
    for ax in fig.axes:
        ax.set_ylim(y_min, y_max)


# ---------------------------------------------------------------------
# Public, backend-aware API (replacing class methods)
# ---------------------------------------------------------------------
def plot_median_heading(
    all_ref_median_info,
    x_var_column_list=['ref_point_value'],
    fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                  'key_for_split': 'ff_seen'},
    changeable_variables=['whether_even_out_dist'],
    columns_to_find_unique_combinations_for_color=None,
    columns_to_find_unique_combinations_for_line=None,
    add_ci_bounds=True,
    use_subplots_based_on_changeable_variables=False,
    is_difference=False,
    backend='matplotlib',
    show_fig=True,
):
    df = all_ref_median_info.copy()

    if add_ci_bounds:
        df = _prepare_ci_bounds(
            df,
            col_median='diff_in_abs_angle_to_nxt_ff_median',
            col_low='diff_in_abs_angle_to_nxt_ff_ci_low_95',
            col_high='diff_in_abs_angle_to_nxt_ff_ci_high_95',
            to_degrees=True,
        )
    else:
        # only convert the median
        df['diff_in_abs_angle_to_nxt_ff_median'] = (
            df['diff_in_abs_angle_to_nxt_ff_median'] * 180 / np.pi
        )

    df = _filter_difference(df, is_difference)

    fig = _build_child_fig(
        backend,
        df,
        fixed_variable_values_to_use=fixed_variable_values_to_use,
        changeable_variables=changeable_variables,
        x_var_column_list=x_var_column_list,
        y_var_column='diff_in_abs_angle_to_nxt_ff_median',
        columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,  # noqa: E501
        columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,  # noqa: E501
        use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables,
        is_difference=is_difference,
        show_fig=show_fig,
    )

    if backend == 'plotly':
        _set_heading_labels_plotly(fig, x_var_column_list, is_difference)
    else:
        _set_heading_labels_mpl(fig, x_var_column_list, is_difference)

    return fig


def plot_median_curv(
    all_ref_median_info,
    x_var_column_list=['ref_point_value'],
    fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                  'key_for_split': 'ff_seen'},
    changeable_variables=['whether_even_out_dist'],
    columns_to_find_unique_combinations_for_color=None,
    columns_to_find_unique_combinations_for_line=None,
    add_ci_bounds=True,
    use_subplots_based_on_changeable_variables=False,
    is_difference=False,
    backend='matplotlib',
    show_fig=True,
):
    df = all_ref_median_info.copy()

    # note: d_curv already degrees/m
    if add_ci_bounds:
        df = _prepare_ci_bounds(
            df,
            col_median='diff_in_abs_d_curv_median',
            col_low='diff_in_abs_d_curv_ci_low_95',
            col_high='diff_in_abs_d_curv_ci_high_95',
            to_degrees=False,
        )

    df = _filter_difference(df, is_difference)

    fig = _build_child_fig(
        backend,
        df,
        fixed_variable_values_to_use=fixed_variable_values_to_use,
        changeable_variables=changeable_variables,
        x_var_column_list=x_var_column_list,
        y_var_column='diff_in_abs_d_curv_median',
        columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,  # noqa: E501
        columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,  # noqa: E501
        use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables,
        is_difference=is_difference,
        show_fig=show_fig,
    )

    if backend == 'plotly':
        if len(x_var_column_list) == 1 and 'ref_point_value' in x_var_column_list:
            if is_difference:
                fig.update_layout(
                    title="Test vs Control Difference in Curvature (Median ± 95% BCa CI)",
                    xaxis_title="Reference Distance (cm)",
                    yaxis_title="Difference in Median Curvature (°/m)",
                )
            else:
                fig.update_layout(
                    title=(
                        "Difference in Curvature Across Two Arc Pairs "
                        "(Median ± 95% BCa CI)"
                    ),
                    xaxis_title="Reference Distance (cm)",
                    yaxis_title="Curvature (°/m)",
                )
    else:
        # Matplotlib annotation
        if len(x_var_column_list) == 1 and 'ref_point_value' in x_var_column_list:
            if is_difference:
                title = "Test vs Control Difference in Curvature (Median ± 95% BCa CI)"
                y_label = "Difference in Median Curvature (°/m)"
            else:
                title = (
                    "Difference in Curvature Across Two Arc Pairs "
                    "(Median ± 95% BCa CI)"
                )
                y_label = "Curvature (°/m)"
        else:
            title = "Difference in Curvature Across Two Arc Pairs"
            y_label = "Curvature (°/m)"

        fig.suptitle(title, fontsize=14, y=0.98)
        axes = fig.axes
        if axes:
            axes[0].set_ylabel(y_label)
            bottom_y = max(ax.get_subplotspec().rowspan.stop for ax in axes)
            for ax in axes:
                if ax.get_subplotspec().rowspan.stop == bottom_y:
                    ax.set_xlabel("Reference Distance (cm)")

    return fig


def plot_same_side_percentage(
    perc_info,
    x_var_column_list=['monkey_name'],
    fixed_variable_values_to_use={
        'if_test_nxt_ff_group_appear_after_stop': 'flexible'},
    changeable_variables=['monkey_name'],
    columns_to_find_unique_combinations_for_color=None,
    add_ci_bounds=True,
    use_subplots_based_on_changeable_variables=False,
    is_difference=False,
    y_min=None,
    y_max=None,
    backend='matplotlib',
    show_fig=True,
):
    df = perc_info.copy()

    df['perc'] = df['perc'] * 100
    if add_ci_bounds:
        df['ci_lower'] = df['perc_ci_low_95'] * 100
        df['ci_upper'] = df['perc_ci_high_95'] * 100

    df = _filter_difference(df, is_difference)
    df = process_variations_utils.make_new_df_for_plotly_comparison(
        df, match_rows_based_on_ref_columns_only=False
    )

    fig = _build_child_fig(
        backend,
        df,
        fixed_variable_values_to_use=fixed_variable_values_to_use,
        changeable_variables=changeable_variables,
        x_var_column_list=x_var_column_list,
        y_var_column='perc',
        var_to_determine_x_offset_direction='test_or_control',
        columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,  # noqa: E501
        use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables,
        is_difference=is_difference,
        show_fig=show_fig,
    )

    # y axis limits
    if y_min is None:
        y_min = df['ci_lower'].min() if add_ci_bounds else df['perc'].min()
        y_min = max(0, y_min - 20)
    if y_max is None:
        y_max = df['ci_upper'].max() if add_ci_bounds else df['perc'].max()
        y_max = min(100, y_max + 5)

    if backend == 'plotly':
        fig.update_layout(
            title="Same-Side Stop Rate",
            xaxis_title=None,
            yaxis_title="Same-Side Stop Rate",
        )
        _apply_same_side_ylim_plotly(fig, y_min, y_max)
        if show_fig:
            fig.show()
    else:
        fig.suptitle("Same-Side Stop Rate", fontsize=14, y=0.98)
        axes = fig.axes
        if axes:
            axes[0].set_ylabel("Same-Side Stop Rate")
            bottom_y = max(ax.get_subplotspec().rowspan.stop for ax in axes)
            for ax in axes:
                if ax.get_subplotspec().rowspan.stop == bottom_y:
                    ax.set_xlabel(None)
        _apply_same_side_ylim_mpl(fig, y_min, y_max)
        if show_fig:
            plt.show()

    return fig


def plot_same_side_percentage_across_monkeys(
    perc_info,
    x_var_column_list=['monkey_name'],
    fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                  'key_for_split': 'ff_seen',
                                  'whether_even_out_dist': False},
    changeable_variables=None,
    columns_to_find_unique_combinations_for_color=None,
    add_ci_bounds=True,
    backend='matplotlib',
):
    changeable_variables = changeable_variables or []
    return plot_same_side_percentage(
        perc_info=perc_info,
        x_var_column_list=x_var_column_list,
        fixed_variable_values_to_use=fixed_variable_values_to_use,
        changeable_variables=changeable_variables,
        columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,  # noqa: E501
        add_ci_bounds=add_ci_bounds,
        use_subplots_based_on_changeable_variables=True,
        y_min=45,
        y_max=65,
        backend=backend,
        show_fig=(backend == 'plotly'),
    )


# ---------------------------------------------------------------------
# Combined “main vs diff” engine (parent figure)
# ---------------------------------------------------------------------


def plot_median_with_difference(
    all_ref_median_info,
    *,
    x_var_column_list,
    fixed_variable_values_to_use,
    changeable_variables,
    columns_to_find_unique_combinations_for_color,
    columns_to_find_unique_combinations_for_line,
    use_subplots_based_on_changeable_variables,
    y_var_column,
    main_y_title,
    diff_y_title,
    overall_title,
    constant_marker_size=None,
    x_title='Reference Distance from Stop at Current Target (cm)',
    backend='matplotlib',
):
    df = all_ref_median_info.copy()

    main_data = df[df['test_or_control'] != 'difference'].copy()
    diff_data = df[df['test_or_control'] == 'difference'].copy()

    if backend == 'plotly':
        main_fig = _build_child_fig(
            'plotly',
            main_data,
            fixed_variable_values_to_use=fixed_variable_values_to_use,
            changeable_variables=changeable_variables,
            x_var_column_list=x_var_column_list,
            y_var_column=y_var_column,
            columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,  # noqa: E501
            columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,  # noqa: E501
            use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables,
            is_difference=False,
            constant_marker_size=constant_marker_size,
        )
        diff_fig = _build_child_fig(
            'plotly',
            diff_data,
            fixed_variable_values_to_use=fixed_variable_values_to_use,
            changeable_variables=changeable_variables,
            x_var_column_list=x_var_column_list,
            y_var_column=y_var_column,
            columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,  # noqa: E501
            columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,  # noqa: E501
            use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables,
            is_difference=True,
            constant_marker_size=constant_marker_size,
        )

        assembler = parent_assembler.ParentFigureAssembler(
            x_title="Optimal Arc Type")
        fig = assembler.assemble(
            main_fig,
            diff_fig,
            main_y_title=main_y_title,
            diff_y_title=diff_y_title,
            overall_title=overall_title,
            x_title=x_title,
        )
        fig.update_xaxes(showline=True, linecolor='black',
                         linewidth=1, mirror=False)
        fig.update_yaxes(showline=True, linecolor='black',
                         linewidth=1, mirror=False)
        return fig

    # ----------------- Matplotlib backend (Option B) -----------------
    plot_styles_mpl.use_publication_style()

    main_fig = _build_child_fig(
        'matplotlib',
        main_data,
        fixed_variable_values_to_use=fixed_variable_values_to_use,
        changeable_variables=changeable_variables,
        x_var_column_list=x_var_column_list,
        y_var_column=y_var_column,
        columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,  # noqa: E501
        columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,  # noqa: E501
        use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables,
        is_difference=False,
        constant_marker_size=constant_marker_size,
    )
    diff_fig = _build_child_fig(
        'matplotlib',
        diff_data,
        fixed_variable_values_to_use=fixed_variable_values_to_use,
        changeable_variables=changeable_variables,
        x_var_column_list=x_var_column_list,
        y_var_column=y_var_column,
        columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,  # noqa: E501
        columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,  # noqa: E501
        use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables,
        is_difference=True,
        constant_marker_size=constant_marker_size,
    )

    # Use the dedicated Matplotlib parent assembler (no Plotly involved)
    assembler_mpl = parent_assembler_mpl.ParentFigureAssemblerMPL(
        x_title="Optimal Arc Type"
    )
    fig = assembler_mpl.assemble(
        main_fig,
        diff_fig,
        main_y_title=main_y_title,
        diff_y_title=diff_y_title,
        overall_title=overall_title,
        x_title=x_title,
    )

    # Prevent child figures from being auto-displayed in notebooks
    plt.close(main_fig)
    plt.close(diff_fig)

    return fig


# Convenience wrappers mirroring the old class API
def plot_median_curv_across_monkeys_and_arc_types_with_difference(
    all_ref_median_info,
    x_var_column_list=['ref_point_value'],
    fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                  'key_for_split': 'ff_seen',
                                  'whether_even_out_dist': False,
                                  'curv_traj_window_before_stop': '[-25, 0]'},
    changeable_variables=['opt_arc_type', 'monkey_name'],
    columns_to_find_unique_combinations_for_color=None,
    columns_to_find_unique_combinations_for_line=None,
    add_ci_bounds=True,
    use_subplots_based_on_changeable_variables=True,
    backend='matplotlib',
):
    df = all_ref_median_info.copy()
    if add_ci_bounds:
        df = _prepare_ci_bounds(
            df,
            col_median='diff_in_abs_d_curv_median',
            col_low='diff_in_abs_d_curv_ci_low_95',
            col_high='diff_in_abs_d_curv_ci_high_95',
            to_degrees=False,
        )
    return plot_median_with_difference(
        df,
        x_var_column_list=x_var_column_list,
        fixed_variable_values_to_use=fixed_variable_values_to_use,
        changeable_variables=changeable_variables,
        columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color or [],  # noqa: E501
        columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line or [],  # noqa: E501
        use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables,  # noqa: E501
        y_var_column='diff_in_abs_d_curv_median',
        main_y_title="Median Curvature (°/m)",
        diff_y_title="Difference in Median Curvature (°/m)",
        overall_title="Difference in Curvature Across Two Arc Pairs: Test vs Control",
        backend=backend,
    )


def plot_median_heading_across_monkeys_and_arc_types_with_difference(
    all_ref_median_info,
    x_var_column_list=['ref_point_value'],
    fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                  'key_for_split': 'ff_seen',
                                  'whether_even_out_dist': False,
                                  'curv_traj_window_before_stop': '[-25, 0]'},
    changeable_variables=['opt_arc_type', 'monkey_name'],
    columns_to_find_unique_combinations_for_color=None,
    columns_to_find_unique_combinations_for_line=None,
    add_ci_bounds=True,
    use_subplots_based_on_changeable_variables=True,
    constant_marker_size=12,
    backend='matplotlib',
):
    df = all_ref_median_info.copy()

    # convert median + bounds to degrees
    df['diff_in_abs_angle_to_nxt_ff_median'] = (
        df['diff_in_abs_angle_to_nxt_ff_median'] * 180 / np.pi
    )
    if add_ci_bounds:
        df['ci_lower'] = df['diff_in_abs_angle_to_nxt_ff_ci_low_95'] * 180 / np.pi
        df['ci_upper'] = df['diff_in_abs_angle_to_nxt_ff_ci_high_95'] * 180 / np.pi

    return plot_median_with_difference(
        df,
        x_var_column_list=x_var_column_list,
        fixed_variable_values_to_use=fixed_variable_values_to_use,
        changeable_variables=changeable_variables,
        columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color or [],  # noqa: E501
        columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line or [],  # noqa: E501
        use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables,  # noqa: E501
        y_var_column='diff_in_abs_angle_to_nxt_ff_median',
        main_y_title="Median Angle (°)",
        diff_y_title="Difference in Median Angle (°)",
        overall_title="Absolute Angle to Next FF: Test vs Control",
        constant_marker_size=constant_marker_size,
        backend=backend,
    )


def _inject_changeable_variable_titles(fig, combinations, backend):
    """
    Ensures subplot titles reflect the current changeable variable combinations.
    Works for both Plotly and Matplotlib.
    """

    if backend == 'plotly':
        # Plotly supports automatic subplot_titles rewriting
        # we overwrite annotations in place
        for ann, combo in zip(fig.layout.annotations, combinations):
            parts = [f"{k}: {v}" for k, v in combo.items()]
            ann.text = " · ".join(parts)
        return

    # Matplotlib backend
    axes = fig.axes
    for ax, combo in zip(axes, combinations):
        parts = [f"{k}: {v}" for k, v in combo.items()]
        ax.set_title(" · ".join(parts), fontsize=10)
