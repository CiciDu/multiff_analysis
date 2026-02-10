import numpy as np
import matplotlib.pyplot as plt
import math


def plot_pgam_tuning_curvetions(
    res,
    indices_of_vars_to_plot=None,
    plot_var_order=None,
):
    """
    Plot PGAM tuning curves with optional variable ordering and numbering.

    Parameters
    ----------
    res : numpy structured array
        PGAM results.
    indices_of_vars_to_plot : array-like or None
        Indices into res specifying which variables to plot.
        Ignored if plot_var_order is provided.
    plot_var_order : list of str or None
        Desired order of variables by name (res['variable']).
        If provided, this overrides indices_of_vars_to_plot.
    """

    # -----------------------------
    # Resolve variable indices
    # -----------------------------
    res_vars = np.array(res['variable'])

    if plot_var_order is not None:
        # keep only variables that actually exist in res
        ordered_vars = [v for v in plot_var_order if v in res_vars]

        if len(ordered_vars) == 0:
            print('No variables from plot_var_order found in res.')
            return

        indices_of_vars_to_plot = np.concatenate([
            np.where(res_vars == v)[0]
            for v in ordered_vars
        ])

    elif indices_of_vars_to_plot is None:
        indices_of_vars_to_plot = np.arange(len(res))

    # -----------------------------
    # Debug: print struct types
    # -----------------------------
    print('\n\n')
    print('Result structarray types\n========================\n')
    for name in res.dtype.names:
        print('%s: \t %s' % (name, type(res[name][0])))

    # -----------------------------
    # Plotting setup
    # -----------------------------
    num_vars = len(indices_of_vars_to_plot)
    num_vars_per_row = 3
    num_plots = math.ceil(num_vars / num_vars_per_row)

    var_counter = 0
    global_var_number = 1

    # -----------------------------
    # Plot loop
    # -----------------------------
    for j in range(num_plots):
        plt.figure(figsize=(5 * num_vars_per_row, 7))

        for k in range(num_vars_per_row):
            if var_counter >= num_vars:
                break

            var_index = indices_of_vars_to_plot[var_counter]
            var_name = res['variable'][var_index]

            # --- log-space ---
            x_kernel = res['x_kernel'][var_index]
            y_kernel = res['y_kernel'][var_index]
            ypCI_kernel = res['y_kernel_pCI'][var_index]
            ymCI_kernel = res['y_kernel_mCI'][var_index]

            plt.subplot(2, num_vars_per_row, k + 1)
            plt.title(f'({global_var_number:02d}) log-space {var_name}')
            plt.plot(x_kernel.reshape(-1), y_kernel.reshape(-1), color='r')
            plt.fill_between(
                x_kernel.reshape(-1),
                ymCI_kernel.reshape(-1),
                ypCI_kernel.reshape(-1),
                color='r',
                alpha=0.3,
            )

            # --- rate-space ---
            x_firing = res['x_rate_Hz'][var_index]
            y_firing_model = res['y_rate_Hz_model'][var_index]
            y_firing_raw = res['y_rate_Hz_raw'][var_index]

            plt.subplot(2, num_vars_per_row, k + 1 + num_vars_per_row)
            plt.title(f'({global_var_number:02d}) rate-space {var_name}')
            plt.plot(
                x_firing[0],
                y_firing_raw.reshape(-1),
                'o-',
                markersize=2,
                color='k',
                label='raw',
            )
            plt.plot(
                x_firing[0],
                y_firing_model.reshape(-1),
                'o-',
                markersize=2,
                color='r',
                label='model',
            )

            plt.legend()
            plt.tight_layout()

            var_counter += 1
            global_var_number += 1

        plt.show()
        
        
def plot_smoothed_temporal_feature(df, column, sm_handler, kernel_h_length):
    event = df[column].values

    # Retrieve the B-spline convolved with the "event" variable
    convolved_ev = sm_handler[column].X.toarray()

    # Retrieve the B-spline used for the convolution
    basis = sm_handler[column].basis_kernel.toarray()

    # Get the x values for the 1st subplot
    tps = np.repeat(np.arange(kernel_h_length) - kernel_h_length //
                    2, basis.shape[1]).reshape(basis.shape)

    # Plot the basis & the convolved events
    plt.figure(figsize=(8, 2.5))

    # Plot the basis for the kernel h
    plt.subplot(121)
    plt.title('Kernel Basis')
    plt.plot(tps, basis)
    plt.xlabel('Time Points')

    # Select an interval containing an event
    event_time_points = np.where(event == 1)[0]
    # Select the first event that occurs after 150 time points
    event_time_points = event_time_points[event_time_points > 150]
    if len(event_time_points) == 0:
        raise ValueError('No event found in specified time interval')
    idx0, idx1 = event_time_points[0] - 100, event_time_points[0] + 400

    # Extract the events convolved with each of the B-spline elements
    conv = convolved_ev[idx0:idx1, :]

    # Get the x values for the 2nd subplot
    tps = np.arange(0, idx1 - idx0)  # - 100
    tps = np.repeat(tps, conv.shape[1]).reshape(conv.shape)

    # Plot the convolved events
    plt.subplot(122)
    plt.title('Convolved Events')
    plt.plot(tps, conv)
    plt.title(column)
    plt.vlines(tps[0, 0] + np.where(event[idx0:idx1])
               [0], 0, 1.5, 'k', ls='--', label='Event')
    plt.xlabel('Time Points')
    plt.legend()
    plt.show()
    plt.close()


def plot_smoothed_spatial_feature(df, column, sm_handler):
    # Retrieve the B-spline evaluated at x
    X_1D = sm_handler[column].X.toarray()

    # Get a sorted version of the variable
    column_values = df[column].values
    idx_srt = np.argsort(column_values)
    X_srt = X_1D[idx_srt]

    # Plot the B-spline basis functions
    plt.figure(figsize=(15, 3))
    plt.subplot(131)
    plt.title(column)
    plt.plot(column_values[idx_srt], X_srt)

    # Unordered scatter plot
    plt.subplot(132)
    plt.title('Unordered scatter plot')
    plt.scatter(range(len(column_values)), column_values, s=1)

    # Ordered scatter plot
    plt.subplot(133)
    plt.title('Ordered scatter plot')
    plt.scatter(range(len(column_values)), column_values[idx_srt], s=1)

    plt.show()
    plt.close()
