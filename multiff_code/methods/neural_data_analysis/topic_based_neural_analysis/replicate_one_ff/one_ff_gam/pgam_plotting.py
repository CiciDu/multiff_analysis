
import matplotlib.pyplot as plt

def plot_tuning_sanity(result_row):
    """
    Overlay:
    - conditional tuning
    - marginal tuning
    - kernel (link space)
    """

    x = result_row['x_rate_Hz']
    y_cond = result_row['y_rate_Hz_model']
    xm = result_row['marginal_x_rate_Hz']
    ym = result_row['marginal_y_rate_Hz']
    lo = result_row['marginal_y_rate_Hz_lo']
    hi = result_row['marginal_y_rate_Hz_hi']

    plt.figure(figsize=(5, 4))
    plt.plot(x.flatten(), y_cond.flatten(), 'k', label='conditional')
    plt.plot(xm.flatten(), ym.flatten(), 'r', lw=2, label='marginal')

    if not isinstance(lo, float):
        plt.fill_between(
            xm.flatten(),
            lo.flatten(),
            hi.flatten(),
            color='r',
            alpha=0.25,
            label='marginal CI'
        )

    plt.xlabel(result_row['variable'])
    plt.ylabel('Rate (Hz)')
    plt.legend()
    plt.tight_layout()
    plt.show()