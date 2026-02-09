import numpy as np
import matplotlib.pyplot as plt

def plot_linear_tuning(var, beta, tuning_meta):
    cols = tuning_meta['groups'][var]
    w = beta[cols].to_numpy()
    gain = np.exp(w)

    plt.figure()
    plt.plot(gain, marker='o')
    plt.xlabel(f'{var} bin')
    plt.ylabel('Gain (× baseline)')
    plt.title(f'Tuning: {var}')
    plt.axhline(1.0, color='k', ls='--', lw=1)
    plt.show()

def plot_angular_tuning(var, beta, tuning_meta):
    cols = tuning_meta['groups'][var]
    w = beta[cols].to_numpy()
    gain = np.exp(w)

    # wrap for circular plotting
    gain = np.r_[gain, gain[0]]
    theta = np.linspace(-np.pi, np.pi, len(gain))

    plt.figure()
    plt.plot(theta, gain)
    plt.xlabel('Angle (rad)')
    plt.ylabel('Gain (× baseline)')
    plt.title(f'Angular tuning: {var}')
    plt.axhline(1.0, color='k', ls='--', lw=1)
    plt.show()