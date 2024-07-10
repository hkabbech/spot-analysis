import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def run_vac_analysis(track, parms, dtime=1):
    """
    Computes the velocity autocorrelation (VAC)

    Parameters
    ----------
    tracks: pd.DataFrame
        Pandas dataframe containing the coordinates of the track
    parms: dict
        Dictionary containing biophysical parameters
    dtime: int
        Time interval for the calculation

    Returns
    -------
    list
        VAC for for x and y
    """
    time_frame = parms['time_frame']
    vac = []
    xcoord = track['x'].to_numpy()
    vac.append(np.dot(((xcoord[dtime:]-xcoord[:-dtime])/dtime)*time_frame,
                          ((xcoord[dtime]-xcoord[0])/dtime)*time_frame))
    ycoord = track['y'].to_numpy()
    vac.append(np.dot(((ycoord[dtime:]-ycoord[:-dtime])/dtime)*time_frame,
                          ((ycoord[dtime]-ycoord[0])/dtime)*time_frame))
    return vac


def plot_vac_curve(all_vac, parms, result_path):
    """
    Plot the velocity autocorrelation (VAC) over all tracks in the condition

    Parameters
    ----------
    all_vac:list
        The calculated VAC for each track
    parms: dict
        Dictionary containing biophysical parameters
    result_path: Path
        Path to the result directory
    """

    # Adjust all VAC values as a matrix, then get the average
    threshold = max([len(vac) for vac in all_vac])
    vac_matrix = np.zeros((len(all_vac), threshold))
    vac_matrix[:] = np.nan
    for i, _ in enumerate(vac_matrix):
        vac_matrix[i][:len(all_vac[i])] = all_vac[i][:threshold]
    vac_avg = np.nanmean(vac_matrix, axis=0)
    table = pd.DataFrame(vac_matrix)


    fig, axs = plt.subplots(1)
    axs.grid(alpha=0.5)
    # plot theoretical curves:
    xaxis = np.arange(0, 100)
    alphas = np.arange(0.1, 1.3, 0.1)
    cmap = cm.get_cmap("viridis_r")
    cmap = cmap(np.linspace(0, 1, len(alphas)))
    theory_prev = None
    for i, alpha_theory in enumerate(alphas[::-1]):
        theory = ((xaxis+1)**(alpha_theory) + abs(xaxis-1)**(alpha_theory) - 2*(xaxis**(alpha_theory)))/2
        if theory_prev is None:
            theory_prev = theory.copy()
        else:
            axs.fill_between(xaxis*parms['time_frame'], theory, theory_prev, facecolor=cmap[i],
                             alpha=1)
            theory_prev = theory.copy()

    x_axis_norm = np.array([i*parms['time_frame'] for i in range(len(vac_avg))])
    axs.plot(x_axis_norm, (vac_avg/vac_avg[0]), 'o-', lw=1, ms=5.5, color='r')
    axs.set_ylabel(r'$\mathrm{C^{(\delta=1)}_{v}(\Delta t)}$'+\
                   r'$\mathrm{/}$ $\mathrm{C^{(\delta=1)}_{v}(0)}$')
    axs.set_xlabel(r'$\mathrm{\Delta t}$' ' $[sec]$')
    axs.set_xlim([0, 10*parms['time_frame']])
    axs.set_xticks(np.arange(0, 10*parms['time_frame']+1, parms['time_frame'])[:11])

    # Plot side colormap:
    thr_curves = plt.contourf([[0, 0], [0, 0]], alphas, cmap='viridis')
    fig.colorbar(thr_curves, ax=axs, ticks=alphas).\
                 set_label(label=r'$\mathrm{\alpha}$ value of theoretical fBm')
    axs.set_title(f'Averaged VAC curve for {result_path.stem} vs. theoretical fBm VAC')
    fig.tight_layout()
    plt.savefig(result_path/f"averaged_vac_theory.png")
    # plt.show()
    plt.close()

    fig, axs = plt.subplots(1)
    first = True
    a = 0
    for vac in all_vac:
        a += 1
        x_axis_norm = np.array([i*parms['time_frame'] for i in range(len(vac))])
        if first:
            axs.plot(x_axis_norm, vac/vac[0], alpha=0.2, lw=0.7, color="C0", label="individual VAC curves")
            first = False
        else:
            axs.plot(x_axis_norm, vac/vac[0], alpha=0.2, lw=0.7, color="C0")
    print(a)
    x_axis_norm = np.array([i*parms['time_frame'] for i in range(len(vac_avg))])
    axs.plot(x_axis_norm, (vac_avg/vac_avg[0]), '-', lw=1.5, ms=8, color='r', label="averaged VAC curve")
    axs.grid(alpha=0.5)
    axs.legend(loc="upper right")
    axs.set_ylabel(r'$\mathrm{C^{(\delta=1)}_{v}(\Delta t)}$'+\
                   r'$\mathrm{/}$ $\mathrm{C^{(\delta=1)}_{v}(0)}$')
    axs.set_xlabel(r'$\mathrm{\Delta t}$' ' $[sec]$')
    axs.set_title(f'Individual and averaged VAC curves for {result_path.stem}')
    axs.set_xlim([0, 10*parms['time_frame']])
    axs.set_xticks(np.arange(0, 10*parms['time_frame']+1, parms['time_frame'])[:11])
    axs.set_ylim([-6, 6])
    fig.tight_layout()
    plt.savefig(result_path/f"individual_averaged_vac.png")
    # plt.show()
    plt.close()

    table.loc['mean'] = table.mean()
    print(table.loc['mean'][0])
    table.loc['mean'] = table.loc['mean']/table.loc['mean'][0]
    table.to_csv(result_path/"vac_curves.csv")
