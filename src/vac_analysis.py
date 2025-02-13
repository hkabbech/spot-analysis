import os
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

def measure_plot_all_vac(condition_dataset_path, result_path, parms):

    delta_list = [2, 8, 16, 32]
    # Measure all VAC:
    all_vac, all_vac_mean = {}, {}
    for filename in sorted(os.listdir(condition_dataset_path)):
        if not filename.endswith(".xml"):
            continue
        coords_filename = condition_dataset_path/filename
        df = pd.read_xml(coords_filename, xpath=".//detection")
        tracks = [df]
        for itrack, track in enumerate(tracks):
            for dtime in delta_list:
                if dtime not in all_vac:
                    all_vac[dtime] = []
                vac = run_vac_analysis(track, parms, dtime=dtime)
                all_vac[dtime].extend(vac)

    # limit vac curves and average:
    for dtime in delta_list:
        m = 1000
        a = np.zeros((len(all_vac[dtime]), m))
        a[:] = np.nan
        for i in range(len(a)):
            a[i][:len(all_vac[dtime][i])] = all_vac[dtime][i][:m]
        all_vac_mean[dtime] = np.nanmean(a, axis=0)


    plot_vac_curve(all_vac_mean, delta_list, result_path)


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


def plot_vac_curve(all_vac_mean, delta_list, result_path):
    """
    Plot the velocity autocorrelation (VAC) over all tracks in the condition
    """
    cmap = matplotlib.cm.get_cmap("viridis_r")
    cmap = cmap(np.linspace(0, 1, len(delta_list)))

    fig, axs = plt.subplots(1, figsize=(5, 5))
    for i, delta in enumerate(delta_list):
        x_axis_norm = np.array([i for i in range(len(all_vac_mean[delta]))])
        axs.plot(x_axis_norm, (all_vac_mean[delta]/all_vac_mean[delta][0]), 'o-', color=cmap[i], lw=1, ms=5.5, label=r"$\mathrm{\delta}$="+f"{delta} s")
    axs.set_ylabel('VAC')
    axs.set_xlabel(r'$\mathrm{\Delta t}$ $[s]$')
    axs.set_xlim([-1, 40])
    axs.set_xticks(np.array(delta_list))
    axs.set_ylim([-0.4, 1.1])
    axs.legend(loc='upper right')
    # CS3 = plt.contourf([[0,0],[0,0]], delta_list, cmap='viridis')
    # fig.colorbar(CS3, ax=axs,ticks=delta_list).set_label(label=r'...')
    axs.grid(alpha=0.5)
    axs.set_title('Velocity autocorrelation curves')
    fig.tight_layout()
    plt.savefig(result_path / "vac.png")
    plt.close()

    fig, axs = plt.subplots(1, figsize=(5, 5))
    for i, delta in enumerate(delta_list):
        x_axis_norm = np.array([i for i in range(len(all_vac_mean[delta]))])
        axs.plot(x_axis_norm/delta, (all_vac_mean[delta]/all_vac_mean[delta][0]), 'o-', color=cmap[i], lw=1, ms=3, label=r"$\mathrm{\delta}$="+f"{delta} s")
    axs.set_ylabel('VAC')
    axs.set_xlabel(r'$\mathrm{\Delta t}$ $[s]$ / $\mathrm{\delta}$')
    axs.set_xlim([-0.1, 4])
    # axs.set_xticks(np.array(delta_list))
    axs.set_ylim([-0.4, 1.1])
    axs.legend(loc='upper right')
    axs.grid(alpha=0.5)
    axs.set_title('Velocity autocorrelation curves with time rescaled')
    fig.tight_layout()
    plt.savefig(result_path / "vac_rescaled.png")
    plt.close()

