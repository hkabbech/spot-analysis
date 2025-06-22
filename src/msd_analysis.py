import os
import sys
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def measure_plot_all_msd(condition_dataset_path, result_path, parms):

    all_results = {"filename": [], "nspot": [], "length": [], "alpha": [], "D": [], "gap_length": []}
    all_msd = []
    # List all files from the given data path and run multiple analysis:
    for filename in sorted(os.listdir(condition_dataset_path)):
        if not filename.endswith(".xml"):
            continue
        coords_filename = condition_dataset_path/filename
        print(f"file: {filename}")

        # # Get dataframe and split it per spot track:
        # df = pd.read_xml(coords_filename, xpath=".//*")[["t", "x", "y"]]
        # nan_indices = df.index[df.isna().all(axis=1)][1:]
        # tracks = [subdf.dropna() for subdf in np.split(df, nan_indices)][1:]


        # nspots = pd.read_xml(filename, xpath="./*")["nSpots"].to_list()
        df = pd.read_xml(coords_filename, xpath=".//detection")
        tracks = [df]

        print("\nMeasure track length:")
        for itrack, track in enumerate(tracks):
            all_results["filename"].append(filename)
            all_results["nspot"].append(itrack+1)
            length = len(track)
            print(f"spot {itrack+1}:")
            print(f"length:\t{length:}")
            all_results["length"].append(length)
            gap_length = track.diff()["t"]
            gap_length = gap_length[gap_length!=1][1:].to_list()
            all_results["gap_length"].append(gap_length)
            print(f"gap length: {gap_length}")


        print("\nRun MSD analysis:")
        results, full_msd = run_msd_analysis(tracks, coords_filename, parms, result_path/"msd_analysis"/"individual_msd_curves")
        all_msd.append(full_msd)
        # Combine all results of the MSD analysis:
        all_results["alpha"].extend(results["alpha"])
        all_results["D"].extend(results["D"])

        print("\n---------------------\n")

    all_results_df = pd.DataFrame.from_dict(all_results)
    all_results_df = pd.concat([all_results_df, all_results_df[["length", "alpha", "D"]].apply(['mean', 'median', 'std', 'sem'])])  # Add mean and median

    print("\n\n*********************\n")
    print(f"Averages:")
    average_estimates = all_results_df[["length", "alpha", "D"]].loc["mean"]
    print(f"length:\t{average_estimates['length']}")
    print(f"alpha:\t{average_estimates['alpha']:.4}")
    print(f"D:\t{average_estimates['D']:.4} um^2/s")

    print("Plot ensemble-averaged MSD")
    plot_ensemble_averaged_msd(all_msd, parms, average_estimates, result_path/"msd_analysis", logscale=False)
    plot_ensemble_averaged_msd(all_msd, parms, average_estimates, result_path/"msd_analysis", logscale=True)


    print("\nEnd.")
    print("\n*********************\n")

    all_results_df.to_csv(result_path/"results_estimates.csv")

def compute_msd(track, size, dim=2):
    """Computes the mean square displacement (MSD) for a given track in order to estimate D and alpha
    using the formula:  log(MSD(dt)) ~ alpha.log(dt) + log(C), with C = 2nD.

    Parameters
    ----------
    track: pd.DataFrame
        Dataframe containing a trajectory's coordinates.
    size: int
       Number of delta time points to use for the MSD curve fit.
    dim: int
        Dimentionality of the track [Defaults: 2].

    Returns
    -------
    (np.array, np.array, float, float, float)
        (msd values, time axis, measured alpha (slope), measured log_C (intercept), resulting diffusion)
    """
    def f_slope_intercept(x_val, a_val, b_val):
        """Linear regression y = ax + b."""
        return a_val*x_val + b_val
    if size <= 2:
        size = 3
    coords = {'x': track['x'].to_numpy(), 'y': track['y'].to_numpy()}
    delta_array = np.arange(1, size+1)
    msd = np.zeros(delta_array.size)
    sigma = np.zeros(delta_array.size)
    for i, delta in enumerate(delta_array):
        if dim == 2:
            x_displ = coords['x'][delta:]-coords['x'][:-delta]
            y_displ = coords['y'][delta:]-coords['y'][:-delta]
            res = abs(x_displ)**2 + abs(y_displ)**2
        elif dim == 1:
            x_displ = coords['x'][delta:]-coords['x'][:-delta]
            res = abs(x_displ)**2
        else:
            print(f'Dimension should be {dim}.')
            return 0
        msd[i] = np.mean(res)
        sigma[i] = np.std(res)
    # popt, _ = curve_fit(f_slope_intercept, np.log(delta_array), np.log(msd))
    popt, _ = curve_fit(f_slope_intercept, np.log(delta_array), np.log(msd), sigma=sigma,
                        absolute_sigma=True)
    alpha = popt[0] # slope
    log_c = popt[1] # intercept
    diffusion = np.exp(log_c)/(2*dim)
    return (msd, delta_array, alpha, log_c, diffusion)


def run_msd_analysis(tracks, filename, parms, result_path):
    """
    Run MSD analysis for a specific file containing trajectory coordinates

    Parameters
    ----------
    tracks: pd.DataFrame
        Pandas dataframe containing the coordinates of the track
    parms: dict
        Dictionary containing biophysical parameters
    result_path: Path
        Path to the result directory
    """
    results = {"alpha": [], "D": []}

    for itrack, track in enumerate(tracks):
        # msd, delta_array, alpha, log_c, diffusion = compute_msd(track, dim=parms["dimension"], size=len(track)//3)
        track_len_ten_perc = len(track)//100*10
        msd, delta_array, alpha, log_c, diffusion = compute_msd(track, dim=parms["dimension"], size=max(10, track_len_ten_perc))
        diffusion_unit = diffusion * (parms['pixel_size']**2/parms['time_frame'])
        deltatime_array = delta_array * parms['time_frame']
        print(f"spot {itrack+1}:")
        print(f"alpha:\t{alpha:.4}")
        print(f"D:\t{diffusion_unit:.4} um^2/s")
        # Save values
        results["alpha"].append(alpha)
        results["D"].append(diffusion_unit)

        # Plot MSD:
        fig, axs = plt.subplots(1)
        axs.plot(deltatime_array, msd*(parms['pixel_size']**2))
        axs.plot(deltatime_array, 2*parms["dimension"]*diffusion*(delta_array**alpha)*(parms['pixel_size']**2),
                    label=rf"$2nDt^\alpha$ with $\alpha=${alpha:0.2}, $D=${diffusion_unit:0.3} $\mu m^2/s$", ls="--", color="r")
        axs.grid(alpha=0.5)
        axs.set_ylabel(r"MSD $[\mu m^2]$")
        axs.set_xlabel(r"$\Delta t$ $[sec]$")
        axs.legend()
        fig.tight_layout()
        filename_plot = filename.stem+f"nspot={itrack}_msd_curve.png"
        plt.savefig(result_path/filename_plot)
        # plt.show()
        plt.close()

        # Plot MSD logscale:
        fig, axs = plt.subplots(1)
        axs.plot(deltatime_array, msd*(parms['pixel_size']**2))
        axs.plot(deltatime_array, 2*parms["dimension"]*diffusion*(delta_array**alpha)*(parms['pixel_size']**2),
                    label=rf"$2nDt^\alpha$ with $\alpha=${alpha:0.2}, $D=${diffusion_unit:0.3} $\mu m^2/s$", ls="--", color="r")
        axs.grid(alpha=0.5)
        axs.set_ylabel(r"MSD $[\mu m^2]$")
        axs.set_xlabel(r"$\mathrm{\Delta t}$ $[sec]$")
        axs.set_yscale('log')
        axs.set_xscale('log')
        axs.legend()
        fig.tight_layout()
        filename_plot = filename.stem+f"nspot={itrack}_msd_curve_logscale.png"
        plt.savefig(result_path/filename_plot)
        # plt.show()
        plt.close()


        full_msd, _, _, _, _ = compute_msd(track, dim=parms["dimension"], size=len(track)-2)
        full_msd = full_msd*parms['pixel_size']**2

    return results, full_msd


def plot_ensemble_averaged_msd(all_msd, parms, average_estimates, result_path, logscale=True):
    """
    Plot the ensemble-averaged MSD curve

    Parameters
    ----------
    all_msd: list
        Result of all msd point values within the same condition
    parms: dict
        Dictionary containing biophysical parameters
    result_path: Path
        Path to the result directory
    average_estimates: pd.DataFrame
        Averaged alpha and D of estimates from individual MSD curves
    logscale: boolean
        Change scale to log if True
    """
    fig, axs = plt.subplots(1)
    first = True
    for msd in all_msd:
        dt_array = np.array([i for i in range(1, len(msd)+1)])*parms['time_frame']
        if first:
            axs.plot(dt_array, msd, alpha=0.2, lw=0.7, color="C0", label="individual MSD curves")
            first = False
        else:
            axs.plot(dt_array, msd, alpha=0.2, lw=0.7, color="C0")

    table = pd.DataFrame(all_msd)

    table_melt = table.melt()
    table_melt['variable'] = ((table_melt['variable']+1)*parms['time_frame'])

    table.loc['mean'] = table.mean()
    table.loc['std'] = table.std()
    table.loc['sem'] = table.sem()
    table.to_csv(result_path/"msd_curves.csv")

    sns.lineplot(data=table_melt, x='variable', y='value', ax=axs, color="black", errorbar=None, label="Ensemble-averaged MSD")

    ## Add line fit from alpha and D averaged:
    alpha = average_estimates["alpha"]
    diffusion_unit = average_estimates["D"]
    diffusion = diffusion_unit / (parms['pixel_size']**2/parms["time_frame"])
    delta_array = np.array([i for i in range(1, 1000)])
    deltatime_array = delta_array * parms['time_frame']
    axs.plot(deltatime_array, 2*parms["dimension"]*diffusion*(delta_array**alpha)*(parms['pixel_size']**2),
            label=rf"$2nDt^\alpha$ with $\alpha=${alpha:0.2}, $D=${diffusion_unit:0.3} $\mu m^2/s$", ls="--", color="r")
    axs.grid(alpha=0.5)

    axs.legend()
    if logscale:
        axs.set_yscale('log')
        axs.set_xscale('log')
    axs.set_ylabel(r'MSD $[\mu m^2]$')
    axs.set_xlabel(r'$\mathrm{\Delta t}$ $[sec]$')
    axs.set_title(f"Individual and ensemble-averaged MSD curves for {result_path.stem}")
    # axs.set_ylim([10**-3, 10**1])
    # axs.set_xlim([parms['time_frame'], 10**1])
    fig.tight_layout()
    plt.savefig(result_path/f"ensemble_averaged_msd_logscale={logscale}.png")
    # plt.show()
    plt.close()

