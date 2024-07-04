import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from src.msd_analysis import run_msd_analysis, plot_ensemble_averaged_msd


if __name__ == "__main__":

    # Extract filename and biophysical pparmeters
    condition_dataset_path = Path(sys.argv[1])
    parms_filename = condition_dataset_path/"parms.csv"
    parms_df = pd.read_csv(parms_filename)[["parm", "value"]]
    parms_df["value"] = parms_df["value"].astype(float)
    parms = parms_df.set_index("parm")["value"].to_dict()

    print("\n\n*********************\n")
    print(f"Run spot analysis on {condition_dataset_path}.")
    print(f"params: {parms}")
    print("\n*********************\n")


    # Create result path
    result_path = Path("results")/condition_dataset_path.stem
    os.makedirs(result_path, exist_ok=True)

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
        results, full_msd = run_msd_analysis(tracks, coords_filename, parms, result_path)
        all_msd.append(full_msd)
        # Combine all results of the MSD analysis:
        all_results["alpha"].extend(results["alpha"])
        all_results["D"].extend(results["D"])

        print("\n---------------------\n")

    all_results_df = pd.DataFrame.from_dict(all_results)
    all_results_df = pd.concat([all_results_df, all_results_df[["length", "alpha", "D"]].apply(['mean', "median"])])  # Add mean and median

    print("\n\n*********************\n")
    print(f"Averages:")
    average_estimates = all_results_df[["length", "alpha", "D"]].loc["mean"]
    print(f"length:\t{average_estimates['length']}")
    print(f"alpha:\t{average_estimates['alpha']:.4}")
    print(f"D:\t{average_estimates['D']:.4} um^2/s")

    print("Plot ensemble-averaged MSD")
    plot_ensemble_averaged_msd(all_msd, parms, result_path)

    print("\nEnd.")
    print("\n*********************\n")

    all_results_df.to_csv(result_path/"msd_analysis.csv")
