import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from src.msd_analysis import measure_plot_all_msd
from src.vac_analysis import measure_plot_all_vac
from src.displ_gaussianity import measure_plot_all_displ

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
    result_path = Path("results")/condition_dataset_path.stem/f"dimension-{int(parms['dimension'])}"
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(result_path/"msd_analysis"/"individual_msd_curves", exist_ok=True)
    os.makedirs(result_path/"vac_analysis", exist_ok=True)
    os.makedirs(result_path/"gaussianity", exist_ok=True)


    print("\nMeasure and plot all displacements (Gaussianity/Laplace)")
    measure_plot_all_displ(condition_dataset_path, result_path/"gaussianity", parms)
    print("\nMeasure and plot all VAC")
    measure_plot_all_vac(condition_dataset_path, result_path/"vac_analysis", parms)
    print("\nRun track measurements and MSD")
    measure_plot_all_msd(condition_dataset_path, result_path, parms)
