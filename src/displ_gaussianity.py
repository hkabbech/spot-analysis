import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, laplace

def measure_plot_all_displ(condition_dataset_path, result_path, parms):

    # Measure all displacements:
    dtime = 1
    all_displ = None
    for filename in sorted(os.listdir(condition_dataset_path)):
        if not filename.endswith(".xml"):
            continue
        coords_filename = condition_dataset_path/filename
        df = pd.read_xml(coords_filename, xpath=".//detection")
        tracks = [df]
        for itrack, track in enumerate(tracks):
            displ_x = track[dtime:].reset_index(drop=True)['x'] - track.reset_index(drop=True)['x']
            displ_x = displ_x.to_numpy().astype(np.float64)
            displ_y = track[dtime:].reset_index(drop=True)['y'] - track.reset_index(drop=True)['y']
            displ_y = displ_y.to_numpy().astype(np.float64)

            if all_displ is None:
                all_displ = {'x': displ_x, 'y': displ_y}
            else:
                all_displ['x'] = np.append(all_displ['x'], displ_x)
                all_displ['y'] = np.append(all_displ['y'], displ_y)

    all_displ['x'] = all_displ['x'][~np.isnan(all_displ['x'])]*parms['pixel_size']
    all_displ['y'] = all_displ['y'][~np.isnan(all_displ['y'])]*parms['pixel_size']
    all_displ_norm = {
        'x': (all_displ['x'] - np.mean(all_displ['x'])) / np.std(all_displ['x']),
        'y': (all_displ['y'] - np.mean(all_displ['y'])) / np.std(all_displ['y'])
    }

    # Plot histogram:
    displ_tmp = np.concatenate((all_displ_norm['x'], all_displ_norm['y']))

    binwidth = 0.2
    bins = np.arange(-4-0.01, 4 + binwidth, binwidth)
    sns.histplot(displ_tmp, bins=bins, stat='density')
    bars = plt.gca().patches[0]
    xy_coords = np.array([[bars.get_x() + (bars.get_width() / 2), bars.get_height()] for bars in plt.gca().patches]).T
    hist = xy_coords
    plt.close()


    fig, axs = plt.subplots(1, figsize=(5, 5))
    axs.grid(alpha=0.5)
    axs.plot(hist[0], hist[1], 'o', color="red", ms=7, lw=1)
    # Generate Gaussian N(0,1) and for comparison
    ls = np.linspace(-4, 4, 500)
    norm_line = norm.pdf(ls, 0, 1)
    axs.plot(ls, norm_line, '-', lw=2.5, color='k', label='Gaussian '+r'$\mathcal{N(0, 1)}$')
    laplace_mu, laplace_sigma = laplace.fit(all_displ_norm['x'].astype(np.float64))
    laplace_line = laplace.pdf(ls, laplace_mu, laplace_sigma)
    axs.plot(ls, laplace_line, '--', lw=2.5, color='k',
             label=f'Laplace '+rf'$\mathcal{{L({round(laplace_mu)}, {laplace_sigma:.2f})}}$')

    axs.set_ylabel('Probability Density')
    axs.set_xlabel(r'$\mathrm{\Delta r}$ $[\mu m]$')
    axs.set_title('Distribution of X and Y displacements')
    axs.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig(result_path / "displacements.png")
    # plt.show()
    plt.close()
