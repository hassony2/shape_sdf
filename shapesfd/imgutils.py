from matplotlib import pyplot as plt

def plot_sdf(grid_vals, gt=None, grid_step=1, cmap='seismic'):
    plot_nb = int(grid_vals.shape[0] / grid_step)

    if gt is None:
        row_nb = 1
    else:
        row_nb = 2
    for plot_idx in range(0, plot_nb):
        plt.subplot(row_nb, plot_nb, plot_idx + 1)
        ax = plt.imshow(grid_vals[grid_step * plot_idx], vmin=-2, vmax=1, cmap=cmap)
        plt.colorbar(ax)

    if gt is not None:
        for plot_idx in range(0, plot_nb):
            plt.subplot(row_nb, plot_nb, plot_nb + plot_idx + 1)
            ax = plt.imshow(gt[grid_step * plot_idx], vmin=-2, vmax=1, cmap=cmap)
            plt.colorbar(ax)

