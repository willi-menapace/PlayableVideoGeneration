from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import gaussian_kde
import seaborn as sns


class DensityPlotter2DMerged:

    @staticmethod
    def plot_density(actions: np.ndarray, vectors: np.ndarray, actions_count: int, output_directory: str, xlim=None, ylim=None, axis_inversion=False, axes=[0, 1], prefix=""):
        '''
        Computes statistics about the vectors associated with each action

        :param actions: (...) array with actions in [0, actions_count)
        :param vectors: (..., vector_size) array with vectors corresponding to actions
        :param actions_count: the number of actions
        :param output_directory: the directory where to output the plots
        :param xlim: x limits
        :param ylim: y limits
        :param axis_inversion: If True, activate plotting mode for BAIR dataset
        :param axes: the axes to plot
        :param prefix: string with which to start the name of each file
        '''

        vector_size = vectors.shape[-1]

        actions = np.reshape(actions, (-1))
        vectors = np.reshape(vectors, (-1, vector_size))
        vectors = vectors[:, axes]

        if xlim is None:
            xlim = (float(np.min(vectors[:, 0])), float(np.max(vectors[:, 0])))
        if ylim is None:
            ylim = (float(np.min(vectors[:, 1])), float(np.max(vectors[:, 1])))

        current_vectors = np.reshape(vectors, (-1, 2))

        sns.set_style("white")
        plt.plot([0, 0], ylim, 'k', linewidth=0.5)
        plt.plot(xlim, [0, 0], 'k', linewidth=0.5)

        if axis_inversion:
            x_data = current_vectors[:, 1]
            y_data = -current_vectors[:, 0]
        else:
            x_data = current_vectors[:, 0]
            y_data = -current_vectors[:, 1]

        plt.scatter(x_data, y_data, c=actions, cmap=plt.cm.get_cmap("jet", actions_count), marker='.', alpha=0.25)
        plt.colorbar(ticks=range(actions_count))
        plt.ylim(ylim)
        plt.xlim(xlim)

        current_filename = os.path.join(output_directory, f"{prefix}2d_action_density_merged.pdf")
        plt.savefig(current_filename, dpi=600)
        plt.clf()

