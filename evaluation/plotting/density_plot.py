import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import gaussian_kde


class DensityPlotter:

    @staticmethod
    def plot_density(actions: np.ndarray, vectors: np.ndarray, actions_count: int, output_directory: str, prefix=""):
        '''
        Computes statistics about the vectors associated with each action

        :param actions: (...) array with actions in [0, actions_count)
        :param vectors: (..., vector_size) array with vectors corresponding to actions
        :param actions_count: the number of actions
        :param output_directory: the directory where to output the plots
        :param prefix: string with which to start the name of each file
        '''

        vector_size = vectors.shape[-1]

        actions = np.reshape(actions, (-1))
        vectors = np.reshape(vectors, (-1, vector_size))

        # Extracts the vectors corresponding to each action
        vectors_by_actions = []
        for action_idx in range(actions_count):
            current_vectors = vectors[actions == action_idx, :]
            vectors_by_actions.append(current_vectors)

        for dimension_idx in range(vector_size):
            for action_idx in range(actions_count):
                # If no actions of this category are present, skip it
                if (actions == action_idx).sum() == 0:
                    continue

                current_vectors = vectors_by_actions[action_idx][:, dimension_idx]
                current_vectors = np.reshape(current_vectors, (-1,))

                min_value = np.min(current_vectors)
                max_value = np.max(current_vectors)
                density = gaussian_kde(current_vectors)
                xs = np.linspace(min_value, max_value, 200)
                density.covariance_factor = lambda: .25
                density._compute_covariance()
                plt.plot(xs, density(xs), label=f"Action {action_idx}")
                plt.legend()

            current_filename = os.path.join(output_directory, f"{prefix}action_density_dim_{dimension_idx}.pdf")
            plt.savefig(current_filename, dpi=600)
            plt.close()
