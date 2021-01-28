from statistics import mean
from typing import Dict

import torch
import torch.nn as nn
import scipy
import numpy as np
import PIL.Image as Image
from scipy.stats import kurtosis

from utils.tensor_displayer import TensorDisplayer


class ActionVariance:

    def __init__(self):
        pass

    def __call__(self, actions: np.ndarray, vectors: np.ndarray, actions_count: int) -> Dict:
        '''
        Computes statistics about the vectors associated with each action

        :param actions: (...) array with actions in [0, actions_count)
        :param vectors: (..., vector_size) array with vectors corresponding to actions
        :param actions_count: the number of actions
        :return: results dictionary
        '''

        vector_size = vectors.shape[-1]

        actions = np.reshape(actions, (-1))
        vectors = np.reshape(vectors, (-1, vector_size))

        vectors_count = vectors.shape[0]

        all_avg_variances = []
        results = {}
        for action_idx in range(actions_count):
            # If no actions of this category are present, skip it
            if (actions == action_idx).sum() == 0:
                continue

            current_vectors = vectors[actions == action_idx, :]
            current_vectors_count = current_vectors.shape[0]

            mean_vector = np.mean(current_vectors, axis=0)
            variance_vector = np.var(current_vectors, axis=0)
            kurtosis_vector = kurtosis(current_vectors, axis=0)
            avg_variance = np.mean(variance_vector)
            all_avg_variances.append(float(avg_variance))

            results[f"action_variance/mean_vector/{action_idx}"] = mean_vector.tolist()
            results[f"action_variance/kurtosis/{action_idx}"] = kurtosis_vector.tolist()
            results[f"action_variance/quantiles/{action_idx}"] = np.quantile(current_vectors, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], axis=0).tolist()
            results[f"action_variance/variance_vector/{action_idx}"] = variance_vector.tolist()
            results[f"action_variance/avg_variance/{action_idx}"] = float(avg_variance)
            results[f"action_variance/frequency/{action_idx}"] = float(current_vectors_count / vectors_count)

        results[f"action_variance/avg_variance/mean"] = mean(all_avg_variances)

        global_mean_vector = np.mean(vectors, axis=0)
        #global_kurtosis_vector = kurtosis(vectors, axis=0)
        global_variance_vector = np.var(vectors, axis=0)
        global_avg_variance = np.mean(global_variance_vector)

        results[f"action_variance/mean_vector/global"] = global_mean_vector.tolist()
        #results[f"action_variance/kurtosis/global"] = global_kurtosis_vector.tolist()
        results[f"action_variance/quantiles/global"] = np.quantile(vectors, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], axis=0).tolist()
        results[f"action_variance/variance_vector/global"] = global_variance_vector.tolist()
        results[f"action_variance/avg_variance/global"] = float(global_avg_variance)

        return results

if __name__ == "__main__":

    actions = np.asarray([0, 1, 0])
    vectors = np.asarray([
        [3.3, 4.4],
        [1.0, 2.2],
        [3.2, 4.1],
    ])

    action_variance = ActionVariance()
    results = action_variance(actions, vectors, 2)

    print(results)