from typing import Dict

import torch
import torch.nn as nn
import numpy as np
import PIL.Image as Image
from utils.tensor_displayer import TensorDisplayer


class DetectionMetric2D():

    def __init__(self):
        pass

    def __call__(self, reference_detections: np.ndarray, generated_detections: np.ndarray, prefix: str) -> Dict:
        '''
        Computes the mean squared error between the reference and the generated observations

        :param reference_detections: (sequences_count, observations_count, 2) tensor with detections, -1 if detection is missing
        :param generated_detections: (sequences_count, observations_count, 2) tensor with detections, -1 if detection is missing
        :param prefix: prefix to use in the result dictionary
        :return: dictionary with detection results
        '''

        sequences_count = reference_detections.shape[0]
        sequence_length = reference_detections.shape[1]

        total_frames = sequences_count * sequence_length

        successful_detections = np.zeros((sequence_length), dtype=np.int)
        missed_detections = np.zeros((sequence_length), dtype=np.int)
        center_distances = np.zeros((sequence_length), dtype=np.float)

        # For each sequence compute the statistics
        for sequence_idx in range(sequences_count):
            # For each position in the sequence compute the statistics
            for observation_idx in range(sequence_length):
                reference_successful = reference_detections[sequence_idx, observation_idx, 0] != -1
                generated_successful = generated_detections[sequence_idx, observation_idx, 0] != -1

                if reference_successful and not generated_successful:
                    missed_detections[observation_idx] += 1
                if reference_successful and generated_successful:
                    successful_detections[observation_idx] += 1
                    center_distances[observation_idx] += np.sqrt(((reference_detections[sequence_idx, observation_idx] - generated_detections[sequence_idx, observation_idx]) ** 2).sum())

        results = {}
        for observation_idx in range(sequence_length):
            results[f"{prefix}/center_distance/{observation_idx}"] = float(center_distances[observation_idx] / successful_detections[observation_idx])
            results[f"{prefix}/successful_detections/{observation_idx}"] = int(successful_detections[observation_idx])
            results[f"{prefix}/missed_detections/{observation_idx}"] = int(missed_detections[observation_idx])
            results[f"{prefix}/reference_detections/{observation_idx}"] = int(missed_detections[observation_idx] + successful_detections[observation_idx])

        results[f"{prefix}/center_distance/global"] = float(center_distances.sum() / successful_detections.sum())
        results[f"{prefix}/successful_detections/global"] = int(successful_detections.sum())
        results[f"{prefix}/missed_detections/global"] = int(missed_detections.sum())
        results[f"{prefix}/reference_detections/global"] = int(missed_detections.sum() + successful_detections.sum())

        return results
