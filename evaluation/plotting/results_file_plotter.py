from math import sqrt
from typing import List

import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class ResultsFilePlotter:

    @staticmethod
    def plot_sequetial_metric(filenames: List[str], metric_name: str, sequences_count: List[int], legend_names: List[str], yname: str, output_filename: str):
        '''
        Plots a sequential metric from the given result files

        :param filenames: list of files from which to load the metrics
        :param metric_name: name of the metric to load
        :param sequences_count: the number of sequences used for the computation of the metrics
        :param legend_names: name to give in the legend to the results of each metric
        :param yname: name to give to the Y axis
        :param output_filename: the name of the output file
        :return:
        '''

        num_files = len(filenames)

        all_values = []
        all_std = []
        for current_results_filename in filenames:
            current_values = {}
            current_variances = {}
            current_max_index = 0
            # Loads configuration file
            with open(current_results_filename) as f:
                current_results = yaml.load(f, Loader=yaml.FullLoader)

            # Extracts all values and variances
            for key, value in current_results.items():
                slash_count = metric_name.count("/")
                if key.startswith(metric_name):
                    try:
                        current_index = int(key.split("/")[1 + slash_count])
                        current_max_index = max(current_max_index, current_index)
                        if len(key.split("/")) == 3 + slash_count and key.split("/")[-1] == "var":
                            current_variances[current_index] = value
                        else:
                            current_values[current_index] = value
                    except:
                        # The current key does not have an index
                        pass

            sorted_values = []
            sorted_std = []
            # Sorts the extracted values and variances
            for current_index in range(current_max_index + 1):
                sorted_values.append(current_values[current_index])
                if len(current_variances) > 0:
                    sorted_std.append(sqrt(current_variances[current_index]))

            all_values.append(sorted_values)
            all_std.append(sorted_std)

        # Plots each line
        for idx in range(num_files):
            indexes = list(range(len(current_values)))
            current_values = all_values[idx]
            current_std = all_std[idx]

            # Plots the values
            last_line = plt.plot(indexes, current_values, label=legend_names[idx])
            color = last_line[0].get_color() # The color of the last drawn line

            # Plots the confidence regions intervals for each position
            if len(current_std) > 0:
                error = 1.96 * np.asarray(current_std) / sqrt(sequences_count[idx])
                plt.fill_between(indexes, current_values - error, current_values + error, color=color, alpha=0.2)

            """for observation_idx in range(len(current_values)):

                # Computes the 0.95 confidence error margin
                error_margin = 1.96 * current_std[observation_idx] / sqrt(sequences_count[idx])
                lower_tick = current_values[observation_idx] - error_margin
                upper_tick = current_values[observation_idx] + error_margin
                plt.plot([observation_idx, observation_idx], [lower_tick, upper_tick], color=color)"""

        plt.grid()
        plt.legend()
        plt.ylabel(yname)
        plt.xlabel("Step")

        plt.savefig(output_filename, dpi=600)
        plt.clf()

'''filenames = [
    "/home/willi/dl/animation/video-generation/evaluation_results/tennis_v4_256_ours_67_300k/data.yml",
    "/home/willi/dl/animation/video-generation/evaluation_results/tennis_v4_128_savp/data.yml",
    "/home/willi/dl/animation/video-generation/evaluation_results/tennis_v4_256_savp/data.yml"
]

sequences_count = [206, 206, 206]

legend_names = [
    "Ours",
    "SAVP",
    "SAVP+"
]

metric = "lpips"
yname = "LPIPS"

output_filename = "/home/willi/dl/animation/video-generation/evaluation_results/tennis_v4_256_ours_67_300k/comparative_lpips.pdf"'''

'''filenames = [
    "/home/willi/dl/animation/video-generation/evaluation_results/tennis_v4_256_ours_67_300k/data.yml",
    "/home/willi/dl/animation/video-generation/evaluation_results/tennis_v4_128_savp/data.yml",
    "/home/willi/dl/animation/video-generation/evaluation_results/tennis_v4_256_savp/data.yml"
]

sequences_count = [206, 206, 206]

legend_names = [
    "Ours",
    "SAVP",
    "SAVP+"
]

metric = "detection/center_distance"
yname = "ADD"

output_filename = "/home/willi/dl/animation/video-generation/evaluation_results/tennis_v4_256_ours_67_300k/comparative_add.pdf"'''

'''filenames = [
    "/home/willi/dl/animation/video-generation/evaluation_results/tennis_v4_256_ours_67_300k/data.yml",
    "/home/willi/dl/animation/video-generation/evaluation_results/tennis_v4_128_savp/data.yml",
    "/home/willi/dl/animation/video-generation/evaluation_results/tennis_v4_256_savp/data.yml"
]

sequences_count = [206, 206, 206]

legend_names = [
    "Ours",
    "SAVP",
    "SAVP+"
]

metric = "detection/missed_detections"
yname = "MDR"

output_filename = "/home/willi/dl/animation/video-generation/evaluation_results/tennis_v4_256_ours_67_300k/comparative_mdr.pdf"'''

'''filenames = [
    "/home/willi/dl/animation/video-generation/evaluation_results/bair_256_ours_03/data.yml",
    "/home/willi/dl/animation/video-generation/evaluation_results/bair_64_savp/data.yml",
    "/home/willi/dl/animation/video-generation/evaluation_results/bair_256_savp/data.yml",
]

sequences_count = [128, 128, 128]

legend_names = [
    "Ours",
    "SAVP",
    "SAVP+"
]

metric = "lpips"
yname = "LPIPS"

output_filename = "/home/willi/dl/animation/video-generation/evaluation_results/bair_256_ours_03/comparative_lpips.pdf"'''

'''filenames = [
    "/home/willi/dl/animation/video-generation/evaluation_results/breakout_v2_160_ours_28/data.yml",
    "/home/willi/dl/animation/video-generation/evaluation_results/breakout_64_savp/data.yml",
    "/home/willi/dl/animation/video-generation/evaluation_results/breakout_160_savp/data.yml"
]

sequences_count = [503, 503, 503]

legend_names = [
    "Ours",
    "SAVP",
    "SAVP+"
]

metric = "detection/center_distance"
yname = "ADD"

output_filename = "/home/willi/dl/animation/video-generation/evaluation_results/breakout_v2_160_ours_28/comparative_add.pdf"'''

'''filenames = [
    "/home/willi/dl/animation/video-generation/evaluation_results/breakout_v2_160_ours_28/data.yml",
    "/home/willi/dl/animation/video-generation/evaluation_results/breakout_64_savp/data.yml",
    "/home/willi/dl/animation/video-generation/evaluation_results/breakout_160_savp/data.yml"
]

sequences_count = [503, 503, 503]

legend_names = [
    "Ours",
    "SAVP",
    "SAVP+"
]

metric = "detection/missed_detections"
yname = "MDR"

output_filename = "/home/willi/dl/animation/video-generation/evaluation_results/breakout_v2_160_ours_28/comparative_mdr.pdf"'''

filenames = [
    "/home/willi/dl/animation/video-generation/evaluation_results/bair_256_ours_03/data.yml",
    "/home/willi/dl/animation/video-generation/evaluation_results/bair_256_ablation_1/data.yml",
    "/home/willi/dl/animation/video-generation/evaluation_results/bair_256_ablation_2/data.yml",
    "/home/willi/dl/animation/video-generation/evaluation_results/bair_256_ablation_3/data.yml",
    "/home/willi/dl/animation/video-generation/evaluation_results/bair_256_ablation_4/data.yml",
]

sequences_count = [206, 206, 206, 206, 206]

legend_names = [
    "Ours",
    "(i)",
    "(ii)",
    "(iii)",
    "(iv)"
]

metric = "lpips"
yname = "LPIPS"

output_filename = "/home/willi/dl/animation/video-generation/evaluation_results/bair_256_ours_03/comparative_ablation_lpips.pdf"

if __name__ == "__main__":

    ResultsFilePlotter.plot_sequetial_metric(filenames, metric, sequences_count, legend_names, yname, output_filename)