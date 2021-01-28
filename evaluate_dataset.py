import argparse
import importlib
import os
import yaml

import torch
import torchvision
import numpy as np

from dataset.dataset_splitter import DatasetSplitter
from dataset.transforms import TransformsGenerator
from dataset.video_dataset import VideoDataset
from evaluation.action_sampler import OneHotActionSampler, GroundTruthActionSampler
from evaluation.evaluator import Evaluator
from training.trainer import Trainer
from utils.configuration import Configuration
from utils.evaluation_configuration import EvaluationConfiguration
from utils.logger import Logger

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":


    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    arguments = parser.parse_args()

    config_path = arguments.config

    configuration = EvaluationConfiguration(config_path)
    configuration.check_config()
    configuration.create_directory_structure()

    config = configuration.get_config()

    logger = Logger(config)

    reference_transform, generated_transform = TransformsGenerator.get_evaluation_transforms(config)

    logger.print("- Loading datasets")
    reference_dataset = VideoDataset(config["reference_data"]["data_root"], config["evaluation"]["batching"], reference_transform)
    generated_dataset = VideoDataset(config["generated_data"]["data_root"], config["evaluation"]["batching"], generated_transform)

    logger.print("- Creating evaluator")
    # Creates trainer and evaluator
    evaluator = getattr(importlib.import_module(config["evaluation"]["evaluator"]), 'evaluator')\
                       (config, logger, reference_dataset, generated_dataset)

    logger.print("===== Computing metrics =====")
    metrics = evaluator.compute_metrics()
    logger.print("===== Computing metrics finished =====")
    logger.print(metrics)

    output_file_name = os.path.join(config["logging"]["output_directory"], 'data.yml')
    with open(output_file_name, 'w') as outfile:
        yaml.dump(metrics, outfile)
