import argparse
import importlib
import os

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
from utils.logger import Logger

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":


    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    arguments = parser.parse_args()

    config_path = arguments.config

    configuration = Configuration(config_path)
    configuration.check_config()
    configuration.create_directory_structure()

    config = configuration.get_config()

    logger = Logger(config)
    search_name = config["model"]["architecture"]
    model = getattr(importlib.import_module(search_name), 'model')(config)
    model.cuda()

    logger.get_wandb().watch(model, log='all')

    datasets = {}

    dataset_splits = DatasetSplitter.generate_splits(config)
    transformations = TransformsGenerator.get_final_transforms(config)

    for key in dataset_splits:
        path, batching_config, split = dataset_splits[key]
        transform = transformations[key]

        datasets[key] = VideoDataset(path, batching_config, transform, split)

    # Creates trainer and evaluator
    trainer = getattr(importlib.import_module(config["training"]["trainer"]), 'trainer')(config, model, datasets["train"], logger)
    # Creates evaluation dataset builder
    evaluation_dataset_builder = getattr(importlib.import_module(config["evaluation_dataset"]["builder"]), 'builder')(
                                 config, datasets["test"], logger)

    # Resume training
    try:
        trainer.load_checkpoint(model)
    except Exception as e:
        logger.print(e)
        #raise Exception("Cannot find checkpoint to load")

    model.eval()
    evaluation_dataset_builder.build(model)
