import argparse
import importlib
import os

import torch
import torch.nn as nn
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

    datasets = {}

    dataset_splits = DatasetSplitter.generate_splits(config)
    transformations = TransformsGenerator.get_final_transforms(config)

    for key in dataset_splits:
        path, batching_config, split = dataset_splits[key]
        transform = transformations[key]

        datasets[key] = VideoDataset(path, batching_config, transform, split)

    # Creates trainer and evaluator
    trainer = getattr(importlib.import_module(config["training"]["trainer"]), 'trainer')(config, model, datasets["train"], logger)
    # Evaluators will be assigned their specific action samplers to implement the evaluation strategy
    evaluator_inferred_actions = getattr(importlib.import_module(config["evaluation"]["evaluator"]), 'evaluator')(config, datasets["validation"], logger, action_sampler=None, logger_prefix="validation_inferred_actions")
    evaluator_inferred_actions_onehot = getattr(importlib.import_module(config["evaluation"]["evaluator"]), 'evaluator')(config, datasets["validation"], logger, action_sampler=OneHotActionSampler(), logger_prefix="validation_inferred_actions_onehot")
    evaluator_ground_truth_actions = getattr(importlib.import_module(config["evaluation"]["evaluator"]), 'evaluator')(config, datasets["validation"], logger, action_sampler=None, logger_prefix="validation_gt_actions")

    # Resume training
    try:
        trainer.load_checkpoint(model)
    except Exception as e:
        logger.print(e)
        logger.print("- Warning: training without loading saved checkpoint")

    model = nn.DataParallel(model)
    model.cuda()

    logger.get_wandb().watch(model, log='all')

    last_save_step = 0
    last_eval_step = 0

    # Makes the model parallel and train
    while trainer.global_step < config["training"]["max_steps"]:

        model.train()

        trainer.train_epoch(model)

        # Saves the model
        trainer.save_checkpoint(model)
        if trainer.global_step > last_save_step + config["training"]["save_freq"]:
            trainer.save_checkpoint(model, f"checkpoint_{trainer.global_step}")
            last_save_step = trainer.global_step

        model.eval()

        # Evaluates the model
        if trainer.global_step > last_eval_step + config["evaluation"]["eval_freq"]:

            # Evaluates with actions predicted from the model
            evaluator_inferred_actions.evaluate(model, trainer.global_step)
            # Evaluates with actions predicted from the model in one hot version
            # Disabled to improve evaluation time
            #evaluator_inferred_actions_onehot.evaluate(model, trainer.global_step)

            if config["data"]["ground_truth_available"]:
                # Evaluates with ground truth actions translated to the model action space
                # Uses the mapping between inferred and ground truth actions to configure the
                # ground truth action space -> model action space  translation function
                action_mapping = evaluator_inferred_actions.get_best_action_mappings()
                ground_truth_action_sampler = GroundTruthActionSampler(action_mapping)
                evaluator_ground_truth_actions.set_action_sampler(ground_truth_action_sampler)
                evaluator_ground_truth_actions.evaluate(model, trainer.global_step)

            last_eval_step = trainer.global_step

