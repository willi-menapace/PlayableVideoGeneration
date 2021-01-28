import os

import torch
import torch.nn as nn

from dataset.video_dataset import VideoDataset
from training.losses import SmoothMutualInformationLoss
from training.trainer import Trainer
from utils.logger import Logger


class SmoothMITrainer(Trainer):
    '''
    Helper class for model training
    '''

    def __init__(self, config, model, dataset: VideoDataset, logger: Logger):
        super(SmoothMITrainer, self).__init__(config, model, dataset, logger)

        # Uses smooth mutual information estimation
        self.mutual_information_loss = SmoothMutualInformationLoss(config).cuda()

    def save_checkpoint(self, model, name=None):
        '''
        Saves the current training state
        :param model: the model to save
        :param name: the name to give to the checkopoint. If None the default name is used
        :return:
        '''

        if name is None:
            filename = os.path.join(self.config["logging"]["save_root_directory"], "latest.pth.tar")
        else:
            filename = os.path.join(self.config["logging"]["save_root_directory"], f"{name}_.pth.tar")

        # If the model is wrapped, save the internal state
        is_data_parallel = isinstance(model, nn.DataParallel)
        if is_data_parallel:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        torch.save({"model": model_state_dict, "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(), "mi_estimator": self.mutual_information_loss.state_dict(),
                    "step": self.global_step}, filename)

    def load_checkpoint(self, model, name=None):
        """
        Loads the model from a saved state
        :param model: The model to load
        :param name: Name of the checkpoint to use. If None the default name is used
        :return:
        """

        if name is None:
            filename = os.path.join(self.config["logging"]["save_root_directory"], "latest.pth.tar")
        else:
            filename = os.path.join(self.config["logging"]["save_root_directory"], f"{name}.pth.tar")

        if not os.path.isfile(filename):
            raise Exception(f"Cannot load model: no checkpoint found at '{filename}'")

        loaded_state = torch.load(filename)
        model.load_state_dict(loaded_state["model"])
        self.optimizer.load_state_dict(loaded_state["optimizer"])
        self.lr_scheduler.load_state_dict(loaded_state["lr_scheduler"])
        self.mutual_information_loss.load_state_dict(loaded_state["mi_estimator"])
        self.global_step = loaded_state["step"]


def trainer(config, model, dataset, logger):
    return SmoothMITrainer(config, model, dataset, logger)