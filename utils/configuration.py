import yaml
import os
from pathlib import Path

from utils.dict_wrapper import DictWrapper


class Configuration:
    '''
    Represents the configuration parameters for running the process
    '''

    def __init__(self, path):
        '''
        Initializes the configuration with contents from the specified file
        :param path: path to the configuration file in json format
        '''

        # Loads configuration file
        with open(path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.config = DictWrapper(config)

    def get_config(self):
        return self.config

    def check_config(self):
        '''
        Raises an exception if the configuration is invalid and creates auxiliary fields
        :return:
        '''

        if not os.path.isdir(self.config["data"]["data_root"]):
            raise Exception(f"Data directory {self.config['data']['data_root']} does not exist")

        self.config["logging"]["output_directory"] = os.path.join(self.config["logging"]["output_root"], self.config["logging"]["run_name"])
        self.config["logging"]["save_root_directory"] = os.path.join(self.config["logging"]["save_root"], self.config["logging"]["run_name"])

        self.config["logging"]["output_images_directory"] = os.path.join(self.config["logging"]["output_directory"], "images")
        self.config["logging"]["amt_sequences"] = os.path.join(self.config["logging"]["output_directory"], "amt_sequences")
        self.config["logging"]["interpolated_sequences"] = os.path.join(self.config["logging"]["output_directory"], "interpolated_sequences")
        self.config["logging"]["evaluation_dataset_directory"] = os.path.join(self.config["logging"]["output_directory"], "evaluation_dataset")
        self.config["logging"]["evaluation_images_directory"] = os.path.join(self.config["logging"]["output_directory"], "evaluation_images")


        # Checks whether it is necessary or not to split the dataset
        if not "dataset_splits" in self.config["data"]:
            self.config["data"]["dataset_style"] = "splitted"
        else:
            self.config["data"]["dataset_style"] = "flat"
            if len(self.config["data"]["dataset_splits"]) != 3:
                raise Exception("Dataset splits must speficy exactly 3 elements")
            if sum(self.config["data"]["dataset_splits"]) != 1.0:
                raise Exception("Dataset splits must sum to 1.0")

        # If crop is not specified set the key anyways
        if not "crop" in self.config["data"]:
            self.config["data"]["crop"] = None

        # If evaluation frequency is not specified set the key to always evaluate after each epoch
        if not "eval_freq" in self.config["evaluation"]:
            self.config["evaluation"]["eval_freq"] = 0

        # By default do not make use of motion to weight loss importance
        if not "use_motion_weights" in self.config["training"]:
            self.config["training"]["use_motion_weights"] = False
        # By default do not make use of bias for the computation of motion weights
        if not "motion_weights_bias" in self.config["training"]:
            self.config["training"]["motion_weights_bias"] = 0.0
        # By default assume ground truth annotations are available
        if not "ground_truth_available" in self.config["data"]:
            self.config["data"]["ground_truth_available"] = True

        # Establish default action space plotting frequency
        if not "action_direction_plotting_freq" in self.config["training"]:
            self.config["training"]["action_direction_plotting_freq"] = 1000
        # Establishes the default lambda to use in the mutual information computation
        if not "action_mutual_information_entropy_lambda" in self.config["training"]:
            self.config["training"]["action_mutual_information_entropy_lambda"] = 1.0
        # Establishes the default lambda to use in the mutual information computation
        if not "max_evaluation_batches" in self.config["evaluation"]:
            self.config["evaluation"]["max_evaluation_batches"] = None

        if not "max_steps_per_epoch" in self.config["training"]:
            self.config["training"]["max_steps_per_epoch"] = 10000

        if self.config["training"]["use_ground_truth_actions"] and not self.config["data"]["ground_truth_available"]:
            raise Exception("Requested to use ground truth data, but no annotations are present in the dataset")

        if not "use_variations" in self.config["model"]["action_network"]:
            self.config["model"]["action_network"]["use_variations"] = True

        return True

    def create_directory_structure(self):
        '''
        Creates directories as required by the configuration file
        :return:
        '''

        Path(self.config["logging"]["output_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["save_root_directory"]).mkdir(parents=True, exist_ok=True)

        Path(self.config["logging"]["output_images_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["amt_sequences"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["interpolated_sequences"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["evaluation_dataset_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["evaluation_images_directory"]).mkdir(parents=True, exist_ok=True)



