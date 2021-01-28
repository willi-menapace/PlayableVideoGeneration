import yaml
import os
from pathlib import Path

from utils.dict_wrapper import DictWrapper


class EvaluationConfiguration:
    '''
    Represents the configuration parameters for running the evaluation process
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

    def check_data_config(self, data_config):
        if not os.path.isdir(data_config["data_root"]):
            raise Exception(f"Data directory {data_config['data_root']} does not exist")


    def check_config(self):
        '''
        Raises an exception if the configuration is invalid and creates auxiliary fields
        :return:
        '''

        self.check_data_config(self.config["reference_data"])
        self.check_data_config(self.config["generated_data"])

        self.config["logging"]["output_directory"] = os.path.join(self.config["logging"]["output_root"], self.config["logging"]["run_name"])
        self.config["logging"]["output_images_directory"] = os.path.join(self.config["logging"]["output_directory"], "images")
        self.config["logging"]["evaluation_images_directory"] = os.path.join(self.config["logging"]["output_directory"], "evaluation_images")

        return True

    def create_directory_structure(self):
        '''
        Creates directories as required by the configuration file
        :return:
        '''

        Path(self.config["logging"]["output_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["output_images_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["evaluation_images_directory"]).mkdir(parents=True, exist_ok=True)
