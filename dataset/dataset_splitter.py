import glob
import os
from typing import Dict


class DatasetSplitter:
    '''
    Helper class for dataset splitting
    '''

    @staticmethod
    def generate_splits(config) -> Dict:
        '''
        Computes the subsets of directory to include in the train, validation and test splits

        :param config: the configuration file
        :return: dictionary with a list of directories for each split
        '''

        dataset_style = config["data"]["dataset_style"]

        # If the data lies in a single directory that needs to be split
        if dataset_style == "flat":
            directory_contents = list(sorted(os.listdir(config["data"]["data_root"])))
            dataset_splits = config["data"]["dataset_splits"]

            contents_length = len(directory_contents)
            num_train = int(contents_length * dataset_splits[0])
            num_val = int(contents_length * dataset_splits[1])
            num_test = contents_length - num_train - num_val

            path = config["data"]["data_root"]
            return {
                "train": (path, config["training"]["batching"], directory_contents[:num_train]),
                "validation": (path, config["evaluation"]["batching"], directory_contents[num_train:num_train + num_val]),
                "test": (path, config["evaluation"]["batching"], directory_contents[num_train + num_val:])
            }
        elif dataset_style == "splitted":

            base_path = config["data"]["data_root"]
            return {
                "train": (os.path.join(base_path, "train"), config["training"]["batching"], None),
                "validation": (os.path.join(base_path, "val"), config["evaluation"]["batching"], None),
                "test": (os.path.join(base_path, "test"), config["evaluation"]["batching"], None)
            }
        else:
            raise Exception(f"Unknown dataset style '{dataset_style}'")
