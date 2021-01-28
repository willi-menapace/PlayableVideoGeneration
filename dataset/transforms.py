from typing import Dict, Tuple, List

import random

import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image


class TransformsGenerator:

    @staticmethod
    def check_and_resize(target_crop: List[int], target_size: Tuple[int]):
        '''
        Creates a function that transforms input PIL images to the target size
        :param target_crop: [left_index, upper_index, right_index, lower_index] list representing the crop region
        :param target_size: (width, height) touple representing the target height and width
        :return: function that transforms a PIL image to the target size
        '''

        # Creates the transformation function
        def transform(image: Image):
            if target_crop is not None:
                image = image.crop(target_crop)
            if image.size != tuple(target_size):
                image = image.resize(target_size, Image.BILINEAR)

            return image

        return transform

    @staticmethod
    def to_float_tensor(tensor):
        return tensor / 1.0

    @staticmethod
    def sample_augmentation_transform(batching_config: Dict):
        '''
        Samples an augmenting transformation from PIL.Image to PIL.Image that can be applied to multiple images
        with the same effect
        :param batching_config: Dict with the batching parameters to use for sampling. Must contain rotation_range,
                                scale_range and translation_range
        :return: function from PIL.Image to PIL.Image representing the samples augmentation transformation
        '''

        rotation_range = batching_config["rotation_range"]
        translation_range = batching_config["translation_range"]
        scale_range = batching_config["scale_range"]

        # Samples transformation parameters
        sampled_translation = [random.uniform(*translation_range),
                               random.uniform(*translation_range),
                               ]
        sampled_rotation = random.uniform(*rotation_range)
        sampled_scale = random.uniform(*scale_range)

        # Builds the transformation function
        def composed_transform(img):
            return transforms.functional.affine(img, sampled_rotation, sampled_translation, sampled_scale,
                                                shear=0, resample=Image.BILINEAR, fillcolor=None)

        return composed_transform

    @staticmethod
    def get_evaluation_transforms(config) -> Tuple:
        '''
        Obtains the transformations to use for the evaluation scripts
        :param config: The evaluation configuration file
        :return: reference_transformation, generated transformation to use for the reference and the generated datasets
        '''

        reference_resize_transform = TransformsGenerator.check_and_resize(config["reference_data"]["crop"],
                                                                          config["data"]["target_input_size"])
        generated_resize_transform = TransformsGenerator.check_and_resize(config["generated_data"]["crop"],
                                                                          config["data"]["target_input_size"])

        # Do not normalize data for evaluation
        reference_transform = transforms.Compose([reference_resize_transform,
                                                  transforms.ToTensor(),
                                                  TransformsGenerator.to_float_tensor])
        generated_transform = transforms.Compose([generated_resize_transform,
                                                  transforms.ToTensor(),
                                                  TransformsGenerator.to_float_tensor])

        return reference_transform, generated_transform

    @staticmethod
    def get_final_transforms(config):
        '''
        Obtains the transformations to use for training and evaluation
        :param config: The configuration file
        :return:
        '''

        resize_transform = TransformsGenerator.check_and_resize(config["data"]["crop"],
                                                                config["model"]["representation_network"]["target_input_size"])
        transform = transforms.Compose([resize_transform,
                                        transforms.ToTensor(),
                                        TransformsGenerator.to_float_tensor,
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        return {
            "train": transform,
            "validation": transform,
            "test": transform,
        }