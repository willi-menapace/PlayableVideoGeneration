import torch
import torch.nn as nn

import model
import model.main_model
import model.main_model.model
from model.layers.centroid_estimator import CentroidEstimator

from model.reduced_model.action_network import ActionNetwork
from model.reduced_model.rendering_network import RenderingNetwork
from model.reduced_model.conv_dynamics_network import ConvDynamicsNetwork
from model.reduced_model.representation_network import RepresentationNetwork


class Model(model.main_model.model.Model):
    '''
    A version of the main model with reduced capacity
    '''

    def __init__(self, config):
        super(Model, self).__init__(config)

        self.action_network = nn.ModuleList([ActionNetwork(config) for _ in range(self.action_network_ensable_size)])
        self.dynamics_network = ConvDynamicsNetwork(config)
        self.representation_network = RepresentationNetwork(config)
        self.rendering_network = RenderingNetwork(config)
        self.centroid_estimator = CentroidEstimator(config["data"]["actions_count"],
                                                    config["model"]["action_network"]["action_space_dimension"],
                                                    config["model"]["centroid_estimator"]["alpha"])


def model(config):
    return Model(config)
