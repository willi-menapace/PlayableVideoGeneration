import os
from pathlib import Path

import torch
import numpy as np

from PIL import Image
#from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE

import matplotlib
import matplotlib.pyplot as plt

class TensorDisplayer:

    @staticmethod
    def show_image_tensor(tensor):
        '''
        Displays a given torch tensor

        :param tensor: (3, height, width) tensor to display
        :return:
        '''

        np_tensor = tensor.detach().cpu().numpy()
        np_tensor = (np_tensor + 1.0) / 2.0
        np_tensor = (np.moveaxis(np_tensor, 0, -1) * 255).astype(np.uint8)

        pil_tensor = Image.fromarray(np_tensor)
        pil_tensor.show()

    @staticmethod
    def reduce_dimensionality(features: np.ndarray):
        '''
        Reduces the dimensionality of the features to 2
        :param features: (..., dimensions)
        :return: (..., 2)
        '''

        dimensions = features.shape[1]
        if dimensions != 1:
            features = TSNE(n_jobs=14).fit_transform(features)
        else:
            zeros_copy = np.zeros_like(features)
            features = np.concatenate([features, zeros_copy], axis=1)

        return features

    @staticmethod
    def show_action_directions(estimated_action_centroids: torch.Tensor, action_directions_distribution: torch.Tensor,
                               action_probabilities: torch.Tensor, filename: str) -> torch.Tensor:
        '''
        Produces a plot showing centroids and action directions

        :param estimated_action_centroids (actions_count, action_space_dimension) centroids associated with each action
        :param action_directions_distribution: (..., 2, action_space_dimension) distribution of action directions with mean and variance
        :param action_probabilities: (..., actions_count) action association probabilities in [0, 1]
        :param filename: name for the file to create
        :return:
        '''

        actions_count = estimated_action_centroids.size(0)
        actions_space_dimension = estimated_action_centroids.size(1)

        action_directions_distribution = action_directions_distribution.view((-1, 2, actions_space_dimension))
        action_directions_distribution = action_directions_distribution[:, 0] # Uses the mean

        action_probabilities = action_probabilities.view((-1, actions_count))
        action_labels = torch.argmax(action_probabilities, dim=1).detach().cpu().numpy()

        # Concatenates centroids with the other points and convert to numpy
        all_features = torch.cat([estimated_action_centroids, action_directions_distribution]).detach().cpu().numpy()
        embeddings = TensorDisplayer.reduce_dimensionality(all_features)

        centroid_embeddings = embeddings[:actions_count]
        action_directions_embeddings = embeddings[actions_count:]

        # Plots domains distribution
        vis_x = action_directions_embeddings[:, 0]
        vis_y = action_directions_embeddings[:, 1]

        # Visualize labels
        plt.scatter(vis_x, vis_y, c=action_labels, cmap=plt.cm.get_cmap("jet", actions_count), marker='.', alpha=0.5)
        vis_x = centroid_embeddings[:, 0]
        vis_y = centroid_embeddings[:, 1]
        plt.scatter(vis_x, vis_y, c=list(range(actions_count)), cmap=plt.cm.get_cmap("jet", actions_count), marker='*', s=100, linewidths=0.5, edgecolors=(0, 0, 0))
        plt.colorbar(ticks=range(actions_count))

        plt.savefig(filename)
        plt.close()

    @staticmethod
    def show_action_states(action_states: torch.Tensor, action_probabilities: torch.Tensor, filename: str) -> torch.Tensor:
        '''
        Produces a plot showing action states and trajectories

        :param action_states: (bs, observations_count, action_space_dimension) action state trajectories
                              or (bs, observations_count, 2, action_space_dimension) action state distribution trajectories
        :param action_probabilities: (..., actions_count) action association probabilities in [0, 1]
        :param filename: name for the file to create
        :return:
        '''

        # If the input represents a distribution use only the mean
        if len(list(action_states.size())) == 4:
            action_states = action_states[:, :, 0]

        batch_size = action_states.size(0)
        observations_count = action_states.size(1)
        actions_space_dimension = action_states.size(2)

        # Concatenates centroids with the other points and convert to numpy
        flat_features = action_states.view((-1, actions_space_dimension)).detach().cpu().numpy()
        folded_selected_actions = torch.argmax(action_probabilities, dim=-1).detach().cpu().numpy()
        flat_embeddings = TSNE(n_jobs=14).fit_transform(flat_features)
        folded_embeddings = np.reshape(flat_embeddings, (batch_size, observations_count, 2))

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        colors_count = len(colors)

        for trajectory_idx in range(batch_size):
            current_embeddings = folded_embeddings[trajectory_idx]
            current_actions = folded_selected_actions[trajectory_idx]

            # Plots domains distribution
            vis_x = current_embeddings[:, 0]
            vis_y = current_embeddings[:, 1]

            # Visualize labels
            plt.scatter(vis_x, vis_y, marker='.', alpha=0.5)

            # Plots lines between points color coded according to the action
            for observation_idx in range(observations_count - 1):
                # Selects the color
                current_color = colors[current_actions[observation_idx] % colors_count]
                plt.plot(vis_x[observation_idx:observation_idx + 2], vis_y[observation_idx:observation_idx + 2], color=current_color, linestyle='-', linewidth=0.5)

        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":

    batch_size = 3
    observations_count = 12
    action_space_dimension = 1
    actions_count = 7

    action_states = torch.randn((batch_size, observations_count, action_space_dimension)).cuda()
    action_probabilities = torch.softmax(torch.randn((batch_size, observations_count - 1, actions_count)), dim=1).cuda()

    TensorDisplayer.show_action_states(action_states, action_probabilities, "test.png")
















