from typing import Tuple, List

import torch
import torch.nn as nn

import random

from model.layers.centroid_estimator import CentroidEstimator

from model.main_model.action_network import ActionNetwork
from model.main_model.rendering_network import RenderingNetwork
from model.main_model.conv_dynamics_network import ConvDynamicsNetwork
from model.layers.gumbel_softmax import GumbelSoftmax
from model.main_model.representation_network import RepresentationNetwork
from utils.tensor_displayer import TensorDisplayer
from utils.tensor_folder import TensorFolder


class Model(nn.Module):
    '''
    The module enclosing all major method components
    '''

    def __init__(self, config):
        super(Model, self).__init__()

        self.config = config
        self.action_network_ensable_size = self.config["model"]["action_network"]["ensamble_size"]
        self.random_noise_size = config["model"]["dynamics_network"]["random_noise_size"]

        self.training_observation_stacking = self.config["training"]["batching"]["observation_stacking"]
        self.use_ground_truth_actions = self.config["training"]["use_ground_truth_actions"]
        self.pretraining_detach = self.config["training"]["pretraining_detach"]
        self.actions_count = self.config["data"]["actions_count"]

        self.state_features = config["model"]["representation_network"]["state_features"]
        self.state_resolution = config["model"]["representation_network"]["state_resolution"]
        self.hidden_state_size = config["model"]["dynamics_network"]["hidden_state_size"]

        # Converts the state to the hidden state during pretraining
        self.state_to_hidden_state_layer = nn.Sequential(
            nn.Conv2d(self.state_features, self.hidden_state_size, 3, padding=1)
        )

        self.gumbel_softmax = GumbelSoftmax(config["model"]["action_network"]["gumbel_temperature"], config["model"]["action_network"]["hard_gumbel"])

        self.action_network = nn.ModuleList([ActionNetwork(config) for _ in range(self.action_network_ensable_size)])
        self.dynamics_network = ConvDynamicsNetwork(config)
        self.representation_network = RepresentationNetwork(config)
        self.rendering_network = RenderingNetwork(config)
        self.centroid_estimator = CentroidEstimator(config["data"]["actions_count"],
                                                    config["model"]["action_network"]["action_space_dimension"],
                                                    config["model"]["centroid_estimator"]["alpha"])

        self.train_forward_counts = 0

    def forward(self, batch_tuple: Tuple[torch.Tensor], ground_truth_observations_init=0,
                pretraining=False, gumbel_temperature=None, action_sampler=None, action_variation_sampler=None):
        '''

        :param batch_tuple: (observations, actions, rewards, dones) input batch.
        :param ground_truth_observations_init: number of ground truth observations to use as sequence input instead of
                                               reconstructed observations. Must be specified when not pretraining
        :param pretraining: forwards in pretraining mode
        :param gumbel_temperature: temperature at which to make the action sampling operate
        :param action_sampler: ActionSampler to use for action sampling
                       May be non differentiable
        :param action_variation_sampler: ActionVariationSampler to use for action variation sampling
                                         May be non differentiable
        :return: refer to forward_full_model and forward_pretraining
        '''

        if pretraining:
            return self.forward_pretraining(batch_tuple, gumbel_temperature=gumbel_temperature,
                                            action_sampler=action_sampler,
                                            action_variation_sampler=action_variation_sampler)
        else:
            if ground_truth_observations_init <= 0:
                raise Exception("To forward the full model specify a number of ground truth observations > 0")
            return self.forward_full_model(batch_tuple, ground_truth_observations_init,
                                           gumbel_temperature=gumbel_temperature, action_sampler=action_sampler,
                                           action_variation_sampler=action_variation_sampler)

    def forward_full_model(self, batch_tuple: Tuple[torch.Tensor], ground_truth_observations_init: int, gumbel_temperature=None, action_sampler=None, action_variation_sampler=None) -> Tuple:
        '''
        Computes a forward pass on the complete model

        :param batch_tuple: (observations, actions, rewards, dones) input batch.
        :param ground_truth_observations_init: number of ground truth observations to use as sequence input instead of
                                               reconstructed observations
        :param gumbel_temperature: temperature at which to make the action sampling operate
        :param action_sampler: ActionSampler to use for action sampling
                               May be non differentiable
        :param action_variation_sampler: ActionVariationSampler to use for action variation sampling
                                 May be non differentiable

        :return: (reconstructed_observations, reconstructed_states, states, hidden_states, selected_actions, actions_distributions, ...)

                 reconstructed_observations (bs, observations_count - 1, 3, height, width) reconstructed observations
                                            produced by the rendering network
                 multiresolution_reconstructed_observations [(bs, observations_count - 1, 3, height/2^i, width/2^i) for i in num_resolutions] reconstructed observations
                                            produced by the rendering network at multiple resolutions
                 reconstructed_states (bs, observations_count, state_features, state_height, state_width) representations of the sequence
                                      given by the first real_observations_init ground truth frames and the remaining
                                      reconstructed observations produced by the rendering network
                 states (bs, observations_count, state_features, state_height, state_width) representations of the ground truth observations
                 hidden_states (bs, observations_count - 1, hidden_state_size, state_height, state_width) sequence of hidden frames produced by the
                               dynamics network, excluding the initial hidden state
                 selected_actions (bs, observations_count - 1) tensor of integers
                 actions_distributions (bs, observations_count - 1, actions_count) tensor with logits actions distributions
                 sampled_actions_distributions (bs, observations_count - 1, actions_count) tensor with probabilities representing the action samples
                 attention (bs, observations_count, 1, state_height, state_width) representation the attention map for the ground truth states
                 reconstructed_attention (bs, observations_count - 1, 1, state_height, state_width) representation the attention map for the reconstructed states
                 action_directions_distribution (bs, observations_count - 1, 2, action_space_dimension) mean and variance of each action direction
                 sampled_action_directions (bs, observations_count - 1, action_space_dimension) the sampled action directions
                 action_states_distribution (bs, observations_count, 2, action_space_dimension) mean and variance of each action state
                 sampled_action_states (bs, observations_count, action_space_dimension) the sampled action states
                 action_variations (bs, observations_count - 1, action_space_dimension) the action variations
                 reconstructed_action_distribution see actions_distributions
                 reconstructed_action_directions_distribution see corresponding non reconstructed version
                 reconstructed_sampled_action_directions see corresponding non reconstructed version
                 reconstructed_action_states_distribution see corresponding non reconstructed version
                 reconstructed_sampled_action_states see corresponding non reconstructed version
        '''

        observations, actions, rewards, dones = batch_tuple

        # Flattens the observations
        observations_folded_shape = tuple(observations.size())
        batch_size = observations_folded_shape[0]
        observations_count = observations_folded_shape[1]
        observation_features = observations_folded_shape[2]
        observation_height = observations_folded_shape[3]
        observation_width = observations_folded_shape[4]
        observations_flat_shape = (-1, observation_features, observation_height, observation_width)
        flat_observations = torch.reshape(observations, observations_flat_shape)

        # Obtains states and attention from the ground truth observations
        states_flat, attention_flat = self.representation_network(flat_observations)

        # Folds the states and the attention
        states_flat_shape = tuple(states_flat.size())
        states_features = states_flat_shape[1]
        states_height = states_flat_shape[2]
        states_width = states_flat_shape[3]
        states_folded_shape = (batch_size, observations_count, states_features, states_height, states_width)
        attention_folded_shape = (batch_size, observations_count, 1, states_height, states_width)
        folded_states = torch.reshape(states_flat, states_folded_shape)
        folded_attention = torch.reshape(attention_flat, attention_folded_shape)

        # Computes actions between states
        sampled_action_network = random.choice(self.action_network)

        # Computes actions
        action_network_results = sampled_action_network(folded_states, folded_attention)

        folded_action_logits, folded_action_directions_distribution, folded_sampled_action_directions, \
        folded_action_states_distribution, folded_sampled_action_states = action_network_results

        # Flattens tensors
        flat_action_logits = TensorFolder.flatten(folded_action_logits)
        flat_action_directions_distribution = TensorFolder.flatten(folded_action_directions_distribution)
        flat_sampled_action_directions = TensorFolder.flatten(folded_sampled_action_directions)

        # Transforms logits into probabilities
        flat_action_log_probabilities = torch.log_softmax(flat_action_logits, dim=1)
        flat_action_probabilities = torch.softmax(flat_action_logits, dim=1)

        # Updates the centroids of each action cluster
        self.centroid_estimator.update_centroids(flat_action_directions_distribution, flat_action_probabilities)

        # If an explicit action sampler is specified we use the sampler
        if action_sampler is not None:
            flat_action_samples = action_sampler(flat_action_log_probabilities, actions[:, :-1].reshape((-1,)))
        # If gumbel sampling is requested use that, otherwise use the probability distribution
        elif self.config["model"]["action_network"]["use_gumbel"]:
            flat_action_samples = self.gumbel_softmax(flat_action_log_probabilities, temperature=gumbel_temperature)
        else:
            flat_action_samples = torch.softmax(flat_action_logits, dim=1)

        # Using ground truth would produce meaningless variation vectors, so we forbid it
        if self.use_ground_truth_actions:
            raise Exception("The use of ground truth actions during training is not supported by the selected model")

        # Computes the variations of each action using the sampled directions
        flat_action_variations = self.centroid_estimator.compute_variations(flat_sampled_action_directions, flat_action_samples)

        if not self.config["model"]["action_network"]["use_variations"]:
            flat_action_variations = flat_action_variations * 0

        if action_variation_sampler is not None:
            flat_action_variations = action_variation_sampler(flat_sampled_action_directions, flat_action_samples)

        actions_count = flat_action_samples.size(1)
        action_space_dimension = flat_action_variations.size(1)

        # Folds the actions
        folded_action_probabilities = torch.reshape(flat_action_probabilities, (-1, observations_count - 1, actions_count))
        folded_action_log_probabilities = torch.reshape(flat_action_log_probabilities, (-1, observations_count - 1, actions_count))
        folded_action_logits = torch.reshape(flat_action_logits, (-1, observations_count - 1, actions_count))
        folded_action_samples = torch.reshape(flat_action_samples, (-1, observations_count - 1, actions_count))
        folded_action_variations = torch.reshape(flat_action_variations, (-1, observations_count - 1, action_space_dimension))
        selected_actions = torch.argmax(folded_action_samples, dim=2)  # Computes the index of the action that was selected by sampling

        assert (folded_action_log_probabilities.size(1) == folded_states.size(1) - 1)  # Actions have an observation less because they only lie between states, so the last is lost

        # Initializes the dynamics network
        self.dynamics_network.reinit_memory(batch_size)

        all_reconstructed_states = [folded_states[:, 0]]
        all_reconstructed_attention = [folded_attention[:, 0]]
        all_hidden_states = []
        all_reconstructed_observations = []
        all_multiresolution_reconstructed_observations = None
        for observation_idx in range(observations_count - 1):

            # The state to use as input is the last reconstructed one
            current_input_states = all_reconstructed_states[-1]

            current_random_noise = self.generate_noise(batch_size)
            current_actions = folded_action_samples[:, observation_idx]
            current_variations = folded_action_variations[:, observation_idx]

            # Computes the hidden states
            current_hidden_state = self.dynamics_network(current_input_states, current_actions, current_variations, current_random_noise)
            # Computes the reconstructed frames
            current_reconstructed_observation, current_multiresolution_reconstructed_observation = self.rendering_network(current_hidden_state)

            all_hidden_states.append(current_hidden_state)
            all_reconstructed_observations.append(current_reconstructed_observation)
            resolutions_count = len(current_multiresolution_reconstructed_observation)
            # Initializes the multi resolution observation container if needed
            if all_multiresolution_reconstructed_observations is None:
                all_multiresolution_reconstructed_observations = [[] for _ in range(resolutions_count)]
            # Adds each reconstructed observation to the corresponding position in the container
            for idx in range(resolutions_count):
                all_multiresolution_reconstructed_observations[idx].append(current_multiresolution_reconstructed_observation[idx])

            # Obtains the next reconstructed state
            # If it is obtained with a real observation we have already computed it
            if observation_idx + 1 < ground_truth_observations_init:
                current_reconstructed_state = folded_states[:, observation_idx + 1]
                current_reconstructed_attention = folded_attention[:, observation_idx + 1]
            # If the state contains reconstructed observations or a mix of reconstructed and real ones,
            # we sample the corresponding observation and pass it through the representation network
            else:
                # Obtains the observation to use as input for the representation network in order to obtain the state to
                # feed to the next dynamics network step
                current_observation_for_representation = self.compute_current_observation(observation_idx + 1, ground_truth_observations_init,
                                                                                          observations, all_reconstructed_observations)

                #images = current_observation_for_representation.size(1) // 3
                #for image_idx in range(images):
                #    start_channel = image_idx * 3
                #    TensorDisplayer.show_tensor(current_observation_for_representation[0, start_channel:start_channel + 3])

                # Obtains the current reonstructed observation
                current_reconstructed_state, current_reconstructed_attention = self.representation_network(current_observation_for_representation)
            all_reconstructed_states.append(current_reconstructed_state)
            all_reconstructed_attention.append(current_reconstructed_attention)

        # Stacks the sequence of results
        folded_reconstructed_states = torch.stack(all_reconstructed_states, dim=1)
        folded_reconstructed_attention = torch.stack(all_reconstructed_attention[1:], dim=1) # The first attention comes from ground truth, so we do not consider it a reconstruction
        folded_hidden_states = torch.stack(all_hidden_states, dim=1)
        folded_multiresolution_reconstructed_observations = []
        for current_resolution_observations in all_multiresolution_reconstructed_observations: # The will yield the same resolut of folded_reconstructed_observations
            folded_multiresolution_reconstructed_observations.append(torch.stack(current_resolution_observations, dim=1))

        folded_reconstructed_observations = folded_multiresolution_reconstructed_observations[0]
        complete_folded_reconstructed_attention = torch.stack(all_reconstructed_attention, dim=1) # Same as folded_reconstructed_attention but also with the first attention

        # Computes reconstructed actions
        action_network_results = sampled_action_network(folded_reconstructed_states, complete_folded_reconstructed_attention)

        folded_reconstructed_action_logits, folded_reconstructed_action_directions_distribution, folded_reconstructed_sampled_action_directions, \
        folded_reconstructed_action_states_distribution, folded_reconstructed_sampled_action_states = action_network_results

        # Returns the results
        return folded_reconstructed_observations, folded_multiresolution_reconstructed_observations, folded_reconstructed_states, folded_states, folded_hidden_states,\
               selected_actions, folded_action_logits, folded_action_samples, folded_attention, folded_reconstructed_attention, \
               folded_action_directions_distribution, folded_sampled_action_directions, \
               folded_action_states_distribution, folded_sampled_action_states, folded_action_variations,\
               folded_reconstructed_action_logits, \
               folded_reconstructed_action_directions_distribution, folded_reconstructed_sampled_action_directions, \
               folded_reconstructed_action_states_distribution, folded_reconstructed_sampled_action_states, \



    def forward_pretraining(self, batch_tuple: Tuple[torch.Tensor], gumbel_temperature=None, action_sampler=None, action_variation_sampler=None) -> Tuple:
        '''
        Computes a forward pass on the complete model

        :param batch_tuple: (observations, actions, rewards, dones) input batch.
        :param gumbel_temperature: temperature at which to make the action sampling operate
        :param action_sampler: ActionSampler to use for action sampling
                       May be non differentiable
        :param action_variation_sampler: ActionVariationSampler to use for action variation sampling
                                         May be non differentiable

        :return: (reconstructed_observations, states, reconstructed_hidden_states, hidden_states, selected_actions and actions_distributions)

                 reconstructed_observations (bs, observations_count - 1, 3, height, width) reconstructed observations
                                            produced by the rendering network
                 multiresolution_reconstructed_observations [(bs, observations_count - 1, 3, height/2^i, width/2^i) for i in num_resolutions] reconstructed observations
                                            produced by the rendering network at multiple resolutions
                 reconstructed_states (bs, observations_count, state_features, state_height, state_width) representations of the reconstructed sequence
                 states (bs, observations_count, state_features, state_height, state_width) representations of the ground truth observations
                 reconstructed_hidden_states (bs, observations_count, hidden_state_size, state_height, state_width) sequence of reconstructed
                                             dynamics network hidden states
                 hidden_states (bs, observations_count - 1, hidden_state_size, state_height, state_width) sequence of hidden frames produced by the
                               dynamics network, excluding the initial hidden state
                 selected_actions (bs, observations_count - 1) tensor of integers
                 actions_distributions (bs, observations_count - 1, actions_count) tensor with logits actions distributions
                 sampled_actions_distributions (bs, observations_count - 1, actions_count) tensor with probabilities representing the action samples
                 folded_attention (bs, observations_count, 1, state_height, state_width) representation the attention map for the ground truth states
                 action_directions_distribution (bs, observations_count - 1, 2, action_space_dimension) mean and variance of each action direction
                 sampled_action_directions (bs, observations_count - 1, action_space_dimension) the sampled action directions
                 action_states_distribution (bs, observations_count, 2, action_space_dimension) mean and variance of each action state
                 sampled_action_states (bs, observations_count, action_space_dimension) the sampled action states
                 action_variations (bs, observations_count - 1, action_space_dimension) the action variations
                 reconstructed_action_distribution see actions_distributions
                 reconstructed_action_directions_distribution see corresponding non reconstructed version
                 reconstructed_sampled_action_directions see corresponding non reconstructed version
                 reconstructed_action_states_distribution see corresponding non reconstructed version
                 reconstructed_sampled_action_states see corresponding non reconstructed version
        '''

        observations, actions, rewards, dones = batch_tuple

        # Flattens the observations
        observations_folded_shape = tuple(observations.size())
        batch_size = observations_folded_shape[0]
        observations_count = observations_folded_shape[1]
        observation_features = observations_folded_shape[2]
        observation_height = observations_folded_shape[3]
        observation_width = observations_folded_shape[4]
        observations_flat_shape = (-1, observation_features, observation_height, observation_width)
        flat_observations = torch.reshape(observations, observations_flat_shape)

        # Obtains states and attention from the ground truth observations
        states_flat, attention_flat = self.representation_network(flat_observations)

        # Folds the states and the attention
        states_flat_shape = tuple(states_flat.size())
        states_features = states_flat_shape[1]
        states_height = states_flat_shape[2]
        states_width = states_flat_shape[3]
        states_folded_shape = (batch_size, observations_count, states_features, states_height, states_width)
        attention_folded_shape = (batch_size, observations_count, 1, states_height, states_width)
        folded_states = torch.reshape(states_flat, states_folded_shape)
        folded_attention = torch.reshape(attention_flat, attention_folded_shape)

        if self.pretraining_detach:
            raise Exception("Pretraining detach is not supported by the current model")

        # Computes actions between states
        sampled_action_network = random.choice(self.action_network)

        action_network_results = sampled_action_network(folded_states, folded_attention)

        folded_action_logits, folded_action_directions_distribution, folded_sampled_action_directions, \
        folded_action_states_distribution, folded_sampled_action_states = action_network_results

        # Flattens tensors
        flat_action_logits = TensorFolder.flatten(folded_action_logits)
        flat_action_directions_distribution = TensorFolder.flatten(folded_action_directions_distribution)
        flat_sampled_action_directions = TensorFolder.flatten(folded_sampled_action_directions)

        # Transforms logits into probabilities
        flat_action_log_probabilities = torch.log_softmax(flat_action_logits, dim=1)
        flat_action_probabilities = torch.softmax(flat_action_logits, dim=1)

        # Updates the centroids of each action cluster
        self.centroid_estimator.update_centroids(flat_action_directions_distribution, flat_action_probabilities)

        # If an explicit action sampler is specified we use the sampler
        if action_sampler is not None:
            flat_action_samples = action_sampler(flat_action_log_probabilities, actions[:, :-1].reshape((-1,)))
        # If gumbel sampling is requested use that, otherwise use the probability distribution
        elif self.config["model"]["action_network"]["use_gumbel"]:
            flat_action_samples = self.gumbel_softmax(flat_action_log_probabilities, temperature=gumbel_temperature)
        else:
            flat_action_samples = torch.softmax(flat_action_logits, dim=1)

        # Using ground truth would produce meaningless variation vectors, so we forbid it
        if self.use_ground_truth_actions:
            raise Exception("The use of ground truth actions during training is not supported by the selected model")

        # Computes the variations of each action using the sampled directions
        flat_action_variations = self.centroid_estimator.compute_variations(flat_sampled_action_directions, flat_action_samples)

        if not self.config["model"]["action_network"]["use_variations"]:
            flat_action_variations = flat_action_variations * 0

        if action_variation_sampler is not None:
            flat_action_variations = action_variation_sampler(flat_sampled_action_directions, flat_action_samples)

        actions_count = flat_action_samples.size(1)
        action_space_dimension = flat_action_variations.size(1)

        # Folds the actions
        folded_action_probabilities = torch.reshape(flat_action_probabilities, (-1, observations_count - 1, actions_count))
        folded_action_log_probabilities = torch.reshape(flat_action_log_probabilities, (-1, observations_count - 1, actions_count))
        folded_action_logits = torch.reshape(flat_action_logits, (-1, observations_count - 1, actions_count))
        folded_action_samples = torch.reshape(flat_action_samples, (-1, observations_count - 1, actions_count))
        folded_action_variations = torch.reshape(flat_action_variations, (-1, observations_count - 1, action_space_dimension))
        selected_actions = torch.argmax(folded_action_samples, dim=2)  # Computes the index of the action that was selected by sampling

        assert (folded_action_log_probabilities.size(1) == folded_states.size(1) - 1)  # Actions have an observation less because they only lie between states, so the last is lost

        # Computes the encoded hidden states
        flat_reconstructed_hidden_states = self.state_to_hidden_state_layer(states_flat)
        folded_reconstructed_hidden_states = flat_reconstructed_hidden_states.reshape((batch_size, -1, self.hidden_state_size, self.state_resolution[0], self.state_resolution[1]))

        # Computes the observations
        flat_reconstructed_observations, flat_multiresolution_reconstructed_observations = self.rendering_network(flat_reconstructed_hidden_states)
        folded_multiresolution_reconstructed_observations = [TensorFolder.fold(current_resolution_reconstructed_observations, observations_folded_shape[1])
                                                             for current_resolution_reconstructed_observations in flat_multiresolution_reconstructed_observations]
        folded_reconstructed_observations = folded_multiresolution_reconstructed_observations[0]

        #Initializes the dynamics network
        self.dynamics_network.reinit_memory(batch_size)

        all_hidden_states = []
        for observation_idx in range(observations_count - 1):

            # The state to use as input is the last reconstructed one
            current_input_states = folded_states[:, observation_idx]

            if self.pretraining_detach:
                # Avoids gradients to flow from the dynamics network to the representation network
                current_input_states = current_input_states.detach()

            current_random_noise = self.generate_noise(batch_size)
            current_actions = folded_action_samples[:, observation_idx]
            current_variations = folded_action_variations[:, observation_idx]

            # Computes the hidden states
            current_hidden_state = self.dynamics_network(current_input_states, current_actions, current_variations, current_random_noise)

            all_hidden_states.append(current_hidden_state)


        # Stacks the sequence of results
        folded_hidden_states = torch.stack(all_hidden_states, dim=1)

        # Computes actions on the reconstructed sequences
        folded_stacked_reconstructed_observations = self.compute_stacked_observations(folded_reconstructed_observations)
        flat_stacked_reconstructed_observations = TensorFolder.flatten(folded_stacked_reconstructed_observations)

        flat_reconstructed_states, flat_reconstructed_attention = self.representation_network(flat_stacked_reconstructed_observations)
        folded_reconstructed_states = TensorFolder.fold(flat_reconstructed_states, observations_count)
        folded_reconstructed_attention = TensorFolder.fold(flat_reconstructed_attention, observations_count)

        action_network_reconstructed_results = sampled_action_network(folded_reconstructed_states, folded_reconstructed_attention)

        folded_reconstructed_action_logits, folded_reconstructed_action_directions_distribution, folded_reconstructed_sampled_action_directions, \
        folded_reconstructed_action_states_distribution, folded_reconstructed_sampled_action_states = action_network_reconstructed_results

        # Returns the results
        return folded_reconstructed_observations, folded_multiresolution_reconstructed_observations, folded_reconstructed_states, folded_states, folded_reconstructed_hidden_states, \
               folded_hidden_states, selected_actions, folded_action_logits,  folded_action_samples, folded_attention, \
               folded_action_directions_distribution, folded_sampled_action_directions, \
               folded_action_states_distribution, folded_sampled_action_states, folded_action_variations, \
               folded_reconstructed_action_logits, \
               folded_reconstructed_action_directions_distribution, folded_reconstructed_sampled_action_directions, \
               folded_reconstructed_action_states_distribution, folded_reconstructed_sampled_action_states, \

    def compute_stacked_observations(self, observations: torch.Tensor):
        '''
        Computes the stacked observations starting from non stacked ones

        :param observations: (bs, observations_count, 3, height, width) tensor with observations
        :return: (bs, observations_count, 3 * observations_stacking, height, width) stacked observations
        '''

        sequences_to_stack = [observations]
        for stack_idx in range(1, self.training_observation_stacking):
            repeated_first_observation = observations[:, 0:1].repeat([1, stack_idx, 1, 1, 1])
            subsequent_observations = observations[:, :-stack_idx]
            concatenated_observations = torch.cat([repeated_first_observation, subsequent_observations], dim=1)
            sequences_to_stack.append(concatenated_observations)

        # Concatenates the channels and returns the observations
        return torch.cat(sequences_to_stack, dim=2)

    def generate_noise(self, batch_size: int) -> torch.Tensor:
        '''
        Generates Normal(0, 1) nosie

        :param batch_size: size of the batch to generate
        :return: (batch_size, random_noise_size) tensor with random noise
        '''

        random_noise = torch.randn((batch_size, self.random_noise_size)).cuda()
        return random_noise

    def compute_current_observation(self, idx: int, ground_truth_observations_init: int,
                                    ground_truth_observations: torch.Tensor,
                                    all_reconstructed_observations: List[torch.Tensor]):
        '''
        Computes the current observation to use to obtain the current reconstructed state

        :param idx: index in [1, self.observations_count - 1] indicating the observation to reconstruct
        :param ground_truth_observations_init: number of ground truth observations to use as sequence input instead of
                                               reconstructed observations
        :param ground_truth_observations: (bs, observations_count, 3 * observations_stacking, height, width) ground truth
                                          observations
        :param all_reconstructed_observations: list of (bs, 3, height, width) tensors. Length must be >= idx.
                                               Reconstructed observations must start from sequence element 1, not 0.
        :return: (bs, 3 * observations_stacking, height, width) tensor with the observation to use at idx.
                 frames preceding real_observations_init are taken from the ground truth
        '''

        assert(ground_truth_observations_init > 0)
        assert(len(all_reconstructed_observations) >= idx)

        # If the requested frame is a ground truth one return it
        if idx < ground_truth_observations_init:
            return ground_truth_observations[:, idx]

        observation_frames = []
        end_index = idx
        start_index = idx - self.training_observation_stacking + 1

        # Computes whether we need to mix ground truth and reconstructed frames
        if start_index < ground_truth_observations_init:
            # Computes the number of channels to take from the last ground truth observation
            channels_to_sample = (ground_truth_observations_init - start_index) * 3
            # The extracted channels go from the most recent frame to the oldest
            ground_truth_observation_portion = ground_truth_observations[:, ground_truth_observations_init - 1, :channels_to_sample]
            observation_frames.append(ground_truth_observation_portion)

        for current_frame_index in range(max(start_index, ground_truth_observations_init), end_index + 1):
            # Computes the corresponding index in the reconstructed observations index which starts with sample 1 instead of 0
            translated_current_frame_index = current_frame_index - 1
            # Since channels must go from most recent frame to the oldest, insert the current frame at the beginning
            observation_frames.insert(0, all_reconstructed_observations[translated_current_frame_index])

        # Concatenates the frames on the channels
        current_reconstructed_observation = torch.cat(observation_frames, dim=1)
        return current_reconstructed_observation

    def actions_one_hot(self, actions: torch.Tensor):
        '''
        Encodes the actions into a one hot vector

        :param actions: (bs) tensor with integer representing actions
        :return: (bs, actions_count) tensor with one hot encoded actions
        '''
        batch_size = actions.size(0)

        actions_onehot = torch.zeros((batch_size, self.actions_count), dtype=torch.float).cuda()

        actions_onehot.zero_()
        actions_onehot.scatter_(1, actions.reshape((-1, 1)).type(torch.LongTensor).cuda(), 1)

        return actions_onehot

    def start_inference(self):
        '''
        Initializes the network for a new inference sequence
        :return:
        '''

        # Initializes the dynamics network memory
        self.dynamics_network.reinit_memory(batch_size=1)

    def generate_next(self, observation: torch.Tensor, action: int, noise=False):
        '''
        Gets the observation starting from the current one and the selected action
        :param observation (observation_stacking * 3, height, width) tensor with the current observation
        :param action: integer in [0, actions_count - 1] representing the action to perform
        :param noise: True if noise should be added to action variations, if False they are posed to 0
        :return (3, height, width), (observation_stacking * 3, height, width) with the current frame and the observation
                to use for the next step
        '''

        observation_batch = observation.unsqueeze(dim=0)

        # Encodes the action into a one hot tensor
        actions_count = self.config["data"]["actions_count"]
        action_space_dimension = self.config["model"]["action_network"]["action_space_dimension"]

        actions_batch = torch.zeros((1, actions_count), dtype=torch.float32).cuda()
        actions_batch[0, action] = 1.0

        if noise:
            action_variation_batch = torch.randn((1, action_space_dimension), dtype=torch.float32).cuda()
        else:
            action_variation_batch = torch.zeros((1, action_space_dimension), dtype=torch.float32).cuda()

        # Obtains the representation of the observation
        state_batch, *other_representation = self.representation_network(observation_batch) # Other may represent attention mechanisms
        noise_batch = self.generate_noise(batch_size=1)
        # Obtains the hidden state of the next frame
        hidden_state_batch = self.dynamics_network(state_batch, actions_batch, action_variation_batch, noise_batch)

        # Renders the next frame
        next_frame, *other_rendering = self.rendering_network(hidden_state_batch)
        next_frame = next_frame.squeeze(dim=0)

        # Builds the next observation by removing the last frame and adding the generated at the front
        next_observation = torch.cat([next_frame, observation[:-3]], dim=0)

        return next_frame, next_observation

    def generate_next_interpolation(self, observation: torch.Tensor, first_action: int, second_action: int, interpolation_factor: float):
        '''
        Gets the observation starting from the current one and the selected action
        :param observation (observation_stacking * 3, height, width) tensor with the current observation
        :param first_action: integer in [0, actions_count - 1] representing the first action
        :param second_action: integer in [0, actions_count - 1] representing the second action
        :param interpolation_factor: float in [0, 1] representing the interpolation of the action to perform between the first and the second
        :return (3, height, width), (observation_stacking * 3, height, width) with the current frame and the observation
                to use for the next step
        '''

        observation_batch = observation.unsqueeze(dim=0)

        # Encodes the action into a one hot tensor
        actions_count = self.config["data"]["actions_count"]
        action_space_dimension = self.config["model"]["action_network"]["action_space_dimension"]

        # Chooses the action that is closer in the interpolation line between centroids
        selected_action = first_action
        if interpolation_factor > 0.5:
            selected_action = second_action

        selected_centroid = self.centroid_estimator.estimated_centroids[selected_action]
        # Finds the point in the space that the interpolation should represent
        first_centroid = self.centroid_estimator.estimated_centroids[first_action]
        second_centroid = self.centroid_estimator.estimated_centroids[second_action]
        interpolated_point = (second_centroid - first_centroid) * interpolation_factor + first_centroid
        action_variation = interpolated_point - selected_centroid

        action_variation_batch = action_variation.unsqueeze(0)
        actions_batch = torch.zeros((1, actions_count), dtype=torch.float32).cuda()
        actions_batch[0, selected_action] = 1.0

        # Obtains the representation of the observation
        state_batch, *other_representation = self.representation_network(observation_batch) # Other may represent attention mechanisms
        noise_batch = self.generate_noise(batch_size=1)
        # Obtains the hidden state of the next frame
        hidden_state_batch = self.dynamics_network(state_batch, actions_batch, action_variation_batch, noise_batch)

        # Renders the next frame
        next_frame, *other_rendering = self.rendering_network(hidden_state_batch)
        next_frame = next_frame.squeeze(dim=0)

        # Builds the next observation by removing the last frame and adding the generated at the front
        next_observation = torch.cat([next_frame, observation[:-3]], dim=0)

        return next_frame, next_observation


def model(config):
    return Model(config)
