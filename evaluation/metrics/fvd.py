# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Minimal Reference implementation for the Frechet Video Distance (FVD).
FVD is a metric for the quality of video generation models. It is inspired by
the FID (Frechet Inception Distance) used for images, but uses a different
embedding to be better suitable for videos.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from typing import List

import torch
import six
import tensorflow.compat.v1 as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub

import numpy as np


def preprocess(videos, target_resolution):
  """Runs some preprocessing on the videos for I3D model.
  Args:
    videos: <T>[batch_size, num_frames, height, width, depth] The videos to be
      preprocessed. We don't care about the specific dtype of the videos, it can
      be anything that tf.image.resize_bilinear accepts. Values are expected to
      be in the range 0-255.
    target_resolution: (width, height): target video resolution
  Returns:
    videos: <float32>[batch_size, num_frames, height, width, depth]
  """
  videos_shape = videos.shape.as_list()
  all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
  resized_videos = tf.image.resize_bilinear(all_frames, size=target_resolution)
  target_shape = [videos_shape[0], -1] + list(target_resolution) + [3]
  output_videos = tf.reshape(resized_videos, target_shape)
  scaled_videos = 2. * tf.cast(output_videos, tf.float32) / 255. - 1
  return scaled_videos

def _is_in_graph(tensor_name):
  """Checks whether a given tensor does exists in the graph."""
  try:
    tf.get_default_graph().get_tensor_by_name(tensor_name)
  except KeyError:
    return False
  return True


def create_id3_embedding(videos):
  """Embeds the given videos using the Inflated 3D Convolution network.
  Downloads the graph of the I3D from tf.hub and adds it to the graph on the
  first call.
  Args:
    videos: <float32>[batch_size, num_frames, height=224, width=224, depth=3].
      Expected range is [-1, 1].
  Returns:
    embedding: <float32>[batch_size, embedding_size]. embedding_size depends
               on the model used.
  Raises:
    ValueError: when a provided embedding_layer is not supported.
  """

  batch_size = 16
  module_spec = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"


  # Making sure that we import the graph separately for
  # each different input video tensor.
  module_name = "fvd_kinetics-400_id3_module_" + six.ensure_str(
      videos.name).replace(":", "_")

  assert_ops = [
      tf.Assert(
          tf.reduce_max(videos) <= 1.001,
          ["max value in frame is > 1", videos]),
      tf.Assert(
          tf.reduce_min(videos) >= -1.001,
          ["min value in frame is < -1", videos]),
      tf.assert_equal(
          tf.shape(videos)[0],
          batch_size, ["invalid frame batch size: ",
                       tf.shape(videos)],
          summarize=6),
  ]
  with tf.control_dependencies(assert_ops):
    videos = tf.identity(videos)

  module_scope = "%s_apply_default/" % module_name

  # To check whether the module has already been loaded into the graph, we look
  # for a given tensor name. If this tensor name exists, we assume the function
  # has been called before and the graph was imported. Otherwise we import it.
  # Note: in theory, the tensor could exist, but have wrong shapes.
  # This will happen if create_id3_embedding is called with a frames_placehoder
  # of wrong size/batch size, because even though that will throw a tf.Assert
  # on graph-execution time, it will insert the tensor (with wrong shape) into
  # the graph. This is why we need the following assert.
  video_batch_size = int(videos.shape[0])
  assert video_batch_size in [batch_size, -1, None], "Invalid batch size"
  tensor_name = module_scope + "RGB/inception_i3d/Mean:0"
  if not _is_in_graph(tensor_name):
    i3d_model = hub.Module(module_spec, name=module_name)
    i3d_model(videos)

  # gets the kinetics-i3d-400-logits layer
  tensor_name = module_scope + "RGB/inception_i3d/Mean:0"
  tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
  return tensor

def fake_create_id3_embedding(videos):
  """Embeds the given videos using the Inflated 3D Convolution network.
  Downloads the graph of the I3D from tf.hub and adds it to the graph on the
  first call.
  Args:
    videos: <float32>[batch_size, num_frames, height=224, width=224, depth=3].
      Expected range is [-1, 1].
  Returns:
    embedding: <float32>[batch_size, embedding_size]. embedding_size depends
               on the model used.
  Raises:
    ValueError: when a provided embedding_layer is not supported.
  """

  batch_size = 16

  return tf.zeros((batch_size, 400))

def calculate_fvd(real_activations,
                  generated_activations):
  """Returns a list of ops that compute metrics as funcs of activations.
  Args:
    real_activations: <float32>[num_samples, embedding_size]
    generated_activations: <float32>[num_samples, embedding_size]
  Returns:
    A scalar that contains the requested FVD.
  """
  return tfgan.eval.frechet_classifier_distance_from_activations(
      real_activations, generated_activations)

def extract_embeddings(output_tensor: tf.Tensor, input_placeholder: tf.Tensor, inputs: np.ndarray, sess, batch_size=16) -> np.ndarray:
    '''
    Computes the embedding corresponding to the inputs, dividing the computation in batches
    :param output_tensor: (batch_size, embeddings_size)  tensor representing the output embeddings
    :param input_placeholder: The placeholder to use to hold the inputs
    :param inputs: (sequences, observations_count, height, width, channels) tensor with the inputs
    :param session: the session to use for the computation
    :return: (sequences, embeddings_size) embeddings
    '''

    sequences, sequence_length, height, width, channels = inputs.shape

    all_embeddings = []
    current_idx = 0
    # Extracts the embeddings one batch at a time
    while current_idx + batch_size <= sequences:
        current_embeddings = sess.run(output_tensor, feed_dict={input_placeholder: inputs[current_idx:current_idx + batch_size]})
        all_embeddings.append(current_embeddings)
        current_idx += batch_size

    return np.concatenate(all_embeddings, axis=0)

def cut_to_multiple_size(array: np.ndarray, size: int) -> np.ndarray:
    '''
    Makes sure the number of elements in the first dimension of the array is a multiple of size
    :param array: (size, ...) tensor
    :param size:
    :return:
    '''

    sequences = array.shape[0]
    if sequences % size != 0:
        array = array[0: sequences - (sequences % size)]
    return array


class FVD:

    def __init__(self):
        super(FVD, self).__init__()

    def __call__(self, reference_observations: np.ndarray, generated_observations: np.ndarray) -> float:
        '''
        Computes the FVD between the reference and the generated observations

        :param reference_observations: (bs, observations_count, channels, height, width) tensor with reference observations
        :param generated_observations: (bs, observations_count, channels, height, width) tensor with generated observations
        :return: The FVD between the two distributions
        '''

        # Multiples of 16 must be fes to the embeddings network
        embedding_batch_size = 16

        # Puts the observations in the expected range [0, 255]
        reference_observations = reference_observations * 255
        generated_observations = generated_observations * 255

        # Converts dimensions to tensorflow format by moving channels
        reference_observations = np.moveaxis(reference_observations, 2, -1)
        generated_observations = np.moveaxis(generated_observations, 2, -1)

        # Cuts the sequences to multiples of the batch size
        reference_observations = cut_to_multiple_size(reference_observations, embedding_batch_size)
        generated_observations = cut_to_multiple_size(generated_observations, embedding_batch_size)

        sequences, sequence_length, height, width, channels = reference_observations.shape
        with tf.Graph().as_default():

            # Builds the graph
            input_placeholder = tf.placeholder(tf.float32, [embedding_batch_size, sequence_length, height, width, channels])
            embeddings = create_id3_embedding(preprocess(input_placeholder, (224, 224)))

            reference_embeddings_placeholder = tf.placeholder(tf.float32, [sequences, 400])
            generated_embeddings_placeholder = tf.placeholder(tf.float32, [sequences, 400])

            fvd = calculate_fvd(reference_embeddings_placeholder, generated_embeddings_placeholder)

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())

                # Computes the embeddings
                reference_embeddings = extract_embeddings(embeddings, input_placeholder,
                                                          reference_observations, sess, batch_size=embedding_batch_size)
                generated_embeddings = extract_embeddings(embeddings, input_placeholder,
                                                          generated_observations, sess, batch_size=embedding_batch_size)

                # Computes the fvd
                fvd_np = sess.run(fvd, feed_dict={reference_embeddings_placeholder: reference_embeddings,
                                            generated_embeddings_placeholder: generated_embeddings})

                return float(fvd_np)


class IncrementalFVD:

    def __init__(self):
        super(IncrementalFVD, self).__init__()

    def extract_all_embeddings(self, dataloader, embeddings: tf.Tensor, input_placeholder: tf.Tensor, sess: tf.Session, embedding_batch_size: int):
        '''
        Extracts all the embeddings from the given dataloader

        :param dataloader: The dataloader from which to extract the emebeddings
        :param embeddings: The tensor representing the output embeddings
        :param input_placeholder: The placeholder holding the input
        :param sess: The session on which to run extraction
        :param embedding_batch_size: The batch size to use for each embedding computations
        :return: (samples, embeddings_sim) array with the embeddings
        '''

        iterator = iter(dataloader)
        buffer = []
        current_observations = self.get_next_batch(iterator, buffer, embedding_batch_size)
        all_embeddings = []
        while current_observations is not None:
            current_embeddings = extract_embeddings(embeddings, input_placeholder, current_observations, sess, batch_size=embedding_batch_size)
            current_observations = self.get_next_batch(iterator, buffer, embedding_batch_size)
            all_embeddings.append(current_embeddings)

        current_embeddings = np.concatenate(all_embeddings, axis=0)
        return current_embeddings

    def get_next_batch(self, dataloader_iterator, buffer: List[np.ndarray], target_batch_size: int) -> np.ndarray:
        '''
        Extracts a batch of the given dimension from a dataloader

        :param dataloader_iterator: Iterator over the dataloader
        :param buffer: Buffer where to store the eccess observation in the form
                       (bs, observations_count, channels, height, width)
        :param target_batch_size: The batch size of the desired output tensor
        :return: (target_batch_size, observations_count, height, width, channels) array with the observations
                 None if a batch with the target size could not be built
        '''

        # Creates a buffer where to store excess sequences
        excess_sequence_buffer = []

        # Computes how much sequences we already have in the buffer
        sequences_in_buffer = 0
        if len(buffer) > 0:
            sequences_in_buffer = buffer[0].shape[0]

        remaining_sequences = target_batch_size - sequences_in_buffer

        try:
            while remaining_sequences > 0:
                next_batch = next(dataloader_iterator)
                # Extracts data
                batch_tuple = next_batch.to_tuple()
                observations, _, _, _ = batch_tuple
                observations = observations.cpu().numpy()

                current_sequences = observations.shape[0]
                if current_sequences > remaining_sequences:
                    required_observations = observations[:remaining_sequences]
                    excess_observation = observations[remaining_sequences:]
                    excess_sequence_buffer.append(excess_observation)
                    buffer.append(required_observations)
                    remaining_sequences = 0
                else:
                    buffer.append(observations)
                    remaining_sequences -= current_sequences
        # When the dataloader is empty we have an incomplete batch
        except StopIteration:
            return None

        # If no exception happens we must have a complete batch
        assert(remaining_sequences == 0)

        return_observations = np.concatenate(buffer, axis=0)

        # The buffer now contains the excess observation, if any
        buffer.clear()
        buffer.extend(excess_sequence_buffer)

        # Puts the observatons in numpy format
        return_observations = np.moveaxis(return_observations, 2, -1)
        return_observations = return_observations * 255
        return return_observations


    def __call__(self, reference_dataloader, generated_dataloader) -> float:
        '''
        Computes the FVD between the reference and the generated observations

        :param reference_dataloader: dataloader for reference observations
        :param generated_dataloader: dataloader for generated observations
        :return: The FVD between the two distributions
        '''

        # Multiples of 16 must be fes to the embeddings network
        embedding_batch_size = 16

        # Extracts information about the batch
        sample_observations = next(iter(reference_dataloader)).to_tuple()[0]
        _, sequence_length, channels, height, width = list(sample_observations.size())

        with tf.Graph().as_default():

            # Builds the embedding computation part of the graph
            input_placeholder = tf.placeholder(tf.float32, [embedding_batch_size, sequence_length, height, width, channels])

            embeddings = create_id3_embedding(preprocess(input_placeholder, (224, 224)))

            # Computes all embeddings
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())

                reference_embeddings = self.extract_all_embeddings(reference_dataloader, embeddings, input_placeholder, sess, embedding_batch_size)
                generated_embeddings = self.extract_all_embeddings(generated_dataloader, embeddings, input_placeholder, sess, embedding_batch_size)

                if reference_embeddings.shape[0] != generated_embeddings.shape[0]:
                    raise Exception(f"Size of reference ({reference_embeddings.shape[0]}) and generated embeddings ({generated_embeddings.shape[0]}) differ")

            sequences = reference_embeddings.shape[0]

            # Builds the fvd computation part of the graph
            reference_embeddings_placeholder = tf.placeholder(tf.float32, [sequences, 400])
            generated_embeddings_placeholder = tf.placeholder(tf.float32, [sequences, 400])

            fvd = calculate_fvd(reference_embeddings_placeholder, generated_embeddings_placeholder)

            # Computes the fvd
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())

                # Computes the fvd
                fvd_np = sess.run(fvd, feed_dict={reference_embeddings_placeholder: reference_embeddings, generated_embeddings_placeholder: generated_embeddings})

                return float(fvd_np)


if __name__ == "__main__":
    with tf.Graph().as_default():
        """first_set_of_videos = tf.zeros([16, 30, 64, 64, 3])
        second_set_of_videos = tf.ones([16, 30, 64, 64, 3]) * 255

        result = calculate_fvd(
            create_id3_embedding(preprocess(first_set_of_videos, (224, 224))),
            create_id3_embedding(preprocess(second_set_of_videos, (224, 224))))

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            print("FVD is: %.2f." % sess.run(result))"""

        embedding_batch_size = 16

        reference = np.zeros([256, 30, 64, 64, 3], dtype=np.float)
        generated = np.ones([256, 30, 64, 64, 3], dtype=np.float) * 255

        # Cuts the sequences to multiples of the batch size
        reference = cut_to_multiple_size(reference, embedding_batch_size)
        generated = cut_to_multiple_size(generated, embedding_batch_size)

        sequences, sequence_length, height, width, channels = reference.shape
        with tf.Graph().as_default():
            # Builds the graph
            input_placeholder = tf.placeholder(tf.float32, [embedding_batch_size, sequence_length, height, width, channels])
            embeddings = create_id3_embedding(preprocess(input_placeholder, (224, 224)))

            reference_embeddings_placeholder = tf.placeholder(tf.float32, [sequences, 400])
            generated_embeddings_placeholder = tf.placeholder(tf.float32, [sequences, 400])

            fvd = calculate_fvd(reference_embeddings_placeholder, generated_embeddings_placeholder)

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())

                # Computes the embeddings
                reference_embeddings = extract_embeddings(embeddings, input_placeholder,
                                                          reference, sess, batch_size=embedding_batch_size)
                generated_embeddings = extract_embeddings(embeddings, input_placeholder,
                                                          generated, sess, batch_size=embedding_batch_size)

                # Computes the fvd
                fvd_np = sess.run(fvd, feed_dict={reference_embeddings_placeholder: reference_embeddings,
                                                  generated_embeddings_placeholder: generated_embeddings})

                print(fvd_np)