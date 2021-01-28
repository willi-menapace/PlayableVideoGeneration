from typing import Dict

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

from utils.tensor_folder import TensorFolder


class InceptionScore:

    def __init__(self):
        self.inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
        self.inception_model.eval();
        self.up = nn.Upsample(size=(299, 299), mode='bilinear').cuda()

    def __call__(self, dataloader, splits=1) -> Dict:
        '''
        Computes the IS between for the given dataset

        :param dataloader: dataloader for observations
        :return dictionary with IS mean and IS std
        '''

        all_preds = []

        for current_batch in dataloader:
            batch_tuple = current_batch.to_tuple()
            observations, _, _, _ = batch_tuple

            # Computes predictions for one sequence at a time
            for sequence_idx in range(observations.size(0)):
                current_observations = observations[sequence_idx]
                current_observations = self.up(current_observations)
                x = self.inception_model(current_observations)
                output = F.softmax(x).data.cpu().numpy()
                all_preds.append(output)

        all_preds = np.concatenate(all_preds, axis=0)

        # Now compute the mean kl-div
        split_scores = []
        samples_count = all_preds.shape[0]

        for k in range(splits):
            part = all_preds[k * (samples_count // splits): (k + 1) * (samples_count // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        results = {
            "is/mean": np.mean(split_scores),
            "is/std": np.std(split_scores)
        }
        return results
