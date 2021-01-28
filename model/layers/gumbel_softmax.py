import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GumbelSoftmax(nn.Module):
    '''
    Module for gumbel sampling
    '''

    def __init__(self, initial_temperature, hard=True):
        '''
        Initializes the samples to operate at the given temperature
        :param initial_temperature: initial temperature at which to make the sampler operate.
                            temperatures close to 0 produce one hot samples, high temperatures approach uniform sampling
        :param hard: if true uses the hard straight through gumbel implementation
        '''
        super(GumbelSoftmax, self).__init__()

        self.current_temperature = initial_temperature
        self.hard = hard


    def sample_gumbel(self, shape, eps=1e-20):
        '''
        Samples gumbel variable with given shape
        :param shape: shape of the variable to output
        :param eps: constant for numeric stability
        :return: (*shape) tensor with gumbel samples
        '''

        U = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_soft_sample(self, input):
        '''
        Computes soft gumbel samples

        :param input: (bs, classes_count) tensor representing log of probabilities
        :return: (bs, classes_count) soft samples
        '''

        y = input + self.sample_gumbel(input.size())
        return F.softmax(y / self.current_temperature, dim=-1)

    def forward(self, input, temperature=None):
        '''

        :param input: (bs, classes_count) tensor representing log of probabilities
        :param temperature: new temperature at which to make the sampler operate.
                            temperatures close to 0 produce one hot samples, high temperatures approach uniform sampling
        :return:
        '''

        if temperature is not None:
            self.current_temperature = temperature

        # Computes soft samples
        soft_samples = self.gumbel_soft_sample(input)

        if self.hard:
            # Performs hard sampling
            shape = soft_samples.size()
            _, ind = soft_samples.max(dim=-1)
            y_hard = torch.zeros_like(soft_samples).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            hard_samples = (y_hard - soft_samples).detach() + soft_samples # Uses y_hard as output but with this detach trick, we use the gradients from y only, the non hard samples
            return hard_samples

        return soft_samples

if __name__ == '__main__':
    import math
    tens = Variable(torch.cuda.FloatTensor([[math.log(0.1), math.log(0.4), math.log(0.3), math.log(0.2)]] * 100000))

    gumbel_softmax = GumbelSoftmax(1.0)
    samples_sum = gumbel_softmax(tens).sum(dim=0)
    print(samples_sum)