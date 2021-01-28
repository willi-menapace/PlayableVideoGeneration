import torch


class MotionMaskCalculator:
    '''
    Class for the creation of motion masks
    '''

    def __init__(self):
        pass

    @staticmethod
    def compute_frame_difference_motion_mask(observations, ):
        '''
        :param observations: (bs, observations_count, 3, h, w) the observed sequences

        :return: (bs, observations_count, 1, h, w) tensor with the motion mask
        '''


        sequence_length = observations.size(1)

        # Computes corresponding predecessor and successor observations
        successor_observations = observations[:, 1:]
        predecessor_observations = observations[:, :-1]

        motion_mask = torch.abs(successor_observations - predecessor_observations)

        # Sums the mask along the channel dimension and normalizes it
        assert(motion_mask.size(2) == 3)
        motion_mask = motion_mask.sum(dim=2, keepdim=True) / 3

        # Adds a dummy first sequence element
        motion_mask = torch.cat([torch.zeros_like(motion_mask[:, 0:1]), motion_mask], dim=1)
        return motion_mask