import torch
import torch.nn as nn


class CentroidEstimator(nn.Module):
    '''
    Estimator for centroid positions
    '''

    def __init__(self, centroids_count: int, space_dimensions: int, alpha: float):
        '''
        Initializes the model

        :param centroids_count: the number of centroids to track
        :param space_dimensions: dimension of the space where the centroids lie
        :param alpha: value to use for the moving average computation
        '''

        super(CentroidEstimator, self).__init__()

        self.centroids_count = centroids_count
        self.alpha = alpha
        self.space_dimensions = space_dimensions

        # Initialies the centroids by sampling them from the reference gaussian distribution
        # Do not allow backpropagation
        initial_centroids = torch.randn((self.centroids_count, self.space_dimensions), dtype=torch.float32)
        self.estimated_centroids = nn.Parameter(initial_centroids, requires_grad=False)

    def get_estimated_centroids(self) -> torch.tensor:
        '''
        Obtains the estimates for the centroids
        :return: (centroids_count, space_dimensions) tensor with estimated centroids
        '''

        return self.estimated_centroids

    def update_centroids(self, points_priors: torch.Tensor, centroid_assignments: torch.Tensor):
        '''

        :param points_priors: (..., 2, space_dimensions) tensor with (mean, variance) for each point
        :param centroid_assignments: (..., centroids_count) tensor with cluster assignment probabilities in 0, 1
                                                            for each point
        :return:
        '''

        # Do not update if the model is not training
        if not self.training:
            return

        points_priors = points_priors.view((-1, 2, self.space_dimensions))
        point_means = points_priors[:, 0] # Obtains the points means

        centroid_assignments = centroid_assignments.view((-1, self.centroids_count))

        # Adds dimensions for broadcasting
        point_means = point_means.unsqueeze(1)  # (..., 1, space_dimensions)
        unsqueezed_centroid_assignments = centroid_assignments.unsqueeze(-1)  # (..., centroids_count, 1)

        # Computes new centroids
        current_centroid_estimate = (point_means * unsqueezed_centroid_assignments).sum(0)  # (centroids_count, space_dimensions)
        mean_weights = centroid_assignments.sum(0).unsqueeze(-1)  # (centroids_count, 1)
        current_centroid_estimate = current_centroid_estimate / mean_weights

        return_centroids = self.estimated_centroids * (1 - self.alpha) + current_centroid_estimate * self.alpha

        # The estimated centroids must be detached from the backpropagation graph to avoid exhaustion of GPU memory
        self.estimated_centroids.data = return_centroids.detach()

    def compute_variations(self, points: torch.Tensor, centroid_assignments: torch.Tensor):
        '''
        Compute the variation vector of points with respect to centroids

        :param points: (..., space_dimensions) tensor with each point
        :param centroid_assignments: (..., centroids_count) tensor with cluster assignment probabilities in 0, 1
                                                            for each point
        :return: (..., space_dimensions) tensor with variation of each point with respect to centroids
        '''

        # The initial dimensions which will be flattened
        initial_dimensions = list(points.size())[:-1]

        points = points.view((-1, self.space_dimensions))
        centroid_assignments = centroid_assignments.view((-1, self.centroids_count))

        #
        variations = points.unsqueeze(1) - self.estimated_centroids   # (..., centroids_count, space_dimensions)
        variations = centroid_assignments.unsqueeze(-1) * variations  # (..., centroids_count, space_dimensions)
        variations = variations.sum(1)  # (..., space_dimensions)

        # Reshapes to the original dimension of points
        variations = variations.reshape(tuple(initial_dimensions + [-1]))

        return variations


if __name__ == "__main__":

    space_dim = 3
    centroids_count = 2
    alpha = 1.0

    points_priors = torch.tensor([
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
    ])

    centroid_assigments_1 = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])

    centroid_assigments_2 = torch.tensor([
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5]
    ])

    points = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
    ])

    centroid_estimator = CentroidEstimator(centroids_count, space_dim, 1.0)

    centroid_estimator(points_priors, centroid_assigments_1)
    variations = centroid_estimator.compute_variations(points, centroid_assigments_2)

    print(variations.cpu().numpy())













