import numpy as np
from scipy.spatial import Voronoi, cKDTree

from Probabilistic_Evaluation.core.sampling._abstractSamplingMethod import AbstractsamplingMethod


class ClassicalSampling(AbstractsamplingMethod):
    """
    A class to represent classical sampling methods and compute their probabilities
    based on Monte Carlo simulation and analytical methods.

    Attributes:
    -----------
    UncertaintyModel : AbstractsamplingMethod
        The uncertainty model used for sampling.
    """

    def __init__(self, uncertaintyModel: AbstractsamplingMethod):
        super().__init__(uncertaintyModel)

    def MCsampling(self, N_samples: int):
        """
        Sample the uncertainty model using Monte Carlo sampling.

        Parameters
        ----------
        N_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : np.ndarray
            The sampled points.
        probs : np.ndarray
            The probabilities associated with each sampled point (1/N_samples).
        """
        samples = self.UncertaintyModel.sample(N_samples)
        probs = np.ones(N_samples) / N_samples
        return samples, probs

    def analyticalSampling(self, N_points_per_dim: int, bounds: np.ndarray):
        """
        Sample the uncertainty model using analytical sampling.
        Creates a grid of points within the specified bounds and evaluates
        the PDF at each point to obtain probabilities.

        Parameters
        ----------
        N_points_per_dim : int
            The number of points to sample per dimension.
        bounds : np.ndarray
            The bounds for sampling in each dimension.

        Returns
        -------
        points : np.ndarray
            The sampled points.
        probabilities : np.ndarray
            The probabilities associated with each sampled point.
        """
        limit_range = [np.linspace(-b, b, N_points_per_dim) for b in bounds]
        XX, YY, ZZ = np.meshgrid(limit_range[0], limit_range[1], limit_range[2])
        points = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
        probabilities = []
        for point in points:
            prob = self.UncertaintyModel.pdf(point)
            probabilities.append(prob)
        probabilities = np.array(probabilities)
        # Normalize probabilities to sum to 1
        probabilities /= np.sum(probabilities)
        return points, probabilities
