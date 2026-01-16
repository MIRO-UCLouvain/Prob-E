import numpy as np

from Probabilistic_Evaluation.core.sampling._abstractSamplingMethod import AbstractsamplingMethod
from Probabilistic_Evaluation.data.uncertaintyModel._abstractUncertaintyModel import AbstractUncertaintyModel



class ClassicalSampling(AbstractsamplingMethod):
    """
    A class to represent classical sampling methods and compute their probabilities
    based on Monte Carlo simulation and analytical methods.

    Attributes:
    -----------
    UncertaintyModel : AbstractsamplingMethod
        The uncertainty model used for sampling.
    """

    def __init__(self, uncertaintyModel: AbstractUncertaintyModel):
        super().__init__(uncertaintyModel)

    def MCsampling(self, **kwargs):
        """
        Sample the uncertainty model using Monte Carlo sampling.

        Parameters
        ----------
        N_samples : int (default: 1000)
            The number of samples to draw.

        Returns
        -------
        samples : np.ndarray
            The sampled points.
        probs : np.ndarray
            The probabilities associated with each sampled point (1/N_samples).
        """
        N_samples = kwargs.get('N_samples', 1000)
        samples = self.UncertaintyModel.sample(N_samples)
        probs = np.ones(N_samples) / N_samples
        return samples, probs

    def analyticalSampling(self, **kwargs):
        """
        Sample the uncertainty model using analytical sampling.
        Creates a grid of points within the specified bounds and evaluates
        the PDF at each point to obtain probabilities.

        Parameters
        ----------
        N_points_per_dim : int (default: 10)
            The number of points to sample per dimension.
        bounds : np.ndarray (default: np.array([5.0, 5.0, 5.0]))
            The bounds for sampling in each dimension.

        Returns
        -------
        points : np.ndarray
            The sampled points.
        probabilities : np.ndarray
            The probabilities associated with each sampled point.
        """
        N_points_per_dim = kwargs.get('N_points_per_dim', 10)
        bounds = kwargs.get('bounds', np.array([5.0, 5.0, 5.0]))
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
