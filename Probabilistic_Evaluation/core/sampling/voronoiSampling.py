import numpy as np
from scipy.spatial import cKDTree
from Probabilistic_Evaluation.core.sampling._abstractSamplingMethod import AbstractsamplingMethod


class VoronoiSampling(AbstractsamplingMethod):
    """
    A class to represent Voronoi sampling and compute their probabilities
    based on Monte Carlo simulation and analytical methods (default is analytical).

    Voronoi sampling precomputes achievable Voronoi points and their associated probabilities
    based on the provided uncertainty model, bounds, and spacing.

    Attributes:
    -----------
    voronoiPoints : np.ndarray
        The Voronoi points generated within the specified bounds and spacing.
    voronoiProbabilities : np.ndarray
        The probabilities associated with each Voronoi point.

    Methods:
    --------
    MCsampling(N_samples: int):
        Perform Monte Carlo sampling based on the Voronoi points and their probabilities.
    analyticalSampling():
        Retrieve the precomputed Voronoi points and their probabilities.
    """

    def __init__(self, uncertaintyModel: AbstractsamplingMethod, bounds: np.ndarray, spacing: np.ndarray):
        super().__init__(uncertaintyModel)
        if bounds.shape != spacing.shape:
            raise ValueError("Bounds and spacing must have the same shape.")
        self._voronoiPoints = None
        self._voronoiProbabilities = None
        self._generateVoronoiSampling(bounds, spacing, computing_method='analytical')

    @property
    def voronoiPoints(self) -> np.ndarray:
        return self._voronoiPoints

    @property
    def voronoiProbabilities(self) -> np.ndarray:
        return self._voronoiProbabilities

    def _generateVoronoiPoints(self, bounds: np.ndarray, spacing: np.ndarray) -> np.ndarray:
        """
        Generate Voronoi points within the specified bounds and spacing.

        Parameters
        ----------
        bounds : np.ndarray
            The bounds for generating Voronoi points.
        spacing : np.ndarray
            The spacing between Voronoi points.

        Returns
        -------
        voronoiPoints : np.ndarray
            The generated Voronoi points.
        """

        rounded_bounds = np.around(bounds / spacing)
        limit_range = [np.arange(-b, b + 1) for b in rounded_bounds]
        XX, YY, ZZ = np.meshgrid(limit_range[0], limit_range[1], limit_range[2])
        print("Generated points {} for Voronoi Cells with spacing: {}".format(self.displacements.shape[0], spacing))
        voronoiPoints = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
        return voronoiPoints

    def _computeVoronoiProbabilitiesMC(self, voronoiPoints: np.ndarray, num_samples: int = 100000):
        """
        Compute Voronoi cell probabilities using Monte Carlo integration.

        Parameters
        ----------
        vornoiPoints : np.ndarray
            The Voronoi points for which to compute probabilities.
        num_samples : int (default=100000)
            The number of Monte Carlo samples to use.

        Returns
        -------
        probabilities : np.ndarray
            The computed probabilities for each Voronoi cell.
        """
        # Generate random samples from the uncertainty model
        samples = self.UncertaintyModel.sample(num_samples)
        # Create a KDTree for efficient nearest neighbor search
        tree = cKDTree(voronoiPoints)
        # Find the nearest Voronoi point for each sample
        _, indices = tree.query(samples)
        # Count occurrences in each Voronoi cell
        counts = np.bincount(indices, minlength=voronoiPoints.shape[0])
        # Calculate probabilities
        probabilities = counts / num_samples
        return probabilities

    def _computeVoronoiProbabilitiesAnalytical(self, voronoiPoints: np.ndarray, spacing: np.ndarray):
        """
        Compute Voronoi cell probabilities using analytical integration.

        Parameters
        ----------
        vornoiPoints : np.ndarray
            The Voronoi points for which to compute probabilities.
        spacing : np.ndarray
            The spacing between Voronoi points.

        Returns
        -------
        probabilities : np.ndarray
            The computed probabilities for each Voronoi cell.
        """
        probabilities = []
        for point in voronoiPoints:
            x, y, z = point
            x = x * spacing[0]
            y = y * spacing[1]
            z = z * spacing[2]
            a = [x - spacing[0] / 2, y - spacing[1] / 2, z - spacing[2] / 2]
            b = [x + spacing[0] / 2, y + spacing[1] / 2, z + spacing[2] / 2]
            prob = self.UncertaintyModel.boundedIntegral(a, b)
            probabilities.append(prob)
        probabilities = np.array(probabilities)
        # Normalize probabilities to sum to 1
        probabilities /= np.sum(probabilities)
        return probabilities

    def _generateVoronoiProbabilities(self, voronoiPoints: np.ndarray, computing_method: str, spacing: np.ndarray):
        """
        Generate Voronoi cell probabilities based on the specified computing method.

        Parameters
        ----------
        voronoiPoints : np.ndarray
            The Voronoi points for which to compute probabilities.
        computing_method : str
            The method to use for computing probabilities ('montecarlo' or 'analytical').
        spacing : np.ndarray
            The spacing between Voronoi points.
        Returns
        -------
        probabilities : np.ndarray
            The computed probabilities for each Voronoi cell.
        """
        if computing_method == 'montecarlo':
            return self._computeVoronoiProbabilitiesMC(voronoiPoints)
        elif computing_method == 'analytical':
            return self._computeVoronoiProbabilitiesAnalytical(voronoiPoints, spacing)
        else:
            raise ValueError("Computing method must be 'montecarlo' or 'analytical'.")

    def _generateVoronoiSampling(self, bounds: np.ndarray, spacing: np.ndarray, computing_method: str):
        """
        Generate Voronoi sampling points and their associated probabilities and store them as attributes.

        Parameters
        ----------
        bounds : np.ndarray
            The bounds for generating Voronoi points.
        spacing : np.ndarray
            The spacing between Voronoi points.
        computing_method : str
            The method to use for computing probabilities ('montecarlo' or 'analytical')

        Returns
        -------
        """
        vornoiPoints = self._generateVoronoiPoints(bounds, spacing)
        probabilities = self._generateVoronoiProbabilities(vornoiPoints, computing_method, spacing)
        self._voronoiPoints = vornoiPoints
        self._voronoiProbabilities = probabilities

    def MCsampling(self, N_samples: int):
        """
        Perform Monte Carlo sampling based on the Voronoi points and their probabilities.

        Parameters
        ----------
        N_samples : int
            The number of samples to generate.

        Returns
        -------
        samples : np.ndarray
            The sampled Voronoi points.
        probs : np.ndarray
            The probabilities associated with the sampled points.
        """
        Nmax = self._voronoiPoints.shape[0]
        indexes = np.random.choice(Nmax, probs=self.voronoiProbabilities, size=N_samples)
        samples = self.voronoiPoints[indexes]
        probs = self.voronoiProbabilities[indexes]
        return samples, probs

    def analyticalSampling(self):
        """
        Retrieve the precomputed Voronoi points and their probabilities.

        Returns
        -------
        voronoiPoints : np.ndarray
            The Voronoi points.
        voronoiProbabilities : np.ndarray
            The probabilities associated with each Voronoi point.
        """
        return self._voronoiPoints, self._voronoiProbabilities
