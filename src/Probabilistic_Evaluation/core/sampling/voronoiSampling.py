import numbers
from unittest import result

import numpy as np
from scipy.spatial import cKDTree
from Probabilistic_Evaluation.logging_utils import logger, log_call
from Probabilistic_Evaluation.core.sampling._abstractSamplingMethod import AbstractsamplingMethod
from Probabilistic_Evaluation.data.uncertaintyModel._abstractUncertaintyModel import AbstractUncertaintyModel


class VoronoiSampling(AbstractsamplingMethod):
    """
    A class to represent Voronoi sampling and compute their probabilities based on Monte Carlo simulation and analytical methods (default is analytical).

    Voronoi sampling precomputes achievable Voronoi points and their associated probabilities
    based on the provided uncertainty model, bounds, and spacing.

    Attributes
    ----------
    voronoiPoints : np.ndarray
        The Voronoi points generated within the specified bounds and spacing.
    voronoiProbabilities : np.ndarray
        The probabilities associated with each Voronoi point.
    probabilityMass : float
        The cumulative probability mass to retain when reducing the number of scenarios (between 0 and 1).

    Methods
    -------
    MCsampling(N_samples: int):
        Perform Monte Carlo sampling based on the Voronoi points and their probabilities.
    analyticalSampling():
        Retrieve the precomputed Voronoi points and their probabilities.
    """

    def __init__(self, uncertaintyModel: AbstractUncertaintyModel, max_displacements: np.ndarray, spacing: np.ndarray,probabilityMass:float=None, enhancedSampling: bool = False):
        super().__init__(uncertaintyModel)
        if max_displacements.shape != spacing.shape:
            raise ValueError("max_displacements and spacing must have the same shape.")
        self._voronoiPoints = None
        self._voronoiProbabilities = None
        self.probabilityMass = probabilityMass
        self.enhanced = enhancedSampling
        self._spacing = np.asarray(spacing, dtype=float)  # mm per grid step, per axis
        self._computing_method = 'analytical'  # Default computing method
        logger.info(f"Voronoi sampling initialized with total probability mass: {self.probabilityMass}, enhanced sampling grid: {self.enhanced}" )
        self._generateVoronoiSampling(max_displacements, spacing)

        
        
    @property
    def voronoiPoints(self) -> np.ndarray:
        """Voronoi points generated within the specified bounds and spacing."""
        return self._voronoiPoints

    @property
    def voronoiProbabilities(self) -> np.ndarray:
        """Probabilities associated with each Voronoi point."""
        return self._voronoiProbabilities

    @log_call(log_result=True)
    def _generateVoronoiPoints(self, max_displacements: np.ndarray, spacing: np.ndarray) -> np.ndarray:
        """
        Generate Voronoi points within the specified bounds and spacing.

        Parameters
        ----------
        max_displacements : np.ndarray
            The maximum displacements for generating Voronoi points.
        spacing : np.ndarray
            The spacing between Voronoi points.
        enhanced : bool (default=False)
            If True, generate additional points to enhance the Voronoi sampling. These points are the middle points between the original points. This is useful for more accurate sampling, but increases the number of scenarios.
        Returns
        -------
        voronoiPoints : np.ndarray
            The generated Voronoi points.
        """
        rounded_bounds = np.around(max_displacements / spacing)
        limit_range = [np.arange(-b, b + 1) for b in rounded_bounds]
        logger.debug(f"Generating Voronoi points with bounds: {rounded_bounds}, spacing: {spacing}, limit_range: {limit_range[0]}")
        XX, YY, ZZ = np.meshgrid(limit_range[0], limit_range[1], limit_range[2])
        voronoiPoints = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
        if self.enhanced:
            self._computing_method = 'montecarlo'
            logger.info("Switching to Monte Carlo probability calculation for enhanced sampling, needed for integral calculation of the probabilities of the Voronoi cells.")
            # Generate additional points in between the original points but only in the middle of the cube defined by the original points. This is done to enhance the Voronoi sampling and provide more accurate results.
            mid_points = []
            for point in voronoiPoints:
                mid_point = point.copy()
                for i in range(3):
                    mid_point[i] += 0.5
                if np.all(np.abs(mid_point) <= rounded_bounds):
                    mid_points.append(mid_point)
            if mid_points:
                voronoiPoints = np.vstack([voronoiPoints, np.array(mid_points)])
            logger.debug("{} more points generated from enhanced sampling".format(len(mid_points)))
        logger.info("{} Total generated points for Voronoi Cells with spacing: {}".format(voronoiPoints.shape[0], spacing))

        return voronoiPoints

    @log_call(log_result=True)
    def _computeVoronoiProbabilitiesMC(self, voronoiPoints: np.ndarray, spacing: np.ndarray, num_samples: int = 10000000):
        """
        Compute Voronoi cell probabilities using Monte Carlo integration.

        Parameters
        ----------
        voronoiPoints : np.ndarray
            The Voronoi points (grid indices) for which to compute probabilities.
        spacing : np.ndarray
            The spacing between Voronoi points in mm, per axis.
        num_samples : int (default=10000000)
            The number of Monte Carlo samples to use.

        Returns
        -------
        probabilities : np.ndarray
            The computed probabilities for each Voronoi cell.
        """
        # Generate random samples from the uncertainty model (in mm)
        samples = self.UncertaintyModel.sample(num_samples)
        # Create a KDTree for efficient nearest neighbor search; the grid points are indices, so
        # convert them to mm before comparing them with the samples
        tree = cKDTree(voronoiPoints * np.asarray(spacing, dtype=float))
        # Find the nearest Voronoi point for each sample
        _, indices = tree.query(samples)
        # Count occurrences in each Voronoi cell
        counts = np.bincount(indices, minlength=voronoiPoints.shape[0])
        # Calculate probabilities
        probabilities = counts / num_samples
        logger.debug(f"Monte Carlo sampling completed with {num_samples} samples. Probabilities computed for {len(probabilities)} Voronoi cells.")
        return probabilities

    @log_call(log_result=True)
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
        # probabilities /= np.sum(probabilities)
        logger.debug(f"Analytical probability computation completed for {len(probabilities)} Voronoi cells.")
        return probabilities

    @log_call(log_result=True)
    def _generateVoronoiProbabilities(self, voronoiPoints: np.ndarray, spacing: np.ndarray):
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
        if self._computing_method == 'montecarlo':
            logger.info(f"Computing Voronoi probabilities via Monte Carlo")
            return self._computeVoronoiProbabilitiesMC(voronoiPoints, spacing)
        elif self._computing_method == 'analytical':
            logger.info(f"Computing Voronoi probabilities analytically")
            return self._computeVoronoiProbabilitiesAnalytical(voronoiPoints, spacing)
        else:
            raise ValueError("Computing method must be 'montecarlo' or 'analytical'.")

    @log_call(log_result=True)
    def _generateVoronoiSampling(self, bounds: np.ndarray, spacing: np.ndarray):
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
        None.
        """
        self._voronoiPoints = self._generateVoronoiPoints(bounds, spacing)
        self._voronoiProbabilities = self._generateVoronoiProbabilities(self._voronoiPoints, spacing)
        logger.debug(f"Sum of the {self._computing_method} probabilities of the {len(self._voronoiPoints)} Voronoi cells: {np.sum(self._voronoiProbabilities):.6f}.")

        if self.probabilityMass is not None:
            logger.info(f"Reducing number of scenarios to retain cumulative probability mass: {self.probabilityMass}")
            self._voronoiPoints, self._voronoiProbabilities = self.reduceNumberOfScenarios()

    @log_call(log_result=True)
    def MCsampling(self, **kwargs):
        """
        Perform Monte Carlo sampling based on the Voronoi points and their probabilities.

        Parameters
        ----------
        N_samples : int (default: 1000)
            The number of samples to generate.

        Returns
        -------
        samples : np.ndarray
            The sampled Voronoi points.
        probs : np.ndarray
            The probabilities associated with the sampled points.
        """
        N_samples = kwargs.get('N_samples', 10000)
        Nmax = self._voronoiPoints.shape[0]
        indexes = np.random.choice(Nmax, p=self._voronoiProbabilities, size=N_samples)

        samples = self.voronoiPoints[indexes]
        probs = self.voronoiProbabilities[indexes]
        logger.debug(f"This function is not implemented yet, and should not be used, is only a placeholder for future implementation.")
        return samples, probs

    @log_call(log_result=True)
    def analyticalSampling(self, **kwargs):
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

    @log_call(log_result=True)
    def reduceNumberOfScenarios(self):
        """
        Reduce the number of Voronoi scenarios based on a specified cumulative probability threshold.

        Returns
        -------
        reduced_points : np.ndarray
            The reduced set of Voronoi points.
        reduced_probabilities : np.ndarray
            The probabilities associated with the reduced Voronoi points.
        """

        if not (0 < self.probabilityMass <= 1):
            raise ValueError("Cumulative probability must be between 0 and 1.")

        if self._computing_method == 'montecarlo':
            # Cells that are images of each other under the symmetries of the lattice and of the
            # uncertainty model have exactly the same probability; average their Monte Carlo estimates
            # to remove the sampling noise (and to keep whole symmetry classes together when reducing).
            keys = np.array([self._symmetry_key(p) for p in self._voronoiPoints])
            unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
            inverse = np.asarray(inverse).ravel()
            class_mean = np.bincount(inverse, weights=self._voronoiProbabilities) / np.bincount(inverse)
            self._voronoiProbabilities = class_mean[inverse]
            logger.debug(f"Averaged Monte Carlo probabilities over {len(unique_keys)} symmetry classes of {len(self._voronoiPoints)} Voronoi points.")

        sorted_indices = np.argsort(self._voronoiProbabilities)[::-1]
        sorted_probabilities = self._voronoiProbabilities[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probabilities)

        # check what is the number of cells needed to reach the specified cumulative probability mass
        num_cells = np.searchsorted(cumulative_probs, self.probabilityMass, side='right')
        if cumulative_probs.size and cumulative_probs[-1] < self.probabilityMass:
            logger.debug(
                f"The requested probability mass {self.probabilityMass} exceeds the total mass of all {len(cumulative_probs)} Voronoi cells "
                f"({cumulative_probs[-1]:.6f}): all cells are kept, increase max_displacements to cover more probability mass."
            )
        # take all the cells with exactly the same probability as the last one to include all cells with the same probability
        while num_cells < len(sorted_probabilities) and sorted_probabilities[num_cells] == sorted_probabilities[num_cells - 1]:
            num_cells += 1
    
        reduced_indices = sorted_indices[:num_cells]
        reduced_points = self._voronoiPoints[reduced_indices]
        reduced_probabilities = self._voronoiProbabilities[reduced_indices]
        logger.info(f"Reduced number of Voronoi scenarios from {len(self._voronoiPoints)} to {len(reduced_points)} to retain cumulative probability mass of +-{self.probabilityMass}.")
        logger.info(f"Effective cumulative probability mass of the reduced scenarios: {round(np.sum(reduced_probabilities), 6)}, to make sure all equiprobable scenarios are included. ")
        # Normalize the reduced probabilities to sum to 1
        #reduced_probabilities /= np.sum(reduced_probabilities)
        return reduced_points, reduced_probabilities

    def _symmetry_key(self, point) -> tuple:
        """
        Canonical representative of all lattice points whose Voronoi cells have, by symmetry, the same
        probability as ``point``.

        A sign flip along an axis is a symmetry when the model mean is zero on that axis (the lattice is
        symmetric about the origin by construction). Two axes are interchangeable when both have a zero
        mean, the same grid spacing and the same standard deviation. Points that are not related by one
        of these operations (e.g. (3,0,0) and (2,2,1), or (1,0,0) and (0,0,1) with different sigma or
        spacing on x and z) keep separate keys.

        Parameters
        ----------
        point : array_like
            Voronoi point in grid-index units.

        Returns
        -------
        tuple
            Key that is identical for all symmetry-equivalent points.
        """
        sys_par = self.UncertaintyModel.sys_parameters or {}
        axes = ('x', 'y', 'z')
        mu = np.array([float(sys_par.get(f'mu_{a}', 0.0)) for a in axes])
        sigma = [sys_par.get(f'sigma_{a}', None) for a in axes]
        zero_mean = np.isclose(mu, 0.0)

        coords = np.asarray(point, dtype=float)
        key = np.where(zero_mean, np.abs(coords), coords)

        visited = [False, False, False]
        for i in range(3):
            if visited[i]:
                continue
            group = [i]
            visited[i] = True
            for j in range(i + 1, 3):
                if visited[j]:
                    continue
                interchangeable = (
                    zero_mean[i] and zero_mean[j]
                    and np.isclose(self._spacing[i], self._spacing[j])
                    and sigma[i] is not None and sigma[j] is not None
                    and np.isclose(float(sigma[i]), float(sigma[j]))
                )
                if interchangeable:
                    group.append(j)
                    visited[j] = True
            key[group] = np.sort(key[group])
        return tuple(np.round(key, 6))
