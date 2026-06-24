import numbers
from unittest import result

import numpy as np
from scipy.spatial import cKDTree
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
        self._computing_method = 'montecarlo'  # Default computing method
        self._generateVoronoiSampling(max_displacements, spacing)

    @property
    def voronoiPoints(self) -> np.ndarray:
        """Voronoi points generated within the specified bounds and spacing."""
        return self._voronoiPoints

    @property
    def voronoiProbabilities(self) -> np.ndarray:
        """Probabilities associated with each Voronoi point."""
        return self._voronoiProbabilities

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
        XX, YY, ZZ = np.meshgrid(limit_range[0], limit_range[1], limit_range[2])
        voronoiPoints = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
        if self.enhanced:
            self._computing_method = 'montecarlo'  # Switch to Monte Carlo for enhanced sampling, needed for integral calculation of the probabilities of the Voronoi cells. 
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
            print("{} more points generated from enhanced sampling".format(len(mid_points)))
        print("{} Total generated points for Voronoi Cells with spacing: {}".format(voronoiPoints.shape[0], spacing))
  
        return voronoiPoints

    def _computeVoronoiProbabilitiesMC(self, voronoiPoints: np.ndarray, num_samples: int = 10000000):
        """
        Compute Voronoi cell probabilities using Monte Carlo integration.

        Parameters
        ----------
        voronoiPoints : np.ndarray
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
        # probabilities /= np.sum(probabilities)

        return probabilities

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
            return self._computeVoronoiProbabilitiesMC(voronoiPoints)
        elif self._computing_method == 'analytical':
            return self._computeVoronoiProbabilitiesAnalytical(voronoiPoints, spacing)
        else:
            raise ValueError("Computing method must be 'montecarlo' or 'analytical'.")

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

        if self.probabilityMass is not None:
            self._voronoiPoints, self._voronoiProbabilities = self.reduceNumberOfScenarios()

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
        return samples, probs

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
            #make sure all points at the same distance have the same probability to avoid MC sampling deviations
            norms = [np.linalg.norm(p) for p in self._voronoiPoints]
            norms = np.round(norms, decimals=6)#round to avoid floating point errors when comparing norms of points that are very close to each other
            unique_norms, inverse = np.unique(norms, return_inverse=True)
            print("Unique norms found:", unique_norms)
    
            # Average the probabilities for Voronoi points with the same norm 
            for  norm_val in unique_norms:
                probs_with_same_norm = self._voronoiProbabilities[norms == norm_val]
                self._voronoiProbabilities[norms == norm_val] = np.average(probs_with_same_norm)
                print(f"Average probability for norm {norm_val}: {self._voronoiProbabilities[norms == norm_val][0]}")

        sorted_indices = np.argsort(self._voronoiProbabilities)[::-1]
        sorted_probabilities = self._voronoiProbabilities[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probabilities)

        # check what is the number of cells needed to reach the specified cumulative probability mass
        num_cells = np.searchsorted(cumulative_probs, self.probabilityMass, side='right')
        # take all the cells with exactly the same probability as the last one to include all cells with the same probability
        while num_cells < len(sorted_probabilities) and sorted_probabilities[num_cells] == sorted_probabilities[num_cells - 1]:
            num_cells += 1
    
        reduced_indices = sorted_indices[:num_cells]
        reduced_points = self._voronoiPoints[reduced_indices]
        reduced_probabilities = self._voronoiProbabilities[reduced_indices]

        # Normalize the reduced probabilities to sum to 1
        #reduced_probabilities /= np.sum(reduced_probabilities)
        return reduced_points, reduced_probabilities
