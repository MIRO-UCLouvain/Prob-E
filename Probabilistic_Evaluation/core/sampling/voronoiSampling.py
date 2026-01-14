import numpy as np
from scipy.spatial import cKDTree
from Probabilistic_Evaluation.core.sampling._abstractSamplingMethod import AbstractsamplingMethod


class VoronoiSampling(AbstractsamplingMethod):

    def __init__(self,uncertaintyModel: AbstractsamplingMethod, bounds: np.ndarray, spacing: np.ndarray):
        super().__init__(uncertaintyModel)
        if bounds.shape != spacing.shape:
            raise ValueError("Bounds and spacing must have the same shape.")
        self._voronoiPoints = None
        self._voronoiProbabilities = None
        self._generateVoronoiSampling(bounds, spacing, computing_method='analytical')


    def _generateVoronoiPoints(self, bounds: np.ndarray, spacing: np.ndarray) -> np.ndarray:
        rounded_bounds = np.around(bounds/spacing)
        limit_range = [np.arange(-b, b+1) for b in rounded_bounds]
        XX,YY,ZZ = np.meshgrid(limit_range[0], limit_range[1], limit_range[2])
        print("Generated points {} for Voronoi Cells with spacing: {}".format(self.displacements.shape[0], spacing))
        voronoiPoints = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
        return voronoiPoints


    def _computeVoronoiProbabilitiesMC(self, vornoiPoints : np.ndarray, num_samples: int = 100000):
        # Generate random samples from the uncertainty model
        samples = self.UncertaintyModel.sample(num_samples)
        # Create a KDTree for efficient nearest neighbor search
        tree = cKDTree(vornoiPoints)
        # Find the nearest Voronoi point for each sample
        _, indices = tree.query(samples)
        # Count occurrences in each Voronoi cell
        counts = np.bincount(indices, minlength=vornoiPoints.shape[0])
        # Calculate probabilities
        probabilities = counts / num_samples
        return probabilities

    def _computeVoronoiProbabilitiesAnalytical(self, vornoiPoints : np.ndarray,spacing: np.ndarray):
        probabilities = []
        for point in vornoiPoints:
            x, y, z = point
            x = x * spacing[0]
            y = y * spacing[1]
            z = z * spacing[2]
            a = [x - spacing[0]/2, y - spacing[1]/2, z - spacing[2]/2]
            b = [x + spacing[0]/2, y + spacing[1]/2, z + spacing[2]/2]
            prob = self.UncertaintyModel.boundedIntegral(a, b)
            probabilities.append(prob)
        probabilities = np.array(probabilities)
        # Normalize probabilities to sum to 1
        probabilities /= np.sum(probabilities)
        return probabilities


    def _generateVoronoiProbabilities(self,vornoiPoints : np.ndarray, computing_method: str,spacing: np.ndarray):
        if computing_method == 'montecarlo':
            return self._computeVoronoiProbabilitiesMC(vornoiPoints)
        elif computing_method == 'analytical':
            return self._computeVoronoiProbabilitiesAnalytical(vornoiPoints, spacing)
        else:
            raise ValueError("Computing method must be 'montecarlo' or 'analytical'.")

    def _generateVoronoiSampling(self, bounds: np.ndarray, spacing: np.ndarray, computing_method: str):
        vornoiPoints = self._generateVoronoiPoints(bounds, spacing)
        probabilities = self._generateVoronoiProbabilities(vornoiPoints, computing_method, spacing)
        self._voronoiPoints = vornoiPoints
        self._voronoiProbabilities = probabilities


    def MCsampling(self,N_samples: int):
        Nmax = self._voronoiPoints.shape[0]
        indexes = np.random.choice(Nmax, probs=self.voronoiProbabilities, size=N_samples)
        samples = self.voronoiPoints[indexes]
        probs = self.voronoiProbabilities[indexes]
        return samples, probs

    def analyticalSampling(self):
        return self._voronoiPoints, self._voronoiProbabilities




