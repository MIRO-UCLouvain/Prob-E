import numpy as np
from scipy.spatial import Voronoi, cKDTree

from Probabilistic_Evaluation.core.sampling._abstractSamplingMethod import AbstractsamplingMethod



class ClassicalSampling(AbstractsamplingMethod):

    def __init__(self,uncertaintyModel: AbstractsamplingMethod):
        super().__init__(uncertaintyModel)


    def MCsampling(self,N_samples: int):
        samples = self.UncertaintyModel.sample(N_samples)
        probs = np.ones(N_samples) / N_samples
        return samples, probs


    def analyticalSampling(self,N_points_per_dim: int, bounds: np.ndarray):
        limit_range = [np.linspace(-b, b, N_points_per_dim) for b in bounds]
        XX,YY,ZZ = np.meshgrid(limit_range[0], limit_range[1], limit_range[2])
        points = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
        probabilities = []
        for point in points:
            prob = self.UncertaintyModel.pdf(point)
            probabilities.append(prob)
        probabilities = np.array(probabilities)
        # Normalize probabilities to sum to 1
        probabilities /= np.sum(probabilities)
        return points, probabilities

