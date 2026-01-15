import numpy as np
import pytest
import matplotlib.pyplot as plt

from Probabilistic_Evaluation.core.sampling.voronoiSampling import VoronoiSampling
from Probabilistic_Evaluation.data.uncertaintyModel._Gaussian3D import Gaussian3DUncertaintyModel

class TestVoronoiReduction():
    def __init__(self):
        pass

    def TestVoronoiCells(self, limit, sigma=5/3.2, plot=True):
        # Generate random points
        bounds = np.array([limit, limit, limit])
        spacing = np.array([1.0, 1.0, 2.0])
        sigmas = (sigma, sigma, sigma)
        # Create Voronoi cells
        UncertaintyModel = Gaussian3DUncertaintyModel(parameters={'mu_x': 0, 'mu_y': 0, 'mu_z': 0,
                                                                 'sigma_x': sigmas[0],
                                                                 'sigma_y': sigmas[1],
                                                                 'sigma_z': sigmas[2]})
        voronoi_cells = VoronoiSampling(UncertaintyModel, bounds, spacing)
        # Monte Carlo simulation to estimate Voronoi cell probabilities
        _, probabilitiesMC = voronoi_cells.MCsampling(N_samples=100000)
        #analytical probabilities
        vor_points, probabilities_analytical = voronoi_cells.analyticalSampling()
       
        
        print("Voronoi Points and Analytical Probabilities:\n")
        print("vor_ponts.shape:", vor_points.shape)
        print("probabilities_analytical.shape:", probabilities_analytical.shape)
        points_probabilities = np.hstack((vor_points, probabilities_analytical.reshape(-1, 1)))
        # sort by probabilities
        points_probabilities = points_probabilities[np.argsort(points_probabilities[:, 3])[::-1]]
        #take the proabibilities needed to reach 95% of cumulative probability
        cumulative_prob = np.cumsum(points_probabilities[:, 3])
        num_cells_95 = np.searchsorted(cumulative_prob, 0.95,side='right') + 1
        threshold_95 = points_probabilities[num_cells_95-1,3]
        cumulative_prob95 = np.cumsum(points_probabilities[:, 3][points_probabilities[:, 3]>=threshold_95])
        print(f"Number of Voronoi cells to reach 95% cumulative probability: {len(cumulative_prob95)} and the minimum probability of a scenario to be included: {threshold_95}")
        num_cells_97 = np.searchsorted(cumulative_prob, 0.97,side='right') + 1
        threshold_97 = points_probabilities[num_cells_97-1,3]
        cumulative_prob97 = np.cumsum(points_probabilities[:, 3][points_probabilities[:, 3]>=threshold_97])
        print(f"Number of Voronoi cells to reach 97% cumulative probability: {len(cumulative_prob97)} and the minimum probability of a scenario to be included: {threshold_97}")
        num_cells_99 = np.searchsorted(cumulative_prob, 0.99,side='right') + 1
        threshold_99 = points_probabilities[num_cells_99-1,3]
        cumulative_prob99 = np.cumsum(points_probabilities[:, 3][points_probabilities[:, 3]>=threshold_99])
        print(f"Number of Voronoi cells to reach 99% cumulative probability: {len(cumulative_prob99)} and the minimum probability of a scenario to be included: {threshold_99}")
        num_cells_9999 = np.searchsorted(cumulative_prob, 0.9999,side='right') + 1
        threshold_9999 = points_probabilities[num_cells_9999-1,3]
        cumulative_prob9999 = np.cumsum(points_probabilities[:, 3][points_probabilities[:, 3]>=threshold_9999])
        print(f"Number of Voronoi cells to reach 99.99% cumulative probability: {len(cumulative_prob9999)} and the minimum probability of a scenario to be included: {threshold_9999}")

        # Plotting
        if plot:
            #plt.plot(probabilitiesMC,label='Monte Carlo',color='red')
            plt.plot(probabilities_analytical, label='Analytical',color='blue', linestyle='dashed') #marker = '.')
            plt.hlines(y=points_probabilities[num_cells_95-1,3], xmin=0, xmax=len(probabilities_analytical), colors='green', linestyles='dotted', label='95% Cumulative Probability Threshold')
            plt.hlines(y=points_probabilities[num_cells_97-1,3], xmin=0, xmax=len(probabilities_analytical), colors='orange', linestyles='dotted', label='97% Cumulative Probability Threshold')
            plt.hlines(y=points_probabilities[num_cells_99-1,3], xmin=0, xmax=len(probabilities_analytical), colors='purple', linestyles='dotted', label='99% Cumulative Probability Threshold')
            plt.hlines(y=points_probabilities[num_cells_9999-1,3], xmin=0, xmax=len(probabilities_analytical), colors='brown', linestyles='dotted', label='99.99% Cumulative Probability Threshold')
            plt.xlabel('Voronoi Cell Index')
            plt.ylabel('Probability')
            plt.title('Voronoi Cell Probabilities: Monte Carlo vs Analytical')
            plt.legend()
            plt.show()

if __name__ == "__main__":

    PTVmargin = 5
    limit = 2*PTVmargin
    test_voronoi = TestVoronoiReduction()
    test_voronoi.TestVoronoiCells(limit, sigma=PTVmargin/(3.2), plot=True)

    # resol = 10
    # limit = 5
    # Npixels = resol * limit
    # Image3Dgaussian = np.zeros((Npixels, Npixels, Npixels))
    # for i in range(Npixels):
    #     for j in range(Npixels):
    #         for k in range(Npixels):
    #             x = i * limit / Npixels - limit / 2
    #             y = j * limit / Npixels - limit / 2
    #             z = k * limit / Npixels - limit / 2
    #             Image3Dgaussian[i, j, k] = gaussian3D(x, y, z, mu=0, sigma=PTVmargin/(3.2))

    # plt.imshow(Image3Dgaussian[ :, :, int(Npixels/2)])
    # plt.title('2D Slice of 3D Gaussian Image')
    # plt.colorbar()
    # plt.show()