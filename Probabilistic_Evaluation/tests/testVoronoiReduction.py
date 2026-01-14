import numpy as np
import pytest
import matplotlib.pyplot as plt

from Probabilistic_Evaluation.core.voronoiReduction import VoronoiCells

class TestVoronoiReduction():
    def __init__(self):
        pass

    def TestVoronoiCells(self, limit, sigma=1.6, plot=True):
        # Generate random points
        spacing = (1.0, 1.0, 1.0)
        sigmas = (sigma, sigma, sigma)
        # Create Voronoi cells
        voronoi_cells = VoronoiCells(limit, spacing, sigmas, method='both')

        # Monte Carlo simulation to estimate Voronoi cell probabilities
        probabilitiesMC = voronoi_cells.probabilitiesMC
        #analytical probabilities
        probabilities_analytical = voronoi_cells.probabilities_analytical
        vor_points = voronoi_cells.displacements
        
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

        assert(np.isclose(np.sum(probabilitiesMC), 1.0, atol=1e-2), "Monte Carlo probabilities do not sum to 1.")
        assert(np.isclose(np.sum(probabilities_analytical), 1.0, atol=1e-6), "Analytical probabilities do not sum to 1.")

        # Plotting
        if plot:
            plt.plot(probabilitiesMC,label='Monte Carlo',color='red')
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