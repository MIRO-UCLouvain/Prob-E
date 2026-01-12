import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
import matplotlib.pyplot as plt
import pytest


def gaussian3D(x, y, z, mu=0, sigma=3):
    """Compute the value of a 3D Gaussian function at point (x, y, z)."""
    coeff = 1 / ((2 * np.pi) ** 1.5 * sigma ** 3)
    exponent = -((x - mu) ** 2 + (y - mu) ** 2 + (z - mu) ** 2) / (2 * sigma ** 2)
    return coeff * np.exp(exponent)

def intergral_gaussian_1D(a, b, mu=0, sigma=3):
    """Compute the integral of a 1D Gaussian function from a to b."""
    from scipy.special import erf
    coeff = 0.5 * (erf((b - mu) / (sigma * np.sqrt(2))) - erf((a - mu) / (sigma * np.sqrt(2))))
    return coeff

class VoronoiCells(Voronoi):
    """
    A class to represent Voronoi cells and compute their probabilities
    based on Monte Carlo simulation and analytical methods (default is analytical).
    Usefull attributes for probabilistic planning are probabilitiesMC, probabilities_analytical and points

    Attributes:
        points (ndarray): The input points for Voronoi diagram.
        probabilitiesMC (ndarray): Probabilities estimated via Monte Carlo.
        probabilities_analytical (ndarray): Probabilities calculated analytically.

    Methods:
        probabilitiesToVoronoiCellsMC(limit=6, sigma=1.6):
            Estimate Voronoi cell probabilities using Monte Carlo simulation.

        probabilitiesToVoronoiCellsAnalytical(limit=6, sigma=1.6):
            Calculate Voronoi cell probabilities analytically.
    
    """
    def __init__(self, points, method = 'analytical'):
        super().__init__(points)
        self.probabilitiesMC = None
        self.probabilities_analytical = None
        if method == 'montecarlo':
            self.probabilitiesToVoronoiCellsMC()
        elif method == 'analytical':
            self.probabilitiesToVoronoiCellsAnalytical()
        elif method == 'both':
            self.probabilitiesToVoronoiCellsMC()
            self.probabilitiesToVoronoiCellsAnalytical()


    def probabilitiesToVoronoiCellsMC(self,limit=6, sigma=1.6):
        # Monte Carlo simulation to estimate Voronoi cell probabilities
        MCsamples = int(1e6)
        counter = np.zeros(len(self.points)+1)
        MCpoints = np.random.normal(0, sigma, (MCsamples, 3))
        tree = cKDTree(self.points)
        dists, locations = tree.query(MCpoints, k=1)
        for loc in locations:
            counter[loc] += 1
        
        self.probabilitiesMC = counter / MCsamples
        return self.probabilitiesMC
        
    def probabilitiesToVoronoiCellsAnalytical(self,limit=6, sigma=1.6):
        # Analytical calculation of Voronoi cell probabilities
        probabilities_analytical = []
        for point in self.points:
            x, y, z = point
            integral_x = intergral_gaussian_1D(x - 0.5, x + 0.5, mu=0, sigma=sigma)
            integral_y = intergral_gaussian_1D(y - 0.5, y + 0.5, mu=0, sigma=sigma)
            integral_z = intergral_gaussian_1D(z - 0.5, z + 0.5, mu=0, sigma=sigma)
            prob = integral_x * integral_y * integral_z
            probabilities_analytical.append(prob)

        
        self.probabilities_analytical = np.array(probabilities_analytical)
        self.probabilities_analytical /= np.sum(self.probabilities_analytical)  # Normalize to ensure sum to 1
        return self.probabilities_analytical

    
def TestVoronoiCells(limit, sigma=1.6, plot=True):
    # Generate random points
    limit_range = np.arange(-(limit), limit+1)
    XX,YY,ZZ = np.meshgrid(limit_range, limit_range, limit_range)
    points = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
    print(f"Generated {points.shape} points for Voronoi diagram.")
    
    # Create Voronoi cells
    voronoi_cells = VoronoiCells(points, method='both')

    # Monte Carlo simulation to estimate Voronoi cell probabilities
    probabilitiesMC = voronoi_cells.probabilitiesMC
    #analytical probabilities
    probabilities_analytical = voronoi_cells.probabilities_analytical
    vor_points = voronoi_cells.points

    # Plotting
    if plot:
        plt.plot(probabilitiesMC,label='Monte Carlo',color='red')
        plt.plot(probabilities_analytical, label='Analytical',color='blue', linestyle='dashed')
        plt.xlabel('Voronoi Cell Index')
        plt.ylabel('Probability')
        plt.title('Voronoi Cell Probabilities: Monte Carlo vs Analytical')
        plt.legend()
        plt.show()

if __name__ == "__main__":

    PTVmargin = 5
    limit = 5
    TestVoronoiCells(limit,sigma=PTVmargin/(3.2),plot = True)

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

