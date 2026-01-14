from scipy.spatial import Voronoi, cKDTree
import matplotlib.pyplot as plt
from Probabilistic_Evaluation.data.uncertaintyModel import Gaussian3DUncertaintyModel
from _abstractSamplingMethod import *

class VoronoiCells(AbstractsamplingMethod):
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
    def __init__(self, limit, spacing, sigmas, method = 'analytical'):
        self.UncertaintyModel = Gaussian3DUncertaintyModel({'mu_x': 0, 'mu_y': 0, 'mu_z': 0, 'sigma_x': sigmas[0], 'sigma_y': sigmas[1], 'sigma_z': sigmas[2]})
        limitX = np.round(limit / spacing[0])
        limitY = np.round(limit / spacing[1])
        limitZ = np.round(limit / spacing[2])
        limit_rangeX = np.arange(-limitX, limitX+1)
        limit_rangeY = np.arange(-limitY, limitY+1)
        limit_rangeZ = np.arange(-limitZ, limitZ+1)

        XX,YY,ZZ = np.meshgrid(limit_rangeX, limit_rangeY, limit_rangeZ)
        self.displacements = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
        print("Generated points {} for Voronoi Cells with spacing: {}".format(self.displacements.shape[0], spacing))
        super().__init__(self.displacements)
        self.name = "VoronoiCells"
        self.spacing = spacing
        self.probabilitiesMC = None
        self.probabilities_analytical = None
        self.probabilities = None

        self.sample(method=method)

        
    def sample(self, method='analytical'):
        if method == 'montecarlo':
            self.probabilitiesToVoronoiCellsMC()
            self.probabilities = self.probabilitiesMC
        elif method == 'analytical':
            self.probabilitiesToVoronoiCellsAnalytical()
            self.probabilities = self.probabilities_analytical
        elif method == 'both':
            self.probabilitiesToVoronoiCellsMC()
            self.probabilitiesToVoronoiCellsAnalytical()
            self.probabilities = self.probabilities_analytical
        else:
            raise ValueError("Method must be 'montecarlo', 'analytical' or  'both'.")

    def probabilitiesToVoronoiCellsMC(self, sigma=5/3.2):
        # Monte Carlo simulation to estimate Voronoi cell probabilities
        MCsamples = int(1e6)
        counter = np.zeros(len(self.displacements)+1)
        #MCpoints = np.random.normal(0, sigma, (MCsamples, 3))
        MCpointsX = np.random.normal(0, sigma, MCsamples) / self.spacing[0]
        MCpointsY = np.random.normal(0, sigma, MCsamples) / self.spacing[1]
        MCpointsZ = np.random.normal(0, sigma, MCsamples) / self.spacing[2]
        MCpoints = np.vstack([MCpointsX, MCpointsY, MCpointsZ]).T
        tree = cKDTree(self.displacements)
        dists, locations = tree.query(MCpoints, k=1)
        for loc in locations:
            counter[loc] += 1
        
        self.probabilitiesMC = counter / MCsamples
        return self.probabilitiesMC
        
    def probabilitiesToVoronoiCellsAnalytical(self):
        # Analytical calculation of Voronoi cell probabilities
        probabilities_analytical = []
        for point in self.displacements:
            x, y, z = point
            x = x * self.spacing[0]
            y = y * self.spacing[1]
            z = z * self.spacing[2]
            a = [x - self.spacing[0]/2, y - self.spacing[1]/2, z - self.spacing[2]/2]
            b = [x + self.spacing[0]/2, y + self.spacing[1]/2, z + self.spacing[2]/2]
            prob = self.UncertaintyModel.boundedIntegral(a, b)
            # sigma = 5/3.2
            # integral_x = intergral_gaussian_1D(x - self.spacing[0]/2, x + self.spacing[0]/2, mu=0, sigma=sigma)
            # integral_y = intergral_gaussian_1D(y - self.spacing[1]/2, y + self.spacing[1]/2, mu=0, sigma=sigma)
            # integral_z = intergral_gaussian_1D(z - self.spacing[2]/2, z + self.spacing[2]/2, mu=0, sigma=sigma)
            # prob = integral_x * integral_y * integral_z
            probabilities_analytical.append(prob)

        
        self.probabilities_analytical = np.array(probabilities_analytical)
        self.probabilities_analytical /= np.sum(self.probabilities_analytical)  # Normalize to ensure sum to 1
        return self.probabilities_analytical
