import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
import matplotlib.pyplot as plt



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
    def __init__(self, limit, spacing, method = 'analytical'):
        limitX = np.round(limit / spacing[0])
        limitY = np.round(limit / spacing[1])
        limitZ = np.round(limit / spacing[2])
        limit_rangeX = np.arange(-limitX, limitX+1)
        limit_rangeY = np.arange(-limitY, limitY+1)
        limit_rangeZ = np.arange(-limitZ, limitZ+1)

        XX,YY,ZZ = np.meshgrid(limit_rangeX, limit_rangeY, limit_rangeZ)
        points = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
        print("Generated points {} for Voronoi Cells with spacing: {}".format(points.shape[0], spacing))
        super().__init__(points)
        self.spacing = spacing
        self.probabilitiesMC = None
        self.probabilities_analytical = None
        if method == 'montecarlo':
            self.probabilitiesToVoronoiCellsMC()
        elif method == 'analytical':
            self.probabilitiesToVoronoiCellsAnalytical()
        elif method == 'both':
            self.probabilitiesToVoronoiCellsMC()
            self.probabilitiesToVoronoiCellsAnalytical()


    def probabilitiesToVoronoiCellsMC(self, sigma=1.6):
        # Monte Carlo simulation to estimate Voronoi cell probabilities
        MCsamples = int(1e6)
        counter = np.zeros(len(self.points)+1)
        #MCpoints = np.random.normal(0, sigma, (MCsamples, 3))
        MCpointsX = np.random.normal(0, sigma, MCsamples) / self.spacing[0]
        MCpointsY = np.random.normal(0, sigma, MCsamples) / self.spacing[1]
        MCpointsZ = np.random.normal(0, sigma, MCsamples) / self.spacing[2]
        MCpoints = np.vstack([MCpointsX, MCpointsY, MCpointsZ]).T
        tree = cKDTree(self.points)
        dists, locations = tree.query(MCpoints, k=1)
        for loc in locations:
            counter[loc] += 1
        
        self.probabilitiesMC = counter / MCsamples
        return self.probabilitiesMC
        
    def probabilitiesToVoronoiCellsAnalytical(self, sigma=1.6):
        # Analytical calculation of Voronoi cell probabilities
        probabilities_analytical = []
        for point in self.points:
            x, y, z = point
            x = x * self.spacing[0]
            y = y * self.spacing[1]
            z = z * self.spacing[2]
            integral_x = intergral_gaussian_1D(x - self.spacing[0]/2, x + self.spacing[0]/2, mu=0, sigma=sigma)
            integral_y = intergral_gaussian_1D(y - self.spacing[1]/2, y + self.spacing[1]/2, mu=0, sigma=sigma)
            integral_z = intergral_gaussian_1D(z - self.spacing[2]/2, z + self.spacing[2]/2, mu=0, sigma=sigma)
            prob = integral_x * integral_y * integral_z
            probabilities_analytical.append(prob)

        
        self.probabilities_analytical = np.array(probabilities_analytical)
        self.probabilities_analytical /= np.sum(self.probabilities_analytical)  # Normalize to ensure sum to 1
        return self.probabilities_analytical

    
def TestVoronoiCells(limit, sigma=1.6, plot=True):
    # Generate random points
    spacing = (1.0, 1.0, 2.0)
    # Create Voronoi cells
    voronoi_cells = VoronoiCells(limit, spacing, method='both')

    # Monte Carlo simulation to estimate Voronoi cell probabilities
    probabilitiesMC = voronoi_cells.probabilitiesMC
    #analytical probabilities
    probabilities_analytical = voronoi_cells.probabilities_analytical
    vor_points = voronoi_cells.points
    
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