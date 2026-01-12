import numpy as np  
from VoronoiCells import VoronoiCells

class ProbabilisticScenarios:
    """
    A class to generate and evaluate probabilistic scenarios based on Voronoi cells.
    
    Attributes:
        ctImage (ndarray): The CT image data.
        doseImage (ndarray): The dose distribution image data.
        targetMask (ndarray): The target region mask.
        displacements (list): List of possible displacements in each dimension.
        points (ndarray): The generated scenario points.
        voronoi_cells (VoronoiCells): Voronoi cells object for scenario probabilities.
    Methods:
        generate_scenario_points():
            Generate scenario points based on displacements.
        compute_scenarios():
            Compute all the scenarios and evaluation metrics.
        shift_dose_image(doseImage, shift):
            Shift the dose image by a given displacement.
        evaluate_scenario(doseImage=None):
            Evaluate the scenario based on the shifted dose image.
    """
    def __init__(self,ctImage,doseImage,targetMask,max_displacement=5):
        """
        Docstring for __init__
        
        :param self: Description
        :param ctImage: Description
        :param doseImage: Description
        :param targetMask: Description
        :param max_displacement: Description
        """
        self.ctImage = ctImage
        self.doseImage = doseImage
        self.targetMask = targetMask
        self.displacements = np.arange(-max_displacement, max_displacement+1, 1)  # Example displacement range from -5 to 5
        self.points = self.generate_scenario_points()
        self.voronoi_cells = VoronoiCells(self.points) 

    def generate_scenario_points(self):
        XX,YY,ZZ = np.meshgrid(self.displacements, self.displacements, self.displacements)
        points = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
        return points
        
        
    def compute_scenarios(self):
        results = []
        for point in self.points:
            shift_x, shift_y, shift_z = point
            shifted_dose = self.shift_dose_image(self.doseImage, shift=(shift_x, shift_y, shift_z))
            result = self.evaluate_scenario(shifted_dose)
            results.append(result)
        return results
    

    def shift_dose_image(self, doseImage, shift):
        shifted_dose = np.roll(doseImage, shift=shift, axis=(0, 1, 2))
        return shifted_dose
    
    def evaluate_scenario(self,doseImage=None):
        shifted_dose = doseImage
        doseInTarget = shifted_dose[self.targetMask > 0]
        # Placeholder for actual evaluation logic
        evaluation_metric = np.mean(doseInTarget)  # Example metric: mean dose
        return evaluation_metric
    

import matplotlib.pyplot as plt
def test_probabilistic_scenarios():
    # Example CT and dose images (3D numpy arrays)
    ctImage = np.zeros((50,50,50))
    ctImage[15:35,15:35,15:35] = 100  # Example high-density region
    doseImage = np.zeros((50,50,50))
    doseImage[20:30,20:30,20:30] = 50  # Example dose distribution
    targetMask = np.zeros((50,50,50),dtype=bool)  # Example target mask
    targetMask[22:28,22:28,22:28] = True  # Define target region

    ps = ProbabilisticScenarios(ctImage, doseImage, targetMask, max_displacement=5)
    results = ps.compute_scenarios()
    
    # print("Evaluation results for all scenarios:", results)
    #2 plots, one showing the results as afucntion of scenario index and other one show probabilities of voronoi cells as function on cell(scenario) index
    plt.subplot(2, 1, 1)
    plt.plot(results)
    plt.title("Evaluation Metric Across Scenarios")
    plt.xlabel("Scenario Index")
    plt.ylabel("Evaluation Metric")
    plt.subplot(2, 1, 2)
    plt.plot(ps.voronoi_cells.probabilities_analytical, label='Analytical')
    plt.title("Voronoi Cell Probabilities")
    plt.xlabel("Voronoi Cell (Scenario) Index")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()

    for i, point in enumerate(ps.points):
        if point.tolist() == [0,0,0]:
            print(f"Scenario {i}: Displacement {point}, Evaluation Metric: {results[i]}, Probability: {ps.voronoi_cells.probabilities_analytical[i]}")
        if point.tolist() == [4,4,4]:
            print(f"Scenario {i}: Displacement {point}, Evaluation Metric: {results[i]}, Probability: {ps.voronoi_cells.probabilities_analytical[i]}")
            shifted_dose = ps.shift_dose_image(ps.doseImage, shift=point)
            plt.imshow(shifted_dose[:,:,25], cmap='jet')
            plt.colorbar(label='Dose')
            plt.contour(ps.targetMask[:,:,25], colors='white', linewidths=0.5)
            plt.title(f"Dose Distribution for Displacement {point}")
            plt.show()
if __name__ == "__main__":
    test_probabilistic_scenarios()