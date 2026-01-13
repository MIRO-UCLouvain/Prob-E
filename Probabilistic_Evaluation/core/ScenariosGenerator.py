import sys
sys.path.append('.')

import numpy as np  
from voronoiReduction import VoronoiCells
from Probabilistic_Evaluation.data._patientData import PatientData
import matplotlib.pyplot as plt
from Probabilistic_Evaluation.data._scenario import Scenario 

class ScenariosGenerator:
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
    def __init__(self,patientData:PatientData,max_displacement=5,sampling_method='voronoi'):

        self.ctImage = patientData.ctImage
        self.doseImage = patientData.doseImage
        self.targetMask = patientData.maskDict['Target']
        self.spacing = patientData.spacing
        self.displacements = np.arange(-max_displacement, max_displacement+1, 1)  # Example displacement range from -5 to 5
        if sampling_method == 'voronoi':
            self.voronoi_cells = VoronoiCells(max_displacement,self.spacing) 
        else:
            raise NotImplementedError("Only 'voronoi' sampling method is implemented.") 

        self.scenarios_list = []
        self.generate_scenarios(patientData)   


    def generate_scenarios(self,patientData:PatientData):
        for i, point in enumerate(self.voronoi_cells.points):
            scenario = Scenario(patientData)
            scenario.displacementScenario = point
            scenario.scenarioProbability = self.voronoi_cells.probabilities_analytical[i]
            self.scenarios_list.append(scenario)


def test_probabilistic_scenarios():
    # Example CT and dose images (3D numpy arrays)
    ctImage = np.zeros((50,50,50))
    ctImage[15:35,15:35,15:35] = 100  # Example high-density region
    doseImage = np.zeros((50,50,50))
    doseImage[20:30,20:30,20:30] = 50  # Example dose distribution
    targetMask = np.zeros((50,50,50),dtype=bool)  # Example target mask
    targetMask[22:28,22:28,22:28] = True  # Define target region
    maskdict = {'Target': targetMask}
    spacing = (1.0, 1.0, 2.0)  # Example spacing

    patientData = PatientData(ctImage, doseImage, maskdict, spacing=spacing)

    ps = ScenariosGenerator(patientData, max_displacement=5)
    
    # print("Evaluation results for all scenarios:", results)
    #2 plots, one showing the results as afucntion of scenario index and other one show probabilities of voronoi cells as function on cell(scenario) index
    plt.subplot(1, 1, 1)
    plt.plot(ps.voronoi_cells.probabilities_analytical, label='Analytical')
    plt.title("Voronoi Cell Probabilities")
    plt.xlabel("Voronoi Cell (Scenario) Index")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()


    for i, point in enumerate(ps.voronoi_cells.points):
        if point.tolist() == [0,0,0]:
            print(f"Scenario {i}: Displacement {point}, Probability: {ps.voronoi_cells.probabilities_analytical[i]}")
        if point.tolist() == [0,1,0]:
            print(f"Scenario {i}: Displacement {point}, Probability: {ps.voronoi_cells.probabilities_analytical[i]}")
        if point.tolist() == [0,0,1]:
            print(f"Scenario {i}: Displacement {point}, Probability: {ps.voronoi_cells.probabilities_analytical[i]}")
            shifted_dose = ps.shift_dose_image(ps.doseImage, shift=point)
            plt.imshow(shifted_dose[:,:,25], cmap='jet')
            plt.colorbar(label='Dose')
            plt.contour(ps.targetMask[:,:,25], colors='white', linewidths=0.5)
            plt.title(f"Dose Distribution for Displacement {point}")
            plt.show()
if __name__ == "__main__":
    test_probabilistic_scenarios()