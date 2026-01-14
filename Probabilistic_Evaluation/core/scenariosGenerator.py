import numpy as np
from Probabilistic_Evaluation.core.sampling.voronoiReduction import VoronoiCells
from Probabilistic_Evaluation.data import PatientData
import matplotlib.pyplot as plt
from Probabilistic_Evaluation.data import Scenario
from Probabilistic_Evaluation.utils import shift_dose_image

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
    def __init__(self,patientData:PatientData,max_displacement=10,sigmas=(1.6, 1.6, 1.6), sampling_method='voronoi'):

        self.ctImage = patientData.ctImage
        self.doseImage = patientData.doseImage
        self.targetMask = patientData.maskDict['Target']
        self.spacing = patientData.spacing
        self.displacements = np.arange(-max_displacement, max_displacement+1, 1)  # Example displacement range from -5 to 5
        if sampling_method == 'voronoi':
            self.voronoi_cells = VoronoiCells(max_displacement,self.spacing,sigmas=sigmas) 
        else:
            raise NotImplementedError("Only 'voronoi' sampling method is implemented.") 

        self.scenarios_list = []
        self.generate_scenarios(patientData)   


    def generate_scenarios(self,patientData:PatientData):
        for i, point in enumerate(self.voronoi_cells.points):
            scenario = Scenario(point, self.voronoi_cells.probabilities[i])
            self.scenarios_list.append(scenario)
