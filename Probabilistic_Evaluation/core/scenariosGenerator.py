import numpy as np
from Probabilistic_Evaluation.core.sampling.voronoiReduction import VoronoiCells
from Probabilistic_Evaluation.data import PatientData
import matplotlib.pyplot as plt
from Probabilistic_Evaluation.data import Scenario
from Probabilistic_Evaluation.utils import shift_dose_image
from Probabilistic_Evaluation.core.sampling._abstractSamplingMethod import AbstractsamplingMethod

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
    def __init__(self,sampling_method:AbstractsamplingMethod):

        self.sampling_method = sampling_method
        self.scenarios_list = []
        self.generate_scenarios()


    def generate_scenarios(self):
        displacements, probabilities = self.sampling_method.analyticalSampling()
        for i in range(len(displacements)):
            scenario = Scenario(displacement=displacements[i], probability=probabilities[i])
            self.scenarios_list.append(scenario)
