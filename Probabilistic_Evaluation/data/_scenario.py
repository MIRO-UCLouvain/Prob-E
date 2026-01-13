import numpy as np
from Probabilistic_Evaluation.utils import *


class Scenario(object):
    """
    A class to represent a clinical scenario for radiation therapy evaluation.

    Attributes
    ----------
    displacement : np.ndarray
        The displacement vector for the scenario.
    probability : float
        The probability of occurrence for the scenario.
    doseImage : np.ndarray
        The dose distribution image for the scenario.
    """

    def __init__(self, displacement: np.ndarray, probability: float):
        self._displacement: np.ndarray = displacement
        self._probability: float = probability
        self._doseImage = None

    @property
    def doseImage(self) -> np.ndarray:
        return self._doseImage

    @doseImage.setter
    def doseImage(self, newDoseImage: np.ndarray):
        self._doseImage = newDoseImage

    @property
    def displacement(self) -> np.ndarray:
        return self._displacement

    @property
    def probability(self) -> float:
        return self._probability

    @probability.setter
    def probability(self, newProbability: float):
        if newProbability < 0 or newProbability > 1:
            raise ValueError("Scenario probability must be between 0 and 1.")
        self._probability = newProbability

    def compute_shifted_image(self,initial_dose_image,displacement: np.ndarray) -> np.ndarray:
        shift_x, shift_y, shift_z = displacement
        self._doseImage = shift_dose_image(initial_dose_image, shift=(shift_x, shift_y, shift_z))

    def delete_doseImage(self):
        self._doseImage = None

