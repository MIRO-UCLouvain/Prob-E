import numpy as np


class PatientData:
    """
    A class to store and manage patient data including CT images, dose images, and structure masks.

    Attributes
    ----------
    ctImage : np.ndarray
        A 3D numpy array representing the CT image of the patient.
    doseImage : np.ndarray
        A 3D numpy array representing the dose distribution for the patient.
    maskDict : dict
        A dictionary where keys are structure names (str) and values are 3D numpy arrays representing binary masks for those structures.
    clinicalGoalsDict : dict
        A dictionary where keys are structure names (str) and values are lists of ClinicalGoal objects for those structures.
    spacing : tuple (default=(1.0, 1.0, 1.0))
        A tuple representing the voxel spacing in each dimension (x, y, z).
    """

    def __init__(self, ctImage: np.ndarray, doseImage: np.ndarray, maskDict: dict, spacing: tuple = (1.0, 1.0, 1.0)):
        # check matrix dimensions matches
        if ctImage.shape != doseImage.shape:
            raise ValueError("CT image and dose image must have the same dimensions.")
        for structureName, mask in maskDict.items():
            if mask.shape != ctImage.shape:
                raise ValueError(f"Mask for structure '{structureName}' must have the same dimensions as the CT image.")
        self._ctImage = ctImage
        self._doseImage = doseImage
        self._maskDict = maskDict
        self._clinicalGoalsDict: dict = {}
        self._spacing = spacing

    @property
    def ctImage(self) -> np.ndarray:
        return self._ctImage

    @ctImage.setter
    def ctImage(self, newCtImage: np.ndarray):
        self._ctImage = newCtImage

    @property
    def doseImage(self) -> np.ndarray:
        return self._doseImage

    @doseImage.setter
    def doseImage(self, newDoseImage: np.ndarray):
        self._doseImage = newDoseImage

    @property
    def maskDict(self) -> dict:
        return self._maskDict

    @maskDict.setter
    def maskDict(self, newMaskDict: dict):
        self._maskDict = newMaskDict

    @property
    def clinicalGoalsDict(self) -> dict:
        return self._clinicalGoalsDict

    @clinicalGoalsDict.setter
    def clinicalGoalsDict(self, newClinicalGoalsDict: dict):
        self._clinicalGoalsDict = newClinicalGoalsDict

    @property
    def spacing(self) -> tuple:
        return self._spacing

    @spacing.setter
    def spacing(self, newSpacing: tuple):
        for dim in newSpacing:
            if dim <= 0:
                raise ValueError("Spacing values must be positive.")
        self._spacing = newSpacing

    def clinicalGoal(self, structureName: str) -> list:
        goal = self._clinicalGoalsDict.get(structureName, None)
        if goal is None:
            raise ValueError(f"Clinical goal for structure '{structureName}' not found in clinicalGoalsDict.")
        return goal

    def getMask(self, structureName: str) -> np.ndarray:
        mask = self._maskDict.get(structureName, None)
        if mask is None:
            raise ValueError(f"Structure '{structureName}' not found in maskDict.")
        return mask
