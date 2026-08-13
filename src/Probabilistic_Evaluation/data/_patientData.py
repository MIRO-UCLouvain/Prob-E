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
    clinicalGoalsList : list
        A list of clinical goal objects associated with the patient.
    spacing : np.ndarray
        A numpy array representing the voxel spacing in each dimension (x, y, z).
    scenarioList : list
        A list of scenario objects associated with the patient.

    Methods
    -------
    getClinicalGoalsForStructure(structureName: str) -> list
        Returns a list of clinical goals associated with the specified structure name.
    getMask(structureName: str) -> np.ndarray
        Returns the mask array for the specified structure name.
    """

    def __init__(self, doseImage: np.ndarray, maskDict: dict, spacing: np.ndarray, patientID= None,prescribedDose=None):
        # check matrix dimensions matches
        for structureName, mask in maskDict.items():
            if mask.shape != doseImage.shape:
                raise ValueError(f"Mask for structure '{structureName}' must have the same dimensions as the dose image.")

        self._doseImage = doseImage
        self._maskDict = maskDict
        self._clinicalGoalsList = []
        self._probabilisticGoalsList = []
        self.patientID = patientID
        self._spacing = spacing
        self.scenarioList = []
        # self.prescribedDose = prescribedDose

    @property
    def doseImage(self) -> np.ndarray:
        return self._doseImage

    @doseImage.setter
    def doseImage(self, newDoseImage: np.ndarray):
        if newDoseImage.shape != self._doseImage.shape:
            raise ValueError("New dose image must have the same dimensions as the existing dose image.")
        self._doseImage = newDoseImage

    @property
    def maskDict(self) -> dict:
        return self._maskDict

    @maskDict.setter
    def maskDict(self, newMaskDict: dict):
        for structureName, mask in newMaskDict.items():
            if mask.shape != self._doseImage.shape:
                raise ValueError(f"Mask for structure '{structureName}' must have the same dimensions as the dose image.")
        self._maskDict = newMaskDict

    @property
    def spacing(self) -> np.ndarray:
        return self._spacing

    @spacing.setter
    def spacing(self, newSpacing: np.ndarray):
        for dim in newSpacing:
            if dim <= 0:
                raise ValueError("Spacing values must be positive.")
        self._spacing = newSpacing

    @property
    def clinicalGoalsList(self) -> list:
        return self._clinicalGoalsList

    @clinicalGoalsList.setter
    def clinicalGoalsList(self, newClinicalGoalsList: list):
        self._clinicalGoalsList = newClinicalGoalsList
        self._probabilisticGoalsList = [goal for goal in newClinicalGoalsList if goal.probabilistic]

    @property
    def probabilisticGoalsList(self) -> list:
        return self._probabilisticGoalsList

    def getClinicalGoalsForStructure(self, structureName: str) -> list:
        """
        Return a list of clinical goals associated with the specified structure name.

        Parameters
        ----------
        structureName : str
            The name of the structure to retrieve clinical goals for.

        Returns
        -------
        list
            A list of clinical goal objects associated with the specified structure name.
        """
        goals = [goal for goal in self._clinicalGoalsList if goal.maskName == structureName]
        if not goals:
            raise KeyError(f"No clinical goals found for structure '{structureName}'.")
        return goals

    def getMask(self, structureName: str) -> np.ndarray:
        """
        Return the mask array for the specified structure name.

        Parameters
        ----------
        structureName : str
            The name of the structure to retrieve the mask for.

        Returns
        -------
        np.ndarray
            The mask array for the specified structure name.
        """
        mask = self._maskDict.get(structureName, None)
        if mask is None:
            raise KeyError(f"Structure '{structureName}' not found in maskDict.")
        return mask
