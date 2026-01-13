import numpy as np
from ._patientData import PatientData
from ._DVH import DVH


class Scenario(object):
    """
    A class to represent a clinical scenario for radiation therapy evaluation.

    Attributes
    ----------
    doseImageScenario : np.ndarray
        A 3D numpy array representing the dose distribution for the scenario.
    patientData : PatientData
        An instance of PatientData containing patient-specific information.
    displacementScenario : np.ndarray
        A 3D numpy array representing the displacement applied in the scenario.
    scenarioProbability : float
        A float representing the probability of the scenario occurring.
    clinicalGoalsValues : dict
        A dictionary storing the computed values for clinical goals in the scenario.
    clinicalGoalsValuesAchieved : dict
        A dictionary indicating whether clinical goals were achieved in the scenario.

    Methods
    -------
    computeGoalValues():
        Computes the values for clinical goals based on the dose distribution and patient data.

    """

    def __init__(self, doseImageScenario: np.ndarray, patientData: PatientData):
        self._doseImageScenario = doseImageScenario
        self._patientData = patientData
        self._displacementScenario: np.ndarray = None
        self._scenarioProbability: float = None
        self._clinicalGoalsValues: dict = {}
        self._clinicalGoalsValuesAchieved: dict = {}

    @property
    def doseImageScenario(self) -> np.ndarray:
        return self._doseImageScenario

    @doseImageScenario.setter
    def doseImageScenario(self, newDoseImageScenario: np.ndarray):
        self._doseImageScenario = newDoseImageScenario

    @property
    def patientData(self) -> PatientData:
        return self._patientData

    @patientData.setter
    def patientData(self, newPatientData: PatientData):
        self._patientData = newPatientData

    @property
    def displacementScenario(self) -> np.ndarray:
        return self._displacementScenario

    @displacementScenario.setter
    def displacementScenario(self, newDisplacementScenario: np.ndarray):
        self._displacementScenario = newDisplacementScenario

    @property
    def clinlicalGoalsValues(self) -> dict:
        return self._clinlicalGoalsValues

    @clinlicalGoalsValues.setter
    def clinlicalGoalsValues(self, newClinlicalGoalsValues: dict):
        self._clinlicalGoalsValues = newClinlicalGoalsValues

    @property
    def clinicalGoalsValuesAchieved(self) -> dict:
        return self._clinicalGoalsValuesAchieved

    @clinicalGoalsValuesAchieved.setter
    def clinicalGoalsValuesAchieved(self, newClinicalGoalsValuesAchieved: dict):
        self._clinicalGoalsValuesAchieved = newClinicalGoalsValuesAchieved

    @property
    def scenarioProbability(self) -> float:
        return self._scenarioProbability

    @scenarioProbability.setter
    def scenarioProbability(self, newScenarioProbability: float):
        if newScenarioProbability < 0 or newScenarioProbability > 1:
            raise ValueError("Scenario probability must be between 0 and 1.")
        self._scenarioProbability = newScenarioProbability

    @property
    def doseImageScenario(self) -> np.ndarray:
        return self._doseImageScenario

    @doseImageScenario.setter
    def doseImageScenario(self, newDoseImageScenario: np.ndarray):
        self._doseImageScenario = newDoseImageScenario

    def computeGoalValues(self):
        """
        Computes the values for clinical goals based on the dose distribution and patient data.

        Returns
        -------

        """
        used_masks = set(self._patientData.clinicalGoalsDict.values()[i].maskName for i in range(len(self._patientData.clinicalGoalsDict)))
        for name in used_masks:
            dvh = DVH(self._doseImageScenario, self._patientData.maskDict[name],spacing=self._patientData.spacing)
            for goal in self._patientData.clinicalGoalsDict.values():
                if goal.maskName != name:
                    continue
                value = goal.compute_value(dvh)
                achieved = goal.achieved
                self._clinicalGoalsValues[goal.__str__()] = value
                self._clinicalGoalsValuesAchieved[goal.__str__()] = achieved
