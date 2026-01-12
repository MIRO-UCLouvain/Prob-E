import numpy as np
from ._patientData import PatientData
from ._DVH import DVH


class Scenario(object):
    """
    A class to represent a clinical scenario for radiation therapy evaluation.

    """

    def __init__(self, doseImageScenario: np.ndarray, patientData: PatientData):
        self._doseImageScenario = doseImageScenario
        self._patientData = patientData
        self._scenarioDisplacement: np.ndarray = None
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
        for structName in self._patientData.clinicalGoalsDict.keys():
            dvh = DVH(self._doseImageScenario, self._patientData.maskDict[structName])
            for goal in self._patientData.clinicalGoalsDict[structName]:
                value = goal.compute_value(dvh)
                achieved = goal.achieved
                self._clinicalGoalsValues[goal.__str__()] = value
                self._clinicalGoalsValuesAchieved[goal.__str__()] = achieved
