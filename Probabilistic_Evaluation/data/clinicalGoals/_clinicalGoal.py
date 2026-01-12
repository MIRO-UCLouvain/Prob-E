import numpy as np
from abc import ABC, abstractmethod


class AbstractClinicalGoal(ABC):
    """
    Abstract base class for clinical goals.

    Attributes
    ----------
    prescription : float
        The prescribed value for the clinical goal.
    value : float
        The evaluated value for the clinical goal.
    lower_is_better : bool
        Indicates if lower values are better for this goal.
    achieved : bool
        Indicates if the goal has been achieved.
    mask : np.ndarray
        The mask defining the region of interest for the clinical goal.

    Methods
    -------
    compute_value() -> float
        Abstract method to compute the value of the clinical goal.
    """

    def __init__(self, prescription: float, mask: np.ndarray, maskName: str, lower_is_better: bool = True, **kwargs):
        if prescription < 0:
            raise ValueError("Clinical goal prescription cannot be negative.")
        self._prescription: float = prescription
        self._lower_is_better: bool = lower_is_better
        self._mask: np.ndarray = mask
        self._maskName: str = maskName
        self._passingRate: float = None
        self._cummulative_passingRate: float = None

    @property
    def prescription(self) -> float:
        return self._prescription

    @prescription.setter
    def prescription(self, newPrescription: float):
        self._prescription = newPrescription

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, newValue: float):
        if newValue < 0:
            raise ValueError("Clinical goal value cannot be negative.")
        if self._value is not None:
            self._achieved = None  # Reset achieved status
        self._value = newValue

    @property
    def lower_is_better(self) -> bool:
        return self._lower_is_better

    @lower_is_better.setter
    def lower_is_better(self, newLowerIsBetter: bool):
        self._lower_is_better = newLowerIsBetter

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @mask.setter
    def mask(self, newMask: np.ndarray):
        self._mask = newMask

    @property
    def maskName(self) -> str:
        return self._maskName

    @maskName.setter
    def maskName(self, newMaskName: str):
        self._maskName = newMaskName

    @passingRate.setter
    def SetPassingRate(self, passingRate: float):
        self._passingRate = passingRate
    
    @_cummulative_passingRate.setter
    def SetCummulative_passingRate(self, cummulative_passingRate: float):
        self._cummulative_passingRate = cummulative_passingRate


    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def compute_value(self, dvh) -> float:
        """
        Compute the value of the clinical goal based on the provided DVH.
        Parameters
        ----------
        dvh

        Returns
        -------
        float
            The computed value of the clinical goal.
        """
        pass
