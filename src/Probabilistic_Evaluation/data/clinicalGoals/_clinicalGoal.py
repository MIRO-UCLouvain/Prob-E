import numpy as np
from abc import ABC, abstractmethod


class AbstractClinicalGoal(ABC):
    """
    Abstract base class for clinical goals.

    Attributes
    ----------
    prescription : float
        The prescribed value for the clinical goal.
    lower_is_better : bool (default=True)
        Indicates if lower values are better for this goal.
    mask : np.ndarray
        The mask defining the region of interest for the clinical goal.
    maskName : str
        The name of the mask defining the region of interest for the clinical goal.
    priority : int (default=0)
        The priority level of the clinical goal, zero if no cummulative evaluation is desired.
    valueList : list
        A list to store list of values computed for the clinical goal on each scenario.
    successList : list
        A list to store success status for each scenario.

    Methods
    -------
    compute_value() -> float
        Abstract method to compute the value of the clinical goal.
    compute_success(value) -> bool
        Method to compute if the clinical goal is achieved based on the value.
    compute(dvh) -> None
        Method to compute the clinical goal value and success status based on the provided DVH.
    """

    def __init__(self, prescription: float, mask: np.ndarray, maskName: str, lower_is_better: bool = True,**kwargs):
        if prescription < 0:
            raise ValueError("Clinical goal prescription cannot be negative.")
        self._prescription: float = prescription
        self._lower_is_better: bool = lower_is_better
        self._mask: np.ndarray = mask
        self._maskName: str = maskName
        self._priority: int = kwargs.get('priority', 0)
        self.valueList: list = []
        self.successList: list = []

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

    @property
    def priority(self) -> int:
        return self._priority
    @priority.setter
    def priority(self, newPriority: int):
        self._priority = newPriority

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


    def compute_success(self,value) -> bool:
        """
        Determine if the clinical goal is achieved based on the computed value.

        Parameters
        ----------
        value : float
            The computed value of the clinical goal.

        Returns
        -------
        bool
            True if the goal is achieved, False otherwise.
        """
        if self._lower_is_better:
            return value <= self._prescription
        else:
            return value >= self._prescription

    def compute(self,dvh) -> None:
        """
        Compute the clinical goal value and success status based on the provided DVH.

        Parameters
        ----------
        dvh : DVH
            The Dose-Volume Histogram of the scenario.

        Returns
        -------
        None
        """
        value = self.compute_value(dvh)
        success = self.compute_success(value)
        self.valueList.append(value)
        self.successList.append(success)
