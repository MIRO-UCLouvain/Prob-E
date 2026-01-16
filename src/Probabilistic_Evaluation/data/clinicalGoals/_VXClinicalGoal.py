import numpy as np

from ._clinicalGoal import AbstractClinicalGoal

class VXClinicalGoal(AbstractClinicalGoal):
    """
    A class to represent a Volume-Dose Percentage Clinical Goal (VX) for radiation therapy evaluation.

    Attributes
    ----------
    prescription : float
        The prescribed volume percentage (in %, e.g. 0.5 for 50%) for the clinical goal.
    lower_is_better : bool (default=True)
        Indicates if lower values are better for this goal.
    mask : np.ndarray
        The mask defining the region of interest for the clinical goal.
    maskName : str
        The name of the mask defining the region of interest for the clinical goal.
    priority : int
        The priority level of the clinical goal, zero if no cummulative evaluation is desired.
    valueList : list
        A list to store list of values computed for the clinical goal on each scenario.
    successList : list
        A list to store success status for each scenario.
    dose : float
        The dose (in Gy) associated with the clinical goal.
    """

    def __init__(self, prescription: float, mask: np.ndarray, maskName: str, lower_is_better: bool = True, **kwargs):
        if prescription > 1:
            raise ValueError("Prescription must be given in percentage (0-1).")
        super().__init__(prescription, mask, maskName, lower_is_better)
        self._dose = kwargs.get('dose', None)  # Dose in Gy
        if self._dose is None:
            raise ValueError("Dose must be provided for VX ClinicalGoal.")
        if self._dose < 0:
            raise ValueError("Dose must be a non-negative value.")

    @property
    def dose(self) -> float:
        return self._dose

    @dose.setter
    def dose(self, newDose: float):
        if newDose < 0:
            raise ValueError("Dose must be a non-negative value.")
        self._dose = newDose

    def __str__(self):
        if self.lower_is_better:
            comparison = "<="
        else:
            comparison = ">="
        return f"{self.maskName}:V{self.dose}{comparison}{self.prescription:.1f}%"

    def compute_value(self, dvh) -> float:
        """
        Compute the volume percentage corresponding to the specified dose from the DVH.
        Parameters
        ----------
        dvh : DVH
            The dose-volume histogram object used for computation.

        Returns
        -------
        float
            The computed volume percentage corresponding to the specified dose.
        """
        value = dvh.computeVx(self.dose)
        return value