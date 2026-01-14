import numpy as np

from ._clinicalGoal import AbstractClinicalGoal

class VCCClinicalGoal(AbstractClinicalGoal):
    """
    A class to represent a Volume-Coverage Clinical Goal (VCC) for radiation therapy evaluation.

    Attributes
    ----------
    prescription : float
        The prescribed absolute volume (in CC) for the clinical goal.
    value : float
        The evaluated value for the clinical goal.
    lower_is_better : bool
        Indicates if lower values are better for this goal.
    mask : np.ndarray
        The mask defining the region of interest for the clinical goal.
    maskName : str
        The name of the mask defining the region of interest for the clinical goal.
    valueList : list
        A list to store list of values computed for the clinical goal on each scenario.
    successList : list
        A list to store success status for each scenario.
    dose : float
        The dose (in Gy) associated with the clinical goal.
    """


    def __init__(self, prescription: float, mask: np.ndarray, maskName: str, lower_is_better: bool = True, **kwargs):
        super().__init__(prescription, mask, maskName, lower_is_better)
        self._dose = kwargs.get('dose', None)  # Dose in Gy
        if self._dose is None:
            raise ValueError("Dose must be provided for VCC ClinicalGoal.")
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
        return f"{self.maskName}:V{self.dose}%{comparison}{self.prescription:.2f}cc"

    def compute_value(self, dvh) -> float:
        """
        Compute the dose corresponding to the specified volume from the DVH.
        Parameters
        ----------
        dvh : DVH
            The dose-volume histogram object used for computation.

        Returns
        -------
        float
            The computed dose value corresponding to the specified volume.
        """
        value = dvh.computeVcc(self.dose)
        return value