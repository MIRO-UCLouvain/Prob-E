import numpy as np

from ._clinicalGoal import AbstractClinicalGoal

class DMeanClinicalGoal(AbstractClinicalGoal):
    """
    A class to represent a Dose Mean Clinical Goal (DMean) for radiation therapy evaluation.

    Attributes
    ----------
    prescription : float
        The prescribed mean dose value for the clinical goal.
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
    """

    def __init__(self, prescription: float, mask: np.ndarray, maskName: str, lower_is_better: bool = True, **kwargs):
        super().__init__(prescription, mask, maskName, lower_is_better)

    def __str__(self):
        if self.lower_is_better:
            comparison = "<="
        else:
            comparison = ">="
        return f"{self.maskName}:DMEAN{comparison}{self.prescription}"

    def compute_value(self, dvh) -> float:
        """
        Compute the mean dose from the DVH.
        Parameters
        ----------
        dvh : DVH
            The dose-volume histogram object used for computation.

        Returns
        -------
        float
            The computed mean dose value.
        """
        value = dvh.Dmean()
        return value