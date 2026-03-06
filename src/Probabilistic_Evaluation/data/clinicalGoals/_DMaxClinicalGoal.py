import numpy as np

from ._clinicalGoal import AbstractClinicalGoal

class DMaxClinicalGoal(AbstractClinicalGoal):
    """
    A class to represent a DMax Clinical Goal for radiation therapy evaluation.

    Attributes
    ----------
    prescription : float
        The prescribed maximum dose value for the clinical goal.
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
    """

    def __init__(self, prescription: float, mask: np.ndarray, maskName: str, lower_is_better: bool = True, priority: int = 0, **kwargs):
        super().__init__(prescription, mask, maskName, lower_is_better, priority)

    def __str__(self):
        if self.lower_is_better:
            comparison = "<="
        else:
            comparison = ">="
        return f"DMAX{comparison}{self.prescription}"

    def compute_value(self, dvh) -> float:
        """
        Compute the maximum dose from the DVH.

        Parameters
        ----------
        dvh : DVH
            The dose-volume histogram object used for computation.

        Returns
        -------
        float
            The computed maximum dose value.
        """
        value  = dvh.DMax
        return value