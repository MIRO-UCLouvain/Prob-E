import numpy as np

from ._clinicalGoal import AbstractClinicalGoal

class DMaxClinicalGoal(AbstractClinicalGoal):
    """
    A class to represent a DMax Clinical Goal for radiation therapy evaluation.

    Attributes
    ----------
    prescription : float
        The prescribed maximum dose value for the clinical goal.
    mask : np.ndarray
        A binary mask defining the region of interest for the clinical goal.
    lower_is_better : bool
        Indicates if lower dose values are better for this goal.
    """

    def __init__(self, prescription: float, mask: np.ndarray, lower_is_better: bool = True, **kwargs):
        super().__init__(prescription, mask, lower_is_better)

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
        value  = dvh.Dmax()
        return value