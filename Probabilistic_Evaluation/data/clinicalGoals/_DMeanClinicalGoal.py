import numpy as np

from ._clinicalGoal import AbstractClinicalGoal

class DMeanClinicalGoal(AbstractClinicalGoal):
    """
    A class to represent a Dose Mean Clinical Goal (DMean) for radiation therapy evaluation.

    Attributes
    ----------
    prescription : float
        The prescribed mean dose value for the clinical goal.
    mask : np.ndarray
        A binary mask defining the region of interest for the clinical goal.
    lower_is_better : bool
        Indicates if lower mean dose values are better for this goal.
    """

    def __init__(self, prescription: float, mask: np.ndarray, lower_is_better: bool = True, **kwargs):
        super().__init__(prescription, mask, lower_is_better)

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