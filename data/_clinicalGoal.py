

class ClinicalGoal:

    def __init__(self,prescription: float,lower_is_better: bool = True):
        self.name: str = "ClinicalGoal"
        self.prescription : float = prescription
        self.lower_is_better : bool = lower_is_better
        self.achieved : bool = False

