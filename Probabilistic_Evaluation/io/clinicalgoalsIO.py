import json
<<<<<<< HEAD
from data.clinicalGoals._clinicalGoal import AbstractClinicalGoal
from data.clinicalGoals._DMaxClinicalGoal import DMaxClinicalGoal
from data.clinicalGoals._DMinClinicalGoal import DMinClinicalGoal
from data.clinicalGoals._DMeanClinicalGoal import DMeanClinicalGoal
from data.clinicalGoals._DCCClinicalGoal import DCCClinicalGoal
from data.clinicalGoals._VCCClinicalGoal import VCCClinicalGoal
from data.clinicalGoals._VXClinicalGoal import VXClinicalGoal
from data.clinicalGoals._DXClinicalGoal import DXClinicalGoal
=======
from Probabilistic_Evaluation.Evaluation.clinicalgoals import *
>>>>>>> 009a2dc2f613ae8635e9689a6f45528702098012

class clinicalgoalsreader():
    """
    A class to represent a Clinical Goals Reader from a JSON file to clinical goal objects.

    Attributes
    ----------
    clinicalgoalpath : str
        The path to the clinical goals JSON file.
    maskDict : dict
        A dictionary of masks with mask names as keys and mask arrays as values.
    clinical_goals_dict : dict
        A dictionary to store clinical in dictionary format loaded from the JSON file.
    """
    def __init__(self, clinicalgoalpath: str, maskDict: dict=None):
        self.path = clinicalgoalpath
        self.maskDict = maskDict
        self.clinical_goals_list = self.load_clinical_goals()


    def load_clinical_goals(self):
        goals_dict_list = self.load_json_list()
        goals_list = []
        for goal in goals_dict_list:
            clinical_goal_obj = self.ClinicalGoalFromDict(goal)
            goals_list.append(clinical_goal_obj)
        return goals_list

    def ClinicalGoalFromDict(self, goal_dict: dict) -> AbstractClinicalGoal:
        #add sufficient checks here
        maskName=goal_dict["ROI"],
        mask = self.maskDict[maskName]
        prescription=goal_dict["dose"],
        lower_is_better=goal_dict["lower_is_better"],
        priority=goal_dict.get("priority", 0)
        if goal_dict.["type"] == "Dmax":
            goal = DMaxClinicalGoal(prescription=prescription, mask=mask, maskName=maskName, lower_is_better=lower_is_better, priority=priority)
        elif goal_dict["type"] == "Dmin":
            goal = DMinClinicalGoal(prescription=prescription, mask=mask, maskName=maskName, lower_is_better=lower_is_better, priority=priority)
        elif goal_dict["type"] == "Dmean":
            goal = DMeanClinicalGoal(prescription=prescription, mask=mask, maskName=maskName, lower_is_better=lower_is_better, priority=priority)
        elif goal_dict["type"] == "DCC":
            volume = goal_dict["absolute_volume"]
            goal = DCCClinicalGoal(prescription=prescription, mask=mask, maskName=maskName, lower_is_better=lower_is_better,  priority=priority, **kwargs={'volume': volume})
        elif goal_dict["type"] == "VCC":
            volume = goal_dict["absolute_volume"]
            goal = VCCClinicalGoal(prescription=prescription, mask=mask, maskName=maskName, lower_is_better=lower_is_better, priority=priority, **kwargs={'volume': volume})
        elif goal_dict["type"] == "VX":
            volume = goal_dict["volume"]
            goal = VXClinicalGoal(prescription=prescription, mask=mask, maskName=maskName, lower_is_better=lower_is_better, priority=priority, **kwargs={'volume': volume})
        elif goal_dict["type"] == "DX":
            volume = goal_dict["volume"]
            goal = DXClinicalGoal(prescription=prescription, mask=mask, maskName=maskName, lower_is_better=lower_is_better, priority=priority, **kwargs={'volume': volume})
        return goal

    def load_json_list(self):
        with open(self.path, "r") as f:
            goals = json.load(f)
        return goals