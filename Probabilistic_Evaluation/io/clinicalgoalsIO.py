import json


from Probabilistic_Evaluation.data.clinicalGoals import *

class clinicalgoalsreader():
    """
    A class to represent a Clinical Goals Reader from a JSON file to clinical goal objects.

    Attributes
    ----------
    maskDict : dict
        A dictionary of masks with mask names as keys and mask arrays as values.
    clinicalgoalpath : str
        The path to the clinical goals JSON file.
    clinical_goals_dict : dict
        A dictionary to store clinical in dictionary format loaded from the JSON file.
    clinical_goals_list : list
        A list to store all AbstractClinicalGoal objects created from the clinical_goals_dict.
    """
    def __init__(self, maskDict: dict=None):
        self.maskDict = maskDict
        self.path = None
        self.clinical_goals_dict = None
        self._clinical_goals_list = None

    @property
    def maskDict(self) -> dict:
        return self._maskDict
    @maskDict.setter
    def maskDict(self, newMaskDict: dict):
        self._maskDict = newMaskDict
    
    @property
    def path(self) -> str:
        return self._path
    @path.setter
    def path(self, newPath: str):
        self._path = newPath

    @property
    def clinical_goals_list(self) -> list:
        return self._clinical_goals_list
    @clinical_goals_list.setter
    def clinical_goals_list(self, newClinicalGoalsList: list):
        self._clinical_goals_list = newClinicalGoalsList

    def load_JSON_list(self, clinicalgoalpath: str, ):
        self.path = clinicalgoalpath
        self._clinical_goals_listclinical_goals_list = self.load_clinical_goals()

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
            goal = DMaxClinicalGoal(prescription=prescription, mask=mask, maskName=maskName, priority=priority, lower_is_better=lower_is_better)
        elif goal_dict["type"] == "Dmin":
            goal = DMinClinicalGoal(prescription=prescription, mask=mask, maskName=maskName, priority=priority, lower_is_better=lower_is_better)
        elif goal_dict["type"] == "Dmean":
            goal = DMeanClinicalGoal(prescription=prescription, mask=mask, maskName=maskName, priority=priority, lower_is_better=lower_is_better)
        elif goal_dict["type"] == "DCC":
            volume = goal_dict["absolute_volume"]
            goal = DCCClinicalGoal(prescription=prescription, mask=mask, maskName=maskName, priority=priority, lower_is_better=lower_is_better,  **kwargs={'volume': volume})
        elif goal_dict["type"] == "VCC":
            volume = goal_dict["absolute_volume"]
            goal = VCCClinicalGoal(prescription=prescription, mask=mask, maskName=maskName, priority=priority, lower_is_better=lower_is_better, **kwargs={'volume': volume})
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