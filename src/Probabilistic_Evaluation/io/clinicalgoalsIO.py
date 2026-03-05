import json


from Probabilistic_Evaluation.data.clinicalGoals._clinicalGoal import AbstractClinicalGoal
from Probabilistic_Evaluation.data.clinicalGoals import * 
from opentps.core.data.images import CTImage

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

    Methods
    -------
    load_JSON_list(clinicalgoalpath: str)
        Load a JSON list of clinical goals and populate the internal list.
    load_clinical_goals() -> list
        Create clinical goal objects from the JSON list.
    ClinicalGoalFromDict(goal_dict: dict) -> AbstractClinicalGoal
        Build a clinical goal object from a dictionary.
    load_json_list() -> list
        Read the clinical goals JSON file.
    """

    def __init__(self, maskDict: dict=None,ct_image:CTImage=None):
        self._maskDict = maskDict
        self._ct = ct_image
        self._path = None
        self.clinical_goals_dict: dict = None
        self._clinical_goals_list: list = None
        self._resampledMasks = {}

    @property
    def maskDict(self) -> dict:
        return self._maskDict
    @maskDict.setter
    def maskDict(self, newMaskDict: dict):
        self._maskDict = newMaskDict

    @property
    def resampledMasks(self) -> dict:
        return self._resampledMasks

    @resampledMasks.setter
    def resampledMasks(self, newResampledMasks: dict):
        self._resampledMasks = newResampledMasks

    @property
    def path(self) -> str:
        return self._path
    @path.setter
    def path(self, newPath: str):
        self._path = newPath

    @property
    def clinical_goals_dict(self) -> dict:
        return self._clinical_goals_dict
    @clinical_goals_dict.setter
    def clinical_goals_dict(self, newClinicalGoalsDict: dict):
        self._clinical_goals_dict = newClinicalGoalsDict

    @property
    def clinical_goals_list(self) -> list:
        return self._clinical_goals_list
    
    @clinical_goals_list.setter
    def clinical_goals_list(self, newClinicalGoalsList: list):
        self._clinical_goals_list = newClinicalGoalsList


    def load_JSON_list(self, clinicalgoalpath: str):
        self.path = clinicalgoalpath
        self._clinical_goals_list = self.load_clinical_goals()
        

    def load_clinical_goals(self):
        goals_dict_list = self.load_json_list()
        goals_list = []
        for goal in goals_dict_list:
            clinical_goal_obj = self.ClinicalGoalFromDict(goal)
            goals_list.append(clinical_goal_obj)
        goals_list.sort(key=lambda x: x.priority, reverse=True)
        return goals_list

    def ClinicalGoalFromDict(self, goal_dict: dict) -> AbstractClinicalGoal:
        #add sufficient checks here
        maskName=goal_dict["ROI"]
        mask = self.maskDict[maskName].getBinaryMask(origin=self._ct.origin, gridSize=self._ct.gridSize, spacing=self._ct.spacing).imageArray
        self.resampledMasks[maskName] = mask
        dose =goal_dict["dose"]
        lower_is_better=goal_dict["lower_is_better"]
        priority=goal_dict.get("priority", 0)
        type = goal_dict["type"].upper()
     
        if type == "DMAX":         
            goal = DMaxClinicalGoal(prescription=dose, mask=mask, maskName=maskName, priority=priority, lower_is_better=lower_is_better)
        elif type == "DMIN":       
            goal = DMinClinicalGoal(prescription=dose, mask=mask, maskName=maskName, priority=priority, lower_is_better=lower_is_better)
        elif type == "DMEAN":       
            goal = DMeanClinicalGoal(prescription=dose, mask=mask, maskName=maskName, priority=priority, lower_is_better=lower_is_better)
        elif type == "DXCC":         
            volume = goal_dict["absolute_volume"]
            goal = DCCClinicalGoal(prescription=dose, mask=mask, maskName=maskName, priority=priority, lower_is_better=lower_is_better,  volume=volume)
        elif type == "VXCC": 
            volume = goal_dict["absolute_volume"]
            goal = VCCClinicalGoal(prescription=volume, mask=mask, maskName=maskName, priority=priority, lower_is_better=lower_is_better, dose=dose)
        elif type == "VX":
            volume = goal_dict["volume"]/100.0
            goal = VXClinicalGoal(prescription=volume, mask=mask, maskName=maskName, lower_is_better=lower_is_better, priority=priority, dose=dose)
        elif type == "DX":
            volume = goal_dict["volume"]/100.0
            goal = DXClinicalGoal(prescription=dose, mask=mask, maskName=maskName, lower_is_better=lower_is_better, priority=priority, volume=volume)
        else:
            raise ValueError(f"Unknown clinical goal type: {goal_dict['type']}")
        return goal

    def load_json_list(self):
        with open(self.path, "r") as f:
            goals = json.load(f)
        return goals
    
