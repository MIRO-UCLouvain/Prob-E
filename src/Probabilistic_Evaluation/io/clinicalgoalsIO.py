import json
import numpy as np
import time

from Probabilistic_Evaluation.data.clinicalGoals._clinicalGoal import AbstractClinicalGoal
from Probabilistic_Evaluation.data.clinicalGoals import * 
from Probabilistic_Evaluation.logging_utils import log_call, logger
from Probabilistic_Evaluation.utils import timed, get_partial_volume_mask


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

    def __init__(self, maskDict: dict=None, spacing=None, gridSize=None, origin=None):
        self._maskDict = maskDict
        self._spacing = spacing
        self._gridSize = gridSize
        self._origin = origin
        self._path = None
        self.clinical_goals_dict: dict = None
        self._clinical_goals_list: list = None
        self._proba_goals_list: list = None
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

    @property
    def proba_goals_list(self) -> list:
        return self._proba_goals_list 

    @timed
    def load_JSON_list(self, clinicalgoalpath: str):
        self.path = clinicalgoalpath
        self._clinical_goals_list, self._proba_goals_list = self.load_clinical_goals()
        

    def load_clinical_goals(self):
        goals_dict_list = self.load_json_list()
        goals_list = []
        proba_goals_list = []
        for goal in goals_dict_list:
            clinical_goal_obj = self.ClinicalGoalFromDict(goal)
            if clinical_goal_obj.probabilistic:
                proba_goals_list.append(clinical_goal_obj)
            goals_list.append(clinical_goal_obj)
        if all(hasattr(goal, "priority") for goal in goals_list):
            goals_list.sort(key=lambda x: x.priority)
            proba_goals_list.sort(key=lambda x: x.priority)
        
        return goals_list, proba_goals_list
    @log_call(log_result=True)
    def ClinicalGoalFromDict(self, goal_dict: dict) -> AbstractClinicalGoal:
        #add sufficient checks here
        if goal_dict["ROI"] not in self.resampledMasks.keys():

            start_time = time.time()
            maskName=goal_dict["ROI"]            
            maskPartialVolume = get_partial_volume_mask(
                contour=self.maskDict[maskName],
                origin=self._origin,
                gridSize=self._gridSize,
                spacing=self._spacing,
                precision= 16
            )
            stop_time = time.time()

            print(f"finished resampling partial volume mask for {goal_dict['ROI']} in {stop_time - start_time:.2f} seconds")
            mask = maskPartialVolume
            self.resampledMasks[maskName] = mask
            logger.info(f"Resampled mask for ROI: {maskName} with shape: {mask.shape} and spacing: {self._spacing}")
        else: 
            maskName = goal_dict["ROI"]
            mask = self.resampledMasks[maskName]
        

        dose =goal_dict["dose"]
        lower_is_better=goal_dict["lower_is_better"]
        priority=goal_dict.get("priority",0)
        probabilistic=goal_dict.get("probabilistic", False)
        type = goal_dict["type"].upper()

        if type == "DMAX":         
            goal = DMaxClinicalGoal(prescription=dose, mask=mask, maskName=maskName, priority=priority,lower_is_better=lower_is_better)
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
            goal = VXClinicalGoal(prescription=volume, mask=mask, maskName=maskName, priority=priority, lower_is_better=lower_is_better, dose=dose)
        elif type == "DX":
            volume = goal_dict["volume"]/100.0
            goal = DXClinicalGoal(prescription=dose, mask=mask, maskName=maskName, priority=priority, lower_is_better=lower_is_better, volume=volume)
        else:
            raise ValueError(f"Unknown clinical goal type: {goal_dict['type']}")
        if probabilistic:
            goal.probabilistic = True
        logger.debug(f"Creating clinical goal for ROI: {maskName}, type: {type}, number of voxels in mask: {np.sum(mask)}")
        return goal

    
    def load_json_list(self):
        with open(self.path, "r") as f:
            goals = json.load(f)
        return goals
    
