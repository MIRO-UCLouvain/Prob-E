import json
from Probabilistic_Evaluation.Evaluation.clinicalgoals import *

class clinicalgoalsreader():

    def __init__(self, clinicalgoalpath: str, maskDict: dict=None):
        self.path = clinicalgoalpath
        self.maskDict = maskDict
        self.clinical_goals_dict = self.load_clinical_goals()


    def load_clinical_goals(self):
        goals_list = self.load_json_list()
        clinical_goals_list = []
        for goal in goals_list:
            clinical_goal_obj = self.ClinicalGoalFromDict(goal)
            clinical_goals_list.append(clinical_goal_obj)
        return clinical_goals_list

    def ClinicalGoalFromDict(self, goal_dict: dict) -> ClinicalGoal:
        #add sufficient checks here
        maskName=goal_dict["ROI"],
        mask = self.maskDict[maskName]
        prescription=goal_dict["dose"],
        lower_is_better=goal_dict["lower_is_better"],
        priority=goal_dict["priority"]
        if goal_dict.["type"] == "Dmax":
            goal = DMaxClinicalGoal()

        return goal

    def load_json_list(self):
        with open(self.path, "r") as f:
            goals = json.load(f)
        return goals