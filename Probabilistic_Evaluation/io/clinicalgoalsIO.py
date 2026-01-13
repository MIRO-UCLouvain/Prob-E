import json
from data.clinicalGoals._clinicalGoal import ClinicalGoal

class clinicalgoalsreader():

    def __init__(self, clinicalgoalpath: str):
        self.path = clinicalgoalpath
        self.clinical_goals_dict = self.load_clinical_goals()


    def load_json_list(self):
        with open(self.path, "r") as f:
            goals = json.load(f)
        return goals