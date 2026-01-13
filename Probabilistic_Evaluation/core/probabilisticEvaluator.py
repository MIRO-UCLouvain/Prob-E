from data._DVH import DVH
from data._scenario import Scenario
from data._patientData import PatientData
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ProbabilisticEvaluator:

    def __init__(self, scenarios:[Scenario], priority_list: list[str]=None):
        self.scenarios = scenarios
        self.clinical_goals = self.scenarios[0].patientData.clinicalGoalsDict 
        self.priority_list = priority_list #assume list with names of clinical goals in order of priority
        for goal in self.clinical_goals.keys():
            PR = 0
            for s in scenarios:
                if s.clinicalGoalsValuesAchieved[goal]:
                    PR += s.scenarioProbability
                
            self.clinical_goals[goal].SetPassingRate(PR) #need passing rate attribute in clinical goal class

    def passingRate_table(self):
        """
        Print a table with clinical goals, prescribed doses, and passing rates.
        """
        data = []
        for goal_name, goal_obj in self.clinical_goals.items():
            data.append({
                'Clinical Goal': goal_name,
                'Dose treshold': goal_obj.prescription,
                'Passing Rate': goal_obj.passingRate
            })
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
    
    def calculate_cummul_PR(self):
        """
        need some form of prioritization of clinical goals to calculate a cummulative passing rate
        """
        i=1
        cummul_PR_prev = self.clinical_goals[self.priority_list[0]].passingRate
        self.clinical_goals[self.priority_list[0]].SetCummulative_passingRate(cummul_PR_prev)
        scenarios_left = [self.scenarios[j] for j in range(len(self.scenarios)) if self.scenarios[j].clinicalGoalsValuesAchieved[self.priority_list[0]]] #should i copy?
        while i < len(self.priority_list): 
            goal_name = self.priority_list[i]
            indices_to_keep = [i for i, x in enumerate(scenarios_left) if x.clinicalGoalsValuesAchieved[goal_name]]
            cummul_PR = sum([scenarios_left[j].scenarioProbability for j in indices_to_keep])
            scenarios_left = [scenarios_left[j] for j in indices_to_keep]
            #two options either absolute cummulative passing rate or relative cummulative passing rate
            #1. absolute
            self.clinical_goals[goal_name].SetCummulative_passingRate(cummul_PR)
            #2. relative
            #self.clinical_goals[goal_name].SetCummulative_passingRate(cummul_PR / cummul_PR_prev)

            cummul_PR_prev = cummul_PR
            i+=1  

    #Next three functions assume that all (displaced) dose images have the same shape
    def voxelwise_min_map(self):
        """
        Compute voxel-wise minimum dose map across all scenarios.
        """
        dose_maps = [s.doseImageScenario for s in self.scenarios]
        min_map = np.minimum.reduce(dose_maps)
        return min_map
    
    def voxelwise_max_map(self):
        """
        Compute voxel-wise maximum dose map across all scenarios.
        """
        dose_maps = [s.doseImageScenario for s in self.scenarios]
        max_map = np.maximum.reduce(dose_maps)
        return max_map
    
    def prob_dose_map(self):
        return np.sum([s.doseImageScenario * s.scenarioProbability for s in self.scenarios], axis=0)