from Probabilistic_Evaluation.core.scenariosGenerator import ScenariosGenerator
from Probabilistic_Evaluation.data import PatientData
import numpy as np
import pandas as pd

class ProbabilisticEvaluator:

    def __init__(self, PatientData:PatientData, calc_VWMin:bool=False, calc_VWMax:bool=False, calc_CummulPR:bool=False):
        self.patientData = PatientData
        self.scenarios = None
        self.clinical_goals = self.patientData.clinicalGoalsList
        self.calc_VWMin = calc_VWMin
        self.calc_VWMax = calc_VWMax
        self.calc_CummulPR = calc_CummulPR
        self.VWMin = None
        self.VWMax = None
        if self.calc_VWMin:
            self.VWMin = self.patientData.doseImage
        if self.calc_VWMax:
            self.VWMax = np.zeros_like(self.patientData.doseImage)
        self.CummulPR = None
        self.PR = None


    def evaluate(self):
        self.scenarios = ScenariosGenerator(self.patientData).scenarios_list

        for scenario in self.scenarios:
            scenario.compute_shifted_image(self.patientData.doseImage, scenario.displacement)
            dvh_dict = {}
            for mask in self.patientData.maskDict.keys():
                dvh_dict[mask] = DVH(doseImage=scenario.doseImage, structureMask=self.patientData.maskDict[mask], spacing=self.patientData.spacing)
            for goal in self.clinical_goals:
                goal.computevalue(dvh_dict[goal.maskName])
            if self.calc_VWMin:
                self.VWMin = np.minimum(self.VWMin, scenario.doseImage)
            if self.calc_VWMax:
                self.VWMax = np.maximum(self.VWMax, scenario.doseImage)
            scenario.delete_doseImage()
        if self.calc_CummulPR:
            self.calc_CummulPR()



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
    

    def calculate_passingRates(self):
        PR = []
        for goal in self.clinical_goals.values():
            p = 0.0
            for i, s in enumerate(goal.scenariosValuesAchieved):

                if s:
                    p += self.scenarios[i].probability
            PR.append(p)
        self.PR = PR

    def calculate_cummul_PR(self):
        """
        need some form of prioritization of clinical goals to calculate a cummulative passing rate
        """
        cummul_PR = []
        i=1
        cummul_PR_prev = self.PR[0]
        cummul_PR.append(cummul_PR_prev)
        remaining_succesList = self.clinical_goals[0].succesList
        while i < len(self.clinical_goals):
            remaining_succesList = [x*y for x,y in (self.clinical_goals[i].succesList,remaining_succesList)]
            cummul_PR_current = sum([self.scenarios[j].probability for j, x in enumerate(remaining_succesList) if x])

            #two options either absolute cummulative passing rate or relative cummulative passing rate
            #1. absolute
            cummul_PR.append(cummul_PR_current)
            #2. relative
            #self.clinical_goals[goal_name].SetCummulative_passingRate(cummul_PR / cummul_PR_prev)
            #cummul_PR = cummul_PR_current*cummul_PR_prev
            cummul_PR_prev = cummul_PR_current
            i+=1
        self.CummulPR = cummul_PR



    def write_to_csv(self, out_path:str):
        """
        Write evaluation results to a CSV file.
        """
        data = []
        for i, goal in enumerate(self.clinical_goals):
            data.append({
                'Clinical Goal': goal.__str__(),
                'Passing Rate': self.PR[i],
            })
            if self.calc_CummulPR:
                data[-1]['Cummulative Passing Rate'] = self.CummulPR[i]
        
        df = pd.DataFrame(data)
        df.to_csv(out_path, index=False)