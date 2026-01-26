from Probabilistic_Evaluation.core.sampling._abstractSamplingMethod import AbstractsamplingMethod
from Probabilistic_Evaluation.core.scenariosGenerator import ScenariosGenerator
from Probabilistic_Evaluation.data import PatientData
from Probabilistic_Evaluation.data import DVH
import numpy as np
import pandas as pd

class ProbabilisticEvaluator:

    def __init__(self, PatientData:PatientData, sampler:AbstractsamplingMethod, calc_VWMin:bool=False, calc_VWMax:bool=False, calc_CummulPR:bool=False):
        self.patientData = PatientData
        self.sampler = sampler
        self.scenarios = None
        self.clinical_goals = self.patientData.clinicalGoalsList
        self.calc_VWMin = calc_VWMin
        self.calc_VWMax = calc_VWMax
        self.calc_CummulPR = calc_CummulPR
        self.prob_list = [p.probability for p in self.patientData.scenarios_list]
        self.VWMin = None
        self.VWMax = None
        if self.calc_VWMin:
            self.VWMin = self.patientData.doseImage
        if self.calc_VWMax:
            self.VWMax = np.zeros_like(self.patientData.doseImage)   
        self.PR = None
        self.CummulPR = None
        self.CummulPR_rel = None


    def evaluate(self):
        self.scenarios = ScenariosGenerator(sampling_method=self.sampler).scenarios_list

        for scenario in self.scenarios:
            print("Evaluating scenario with displacement:", scenario.displacement)
            scenario.compute_shifted_image(self.patientData.doseImage, scenario.displacement)
            print("Shifted dose image computed.")
            dvh_dict = {}
            for mask in self.patientData.maskDict.keys():
                print(f"Computing DVH for mask: {mask}")
                dvh_dict[mask] = DVH(dosemap=scenario.doseImage, mask=self.patientData.maskDict[mask], spacing=self.patientData.spacing)
                print(f"DVH computed for mask: {mask}")
            for goal in self.clinical_goals:
                print(f"Computing value for clinical goal: {goal.maskName}")
                goal.compute_value(dvh_dict[goal.maskName])
                print(f"Value computed for clinical goal: {goal.maskName}")
            if self.calc_VWMin:
                self.VWMin = np.minimum(self.VWMin, scenario.doseImage)
            if self.calc_VWMax:
                self.VWMax = np.maximum(self.VWMax, scenario.doseImage)
            scenario.delete_doseImage()
        self.calculate_passingRates()
        if self.calc_CummulPR:
            self.calculate_cummulPR()
            self.calculate_cummulPR_relative()

    def calculate_passingRates(self):
        PR = []
        for goal in self.clinical_goals.values():
            p = 0.0
            for i, s in enumerate(goal.scenariosValuesAchieved):

                if s:
                    p += self.prob_list[i]
            PR.append(p)
        self.PR = PR

    def calculate_cummulPR(self):
      
        cummul_PR = []
        i=1
        cummul_PR_prev = self.PR[0]
        cummul_PR.append(cummul_PR_prev)
        remaining_succesList = self.clinical_goals[0].succesList
        while i < len(self.clinical_goals):
            remaining_succesList = [x*y for x,y in (self.clinical_goals[i].succesList,remaining_succesList)]
            cummul_PR_current = sum(x*p for x,p in zip(remaining_succesList, self.prob_list))

            cummul_PR.append(cummul_PR_current)
            
            cummul_PR_prev = cummul_PR_current
            i+=1
        self.CummulPR = cummul_PR

    def calculate_cummulPR_relative(self):
        cummul_PR_rel = []
        i=1
        cummulPR_prev= self.CummulPR[0]
        cummul_PR_rel.append(cummulPR_prev)
        while i < len(self.clinical_goals):
            cummul_PR = self.CummulPR[i]
            cummul_PR_relative = cummul_PR / cummulPR_prev
            cummul_PR_rel.append(cummul_PR_relative)
            cummulPR_prev = cummul_PR
            i+=1
        self.CummulPR_rel = cummul_PR_rel


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
                data[-1]['Cummulative Passing Rate (Relative)'] = self.CummulPR_rel[i]
        
        df = pd.DataFrame(data)
        df.to_csv(out_path, index=False)