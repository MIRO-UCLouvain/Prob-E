from Probabilistic_Evaluation.core.sampling._abstractSamplingMethod import AbstractsamplingMethod
from Probabilistic_Evaluation.core.scenariosGenerator import ScenariosGenerator
from Probabilistic_Evaluation.data import PatientData
from Probabilistic_Evaluation.data import DVH

import threading
import multiprocessing
import time
import numpy as np
import pandas as pd

class ProbabilisticEvaluator:

    def __init__(self, PatientData:PatientData, sampler:AbstractsamplingMethod, calc_VWMin:bool=False, calc_VWMax:bool=False, calc_CummulPR:bool=False,nThreads:int=-1):
        self.patientData = PatientData
        self.sampler = sampler
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
        if nThreads < 1:
            self.nThreads = multiprocessing.cpu_count()-1
        else:
            self.nThreads = nThreads


    def evaluate(self):
        self.scenarios = ScenariosGenerator(sampling_method=self.sampler).scenarios_list

        threads = []
        for i, scenario in enumerate(self.scenarios):
            while threading.active_count() >= self.nThreads:
                time.sleep(0.1)
            print(f"Starting thread for scenario {i+1}/{len(self.scenarios)}")
            t = threading.Thread(target=self.compute_and_evaluate_scenario, args=(scenario,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

            # print("Evaluating scenario with displacement:", scenario.displacement)
            # scenario.compute_shifted_image(self.patientData.doseImage, scenario.displacement)
            # print("Shifted dose image computed.")
            # dvh_dict = {}
            # for mask in self.patientData.maskDict.keys():
            #     print(f"Computing DVH for mask: {mask}")
            #     dvh_dict[mask] = DVH(dosemap=scenario.doseImage, mask=self.patientData.maskDict[mask], spacing=self.patientData.spacing)
            #     print(f"DVH computed for mask: {mask}")
            # for goal in self.clinical_goals:
            #     print(f"Computing value for clinical goal: {goal.maskName}")
            #     goal.compute_value(dvh_dict[goal.maskName])
            #     print(f"Value computed for clinical goal: {goal.maskName}")
            # if self.calc_VWMin:
            #     self.VWMin = np.minimum(self.VWMin, scenario.doseImage)
            # if self.calc_VWMax:
            #     self.VWMax = np.maximum(self.VWMax, scenario.doseImage)
            # scenario.delete_doseImage()
        
        self.calculate_passingRates()
        if self.calc_CummulPR:
            self.calculate_cummul_PR()

    def compute_and_evaluate_scenario(self, scenario):
        #print("Evaluating scenario with displacement:", scenario.displacement)
        scenario.compute_shifted_image(self.patientData.doseImage, scenario.displacement)
        dvh_dict = {}
        for mask in self.patientData.maskDict.keys():
            dvh_dict[mask] = DVH(dosemap=scenario.doseImage, mask=self.patientData.maskDict[mask], spacing=self.patientData.spacing)
        for goal in self.clinical_goals:
            goal.compute(dvh_dict[goal.maskName])
        if self.calc_VWMin:
            self.VWMin = np.minimum(self.VWMin, scenario.doseImage)
        if self.calc_VWMax:
            self.VWMax = np.maximum(self.VWMax, scenario.doseImage)
        scenario.delete_doseImage()

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
        for goal in self.clinical_goals:
            p = 0.0
            for i, s in enumerate(goal.successList):
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
        remaining_successList = self.clinical_goals[0].successList
        while i < len(self.clinical_goals):
            remaining_successList = [x*y for x,y in zip(self.clinical_goals[i].successList,remaining_successList)]
            cummul_PR_current = sum([self.scenarios[j].probability for j, x in enumerate(remaining_successList) if x])

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
            print(goal)
            print(self.PR[i])
            data.append({
                'Clinical Goal': goal.__str__(),
                'Passing Rate': "{:.3f}".format(self.PR[i]),
            })
            if self.calc_CummulPR:
                data[-1]['Cummulative Passing Rate'] = "{:.3f}".format(self.CummulPR[i])

        import matplotlib.pyplot as plt
        z_idx = 110
        plt.imshow(self.patientData.ctImage[:,:,z_idx], cmap='gray')
        plt.contour(self.patientData.maskDict['PTVp_High'][:,:,z_idx], levels=[0.5], colors='red', linewidths=0.5)
        plt.imshow(self.VWMin[:,:,z_idx], cmap='jet',alpha=0.5)
        plt.colorbar(label='Dose')
        plt.title('VWMin Dose Distribution (Slice 99)')
        plt.show()

        plt.imshow(self.patientData.ctImage[:,:,z_idx], cmap='gray')
        plt.contour(self.patientData.maskDict['PTVp_High'][:,:,z_idx], levels=[0.5], colors='red', linewidths=0.5)
        plt.imshow(self.VWMax[:,:,z_idx], cmap='jet',alpha=0.5)
        plt.colorbar(label='Dose')
        plt.title('VWMax Dose Distribution (Slice 99)')
        plt.show()

        #diff between VWMax and VWMin
        plt.imshow(self.patientData.ctImage[:,:,z_idx], cmap='gray')
        plt.contour(self.patientData.maskDict['PTVp_High'][:,:,z_idx], levels=[0.5], colors='red', linewidths=0.5)
        plt.imshow(self.VWMax[:,:,z_idx]-self.VWMin[:,:,z_idx], cmap='jet',alpha=0.5)
        plt.colorbar(label='Dose Difference')
        plt.title('VWMax - VWMin Dose Distribution (Slice 99)')
        plt.show()
        
        df = pd.DataFrame(data)
        df.to_csv(out_path, index=False)