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

    """
    A class to evaluate clinical goals across probabilistic scenarios.

    Attributes
    ----------
    patientData : PatientData
        patientData object containing dose, masks, and clinical goals.
    sampler : AbstractsamplingMethod
        Sampling method used to generate scenarios.
    scenarios : list
        List of generated scenarios for evaluation.
    clinical_goals : list
        List of clinical goals associated with the patient.
    calc_VWMin : bool
        Whether to compute voxel-wise minimum dose across scenarios.
    calc_VWMax : bool
        Whether to compute voxel-wise maximum dose across scenarios.
    calc_CummulPR : bool
        Whether to compute cumulative passing rates.
    prob_list : list
        Scenario probabilities in the same order as scenarios.
    VWMin : np.ndarray or None
        Voxel-wise minimum dose image if enabled.
    VWMax : np.ndarray or None
        Voxel-wise maximum dose image if enabled.
    PR : list or None
        Passing rates for each clinical goal.
    CummulPR : list or None
        Cumulative passing rates for each clinical goal.
    CummulPR_rel : list or None
        Relative cumulative passing rates for each clinical goal.

    Methods
    -------
    evaluate()
        Evaluate all scenarios and compute clinical goal values and passing rates.
    calculate_passingRates()
        Compute passing rates for each clinical goal.
    calculate_cummulPR()
        Compute cumulative passing rates.
    calculate_cummulPR_relative()
        Compute relative cumulative passing rates.
    write_to_csv(out_path: str)
        Write evaluation results to a CSV file.
    """

    def __init__(self, PatientData:PatientData, sampler:AbstractsamplingMethod, calc_VWMin:bool=False, calc_VWMax:bool=False, calc_CummulPR:bool=False,nThreads:int=-1):
        """
        Initialize a probabilistic evaluator.

        Parameters
        ----------
        PatientData : PatientData
            Patient data object containing dose, masks, and clinical goals.
        sampler : AbstractsamplingMethod
            Sampling method used to generate scenarios.
        calc_VWMin : bool, optional
            Whether to compute voxel-wise minimum dose across scenarios.
        calc_VWMax : bool, optional
            Whether to compute voxel-wise maximum dose across scenarios.
        calc_CummulPR : bool, optional
            Whether to compute cumulative passing rates.
        """
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
        self.PR = None
        if nThreads < 1:
            self.nThreads = multiprocessing.cpu_count()-1
        else:
            self.nThreads = nThreads
        self.CummulPR_rel = None


    def evaluate(self):
        """
        Evaluate scenarios and compute clinical goal values and passing rates.
        """
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
        self.calculate_passingRates()
        if self.calc_CummulPR:
            self.calculate_cummulPR()
            self.calculate_cummulPR_relative()

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
        """
        Compute passing rates for each clinical goal.
        """
        PR = []
        for goal in self.clinical_goals:
            p = 0.0
            for i, s in enumerate(goal.successList):
                if s:
                    p += self.prob_list[i]
            PR.append(p)
        self.PR = PR

    def calculate_cummulPR(self):
        """
        Compute cumulative passing rates for ordered clinical goals.
        """
      
        cummul_PR = []
        i=1
        cummul_PR_prev = self.PR[0]
        cummul_PR.append(cummul_PR_prev)
        remaining_successList = self.clinical_goals[0].successList
        while i < len(self.clinical_goals):
            remaining_succesList = [x*y for x,y in (self.clinical_goals[i].succesList,remaining_succesList)]
            cummul_PR_current = sum(x*p for x,p in zip(remaining_succesList, self.prob_list))

            cummul_PR.append(cummul_PR_current)
            
            cummul_PR_prev = cummul_PR_current
            i+=1
        self.CummulPR = cummul_PR

    def calculate_cummulPR_relative(self):
        """
        Compute relative cumulative passing rates.
        """
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

        Parameters
        ----------
        out_path : str
            Output path for the CSV file.
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
                data[-1]['Cummulative Passing Rate (Relative)'] = "{:.3f}".format(self.CummulPR_rel[i])

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