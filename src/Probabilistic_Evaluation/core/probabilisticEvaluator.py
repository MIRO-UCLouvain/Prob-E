from Probabilistic_Evaluation.core.sampling._abstractSamplingMethod import AbstractsamplingMethod
from Probabilistic_Evaluation.core.scenariosGenerator import ScenariosGenerator
from Probabilistic_Evaluation.data import PatientData
from Probabilistic_Evaluation.data import DVH
import os

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
        self.scenarios = ScenariosGenerator(sampling_method=self.sampler).scenarios_list
        self.clinical_goals = self.patientData.clinicalGoalsList
        self.calc_VWMin = calc_VWMin
        self.calc_VWMax = calc_VWMax
        self.calc_CummulPR = calc_CummulPR
        self.prob_list = [p.probability for p in self.scenarios]
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

        n_scenarios = len(self.scenarios)
        for goal in self.clinical_goals:
            goal.valueList = [None] * n_scenarios
            goal.successList = [None] * n_scenarios

        threads = []
        for i, scenario in enumerate(self.scenarios):
            while threading.active_count() > self.nThreads:
                time.sleep(0.1)
            print(f"Starting thread for scenario {i+1}/{len(self.scenarios)}")
            t = threading.Thread(target=self.compute_and_evaluate_scenario, args=(i, scenario))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        
        self.calculate_passingRates()
        if self.calc_CummulPR:
            self.calculate_cummulPR()

    def compute_and_evaluate_scenario(self, scenario_idx, scenario):
        #print("Evaluating scenario with displacement:", scenario.displacement)
        scenario.compute_shifted_image(self.patientData.doseImage, scenario.displacement)
        dvh_dict = {}
        for mask in self.patientData.maskDict.keys():
            dvh_dict[mask] = DVH(dosemap=scenario.doseImage, mask=self.patientData.maskDict[mask], spacing=self.patientData.spacing)
        for goal in self.clinical_goals:
            goal.compute(dvh_dict[goal.maskName], scenario_idx=scenario_idx)
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
                'Dose threshold': goal_obj.prescription,
                'Passing Rate': goal_obj.passingRate
            })
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))

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
            remaining_successList = [x*y for x,y in zip(self.clinical_goals[i].successList, remaining_successList)]
            cummul_PR_current = sum(x*p for x,p in zip(remaining_successList, self.prob_list))

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
            if cummulPR_prev == 0:
                cummul_PR_relative = 0.0
            else:
                cummul_PR_relative = cummul_PR / cummulPR_prev
            cummul_PR_rel.append(cummul_PR_relative)
            cummulPR_prev = cummul_PR
            i+=1
        self.CummulPR_rel = cummul_PR_rel


    def write_to_csv(self, dir_path:str,proba:float=None):
        """
        Write evaluation results to a CSV file.

        Parameters
        ----------
        dir_path : str
            Directory path for the CSV file.
        proba : float, optional
            Probability value for the CSV file name.
        """
        self.calculate_passingRates()
        self.calculate_cummulPR()
        self.calculate_cummulPR_relative()
        data = []
        zero_disp = np.array([0, 0, 0])
        zero_idx = None
        for i, scenario in enumerate(self.scenarios):
            try:
                if np.allclose(np.asarray(scenario.displacement), zero_disp):
                    zero_idx = i
                    break
            except Exception:
                continue
        for i, goal in enumerate(self.clinical_goals):
            print(goal)
            print(self.PR[i])
            if zero_idx is not None and len(goal.valueList) > zero_idx:
                nominal_value = "{:.3f}".format(goal.valueList[zero_idx])
                nominal_passed = bool(goal.successList[zero_idx]) if len(goal.successList) > zero_idx else None
            else:
                nominal_value = "N/A"
                nominal_passed = None
            data.append({
                'Clinical Goal': goal.__str__(),
                'Nominal Scenario': nominal_value,
                'Nominal Scenario Passed': nominal_passed,
                'Passing Rate': "{:.3f}".format(self.PR[i]),
                'Cumulative Passing Rate': "{:.3f}".format(self.CummulPR[i]),
                'Cumulative Passing Rate (Relative)': "{:.3f}".format(self.CummulPR_rel[i])
            })
        #for each objective; store the success list
        for i, goal in enumerate(self.clinical_goals):
            for j, success in enumerate(goal.successList):
                data[i][f'Scenario {j+1} Success'] = success
            for j, scenario_prob in enumerate(self.prob_list):
                data[i][f'Scenario {j+1} Probability'] = float(scenario_prob)

        df = pd.DataFrame(data)
        # check if outpath exist and add number at the end if it does
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            i = 1
            while os.path.exists(f"{dir_path[:]}_{i}"):
                i += 1
            dir_path = f"{dir_path[:]}_{i}"
            os.makedirs(dir_path)

        out_path = os.path.join(dir_path, 'evaluation_results.csv')

        if proba is not None:
            out_path = out_path.replace('.csv', f'_{proba*100:.0f}proba.csv')
            if self.calc_VWMax:
                np.save(os.path.join(dir_path, f'VWMax{proba*100:.0f}.npy'), self.VWMax)
            if self.calc_VWMin:
                np.save(os.path.join(dir_path, f'VWMin{proba*100:.0f}.npy'), self.VWMin)

        df.to_csv(out_path, index=False)


    def display_tables(self):
        """
        display in a interface the passing rates and cumulative passing rates in a table format, with clinical goals as rows and passing rates as columns.
        """
        self.calculate_passingRates()
        self.calculate_cummulPR()
        self.calculate_cummulPR_relative()
        zero_disp = np.array([0, 0, 0])
        zero_idx = None
        for i, scenario in enumerate(self.scenarios):
            try:
                if np.allclose(np.asarray(scenario.displacement), zero_disp):
                    zero_idx = i
                    break
            except Exception:
                continue
        data = []
        for i, goal in enumerate(self.clinical_goals):
            row = {
                'Clinical Goal': goal.__str__(),
                'Passing Rate': "{:.3f}".format(self.PR[i])
            }
            if zero_idx is not None and len(goal.valueList) > zero_idx:
                row['Scenario [0,0,0]'] = "{:.3f}".format(goal.valueList[zero_idx])
            else:
                row['Scenario [0,0,0]'] = "N/A"
            if self.calc_CummulPR:
                row['Cumulative Passing Rate'] = "{:.3f}".format(self.CummulPR[i])
                row['Cumulative Passing Rate (Relative)'] = "{:.3f}".format(self.CummulPR_rel[i])
            data.append(row)
        
        df = pd.DataFrame(data)
        ordered_cols = ['Clinical Goal', 'Scenario [0,0,0]']
        ordered_cols += [c for c in df.columns if c not in ordered_cols]
        df = df[ordered_cols]

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 2.2), max(3, len(df) * 0.6)))
        ax.axis('off')

        import textwrap
        from matplotlib import cm

        def value_to_color(val: float, alpha: float = 0.6):
            try:
                v = float(val)
            except (TypeError, ValueError):
                return None
            v = max(0.0, min(1.0, v))
            color = list(cm.get_cmap('RdYlGn')(v))
            color[3] = alpha
            return tuple(color)

        def text_color_for_bg(color) -> str:
            if not color:
                return 'black'
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                r, g, b = color[0], color[1], color[2]
            else:
                return 'black'
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            return 'black' if luminance > 0.6 else 'white'

        def wrap_label(label, max_len=14):
            if len(label) <= max_len:
                return label
            parts = textwrap.wrap(label, width=max_len)
            if len(parts) <= 2:
                return "\n".join(parts)
            return "\n".join([parts[0], " ".join(parts[1:])])

        col_labels = [wrap_label(col) for col in df.columns]

        table = ax.table(
            cellText=df.values,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.4)

        # Color numeric cells based on thresholds
        for (row_idx, col_idx), cell in table.get_celld().items():
            if row_idx == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4C72B0')
                continue

            col_name = df.columns[col_idx]
            if col_name in ['Passing Rate', 'Cumulative Passing Rate', 'Cumulative Passing Rate (Relative)']:
                bg = value_to_color(df.iloc[row_idx - 1, col_idx])
                if bg is None:
                    continue
                cell.set_facecolor(bg)
                cell.set_text_props(color=text_color_for_bg(bg))
            elif col_name == 'Scenario [0,0,0]':
                if zero_idx is None:
                    continue
                try:
                    success = self.clinical_goals[row_idx - 1].successList[zero_idx]
                except Exception:
                    continue
                if success:
                    cell.set_text_props(color='green')
                else:
                    cell.set_text_props(color='red')

        plt.tight_layout()
        plt.show()

    def displayVminVmax(self):
        """
        Display voxel-wise minimum and maximum dose images.
        """
        import matplotlib.pyplot as plt
        z_idx = 110
        if self.calc_VWMin and self.calc_VWMax:
            plt.imshow(self.patientData.ctImage[:,:,z_idx], cmap='gray')
            plt.contour(self.patientData.maskDict['CTVp_High'][:,:,z_idx], levels=[0.5], colors='red', linewidths=0.5)
            plt.imshow(self.VWMin[:,:,z_idx], cmap='jet',alpha=0.5)
            plt.colorbar(label='Dose')
            plt.title('VWMin Dose Distribution (Slice {0})'.format(z_idx))
            plt.show()

            plt.imshow(self.patientData.ctImage[:,:,z_idx], cmap='gray')
            plt.contour(self.patientData.maskDict['CTVp_High'][:,:,z_idx], levels=[0.5], colors='red', linewidths=0.5)
            plt.imshow(self.VWMax[:,:,z_idx], cmap='jet',alpha=0.5)
            plt.colorbar(label='Dose')
            plt.title('VWMax Dose Distribution (Slice {0})'.format(z_idx))
            plt.show()

            #diff between VWMax and VWMin
            plt.imshow(self.patientData.ctImage[:,:,z_idx], cmap='gray')
            plt.contour(self.patientData.maskDict['CTVp_High'][:,:,z_idx], levels=[0.5], colors='red', linewidths=0.5)
            plt.imshow(self.VWMax[:,:,z_idx]-self.VWMin[:,:,z_idx], cmap='jet',alpha=0.5)
            plt.colorbar(label='Dose Difference')
            plt.title('VWMax - VWMin Dose Distribution (Slice {0})'.format(z_idx))
            plt.show()