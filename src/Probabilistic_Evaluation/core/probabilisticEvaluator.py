from docutils.nodes import header

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
    PatientData : PatientData
        Patient data object containing dose, masks, and clinical goals.
    sampler : AbstractsamplingMethod
        Sampling method used to generate scenarios.
    computeVWMin : bool, optional
        Whether to compute voxel-wise minimum dose across scenarios.
    computeVWMax : bool, optional
        Whether to compute voxel-wise maximum dose across scenarios.
    computeCumulativePassingRate : bool, optional
        Whether to compute cumulative passing rates.
    scenarios : list
        A list of generated scenarios for evaluation.
    prob_list : np.ndarray
        An array of probabilities corresponding to each scenario.
    VWMin : np.ndarray or None
        Voxel-wise minimum dose image across scenarios, if computeVWMin is True.
    VWMax : np.ndarray or None
        Voxel-wise maximum dose image across scenarios, if computeVWMax is True.
    nominal_index : int or None
        The index of the nominal scenario (displacement close to [0,0,0]),
        or None if no nominal scenario is found.
    nThreads : int
        The number of threads to use for parallel evaluation, determined based on the number of CPU cores if not specified.
    
    Methods
    -------
    evaluate() -> pd.DataFrame
        Evaluate scenarios and compute clinical goal values and passing rates, returning a summary table as a DataFrame.
    """

    def __init__(self, PatientData:PatientData, sampler:AbstractsamplingMethod,kwargs):
        self.patientData = PatientData
        self.sampler = sampler
        self.scenarios = ScenariosGenerator(sampling_method=self.sampler).scenarios_list
        self.computeVWMin = kwargs.get('computeVWMin', False)
        self.computeVWMax = kwargs.get('computeVWMax', False)
        self.computeCumulativePassingRate = kwargs.get('computeCumulativePassingRate', False)

        self.prob_list = np.array([s.probability for s in self.scenarios])

        self.VWMin = None
        self.VWMax = None
        self.nominal_index = [i for i, s in enumerate(self.scenarios) if np.allclose(s.displacement, [0, 0, 0])][
            0] if any(np.allclose(s.displacement, [0, 0, 0]) for s in self.scenarios) else None

        if self.computeVWMin:
            self.VWMin = self.patientData.doseImage
        if self.computeVWMax:
            self.VWMax = np.zeros_like(self.patientData.doseImage)

        nThreads = kwargs.get('nThreads', -1)
        if nThreads < 1:
            self.nThreads = multiprocessing.cpu_count()-1
        else:
            self.nThreads = nThreads


    def evaluate(self)->pd.DataFrame:
        """
        Evaluate scenarios and compute clinical goal values and passing rates.
        
        Returns
        -------
        pd.DataFrame     
            A DataFrame containing clinical goals, nominal values, passing rates, and cumulative passing rates if computed
        """

        threads = []
        for i, scenario in enumerate(self.scenarios):
            while threading.active_count() > self.nThreads:
                time.sleep(0.1)
            print(f"Starting thread for scenario {i+1}/{len(self.scenarios)}")
            t = threading.Thread(target=self.compute_and_evaluate_scenario, args=(scenario,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        passingRates = self.computePassingRate()
        cumulativePassingRates = self.computeCumulativePassingRate() if self.computeCumulativePassingRate else None
        cumulativeRelativePassingRates = self.computeCumulativeRelativePassingRate() if self.computeCumulativePassingRate else None
        table = self.createPassingRateTable(passingRates, cumulativePassingRates, cumulativeRelativePassingRates)
        return table

    def compute_and_evaluate_scenario(self, scenario):
        """
            Compute the dose image for a scenario, evaluate clinical goals, and update voxel-wise min/max if enabled.
            
        Parameters
        ----------
        scenario : Scenario
            The scenario object containing displacement and probability information for evaluation.

        Returns
        -------
        None

        """
        scenario.compute_shifted_image(self.patientData.doseImage, scenario.displacement)
        dvh_dict = {}
        for mask in self.patientData.maskDict.keys():
            dvh_dict[mask] = DVH(dosemap=scenario.doseImage, mask=self.patientData.maskDict[mask], spacing=self.patientData.spacing)
        for goal in self.patientData.clinicalGoalsList:
            goal.compute(dvh_dict[goal.maskName])
        if self.computeVWMin:
            self.VWMin = np.minimum(self.VWMin, scenario.doseImage)
        if self.computeVWMax:
            self.VWMax = np.maximum(self.VWMax, scenario.doseImage)
        scenario.delete_doseImage()

    def computePassingRate(self)->list:
        """
        Compute passing rates for each clinical goal.
        
        Returns
        -------
        list
            A list of passing rates corresponding to each clinical goal.
        """
        passingRates = []
        for goal in self.patientData.clinicalGoalsList:
            success_arr = np.array(goal.successList)
            p = np.sum(self.prob_list[success_arr])
            passingRates.append(p)
        return passingRates


    def computeCumulativePassingRate(self)->list:
        """
        Compute cumulative passing rates for ordered clinical goals.
        
        Returns
        -------
        list
            A list of cumulative passing rates corresponding to each clinical goal in order.

        """

        cumulativeSuccessList = np.ones_like(self.scenarios, dtype=bool)
        cumulativePassingRateList = []
        for goal in self.patientData.clinicalGoalsList:
            success_arr = np.array(goal.successList)
            cumulativeSuccessList = np.logical_and(cumulativeSuccessList, success_arr)
            cumulativePassingRate = np.sum(self.prob_list[cumulativeSuccessList])
            cumulativePassingRateList.append(cumulativePassingRate)

        return cumulativePassingRateList

    def computeCumulativeRelativePassingRate(self)->list:
        """
        Compute relative cumulative passing rates for ordered clinical goals.
        
        Returns
        -------
        list
            A list of relative cumulative passing rates corresponding to each clinical goal in order.

        """

        cumulativeSuccessList = np.ones_like(self.scenarios, dtype=bool)
        cumulativeRelativePassingRateList = []
        for goal in self.patientData.clinicalGoalsList:
            success_arr = np.array(goal.successList)
            cumulativeSuccessList = np.logical_and(cumulativeSuccessList, success_arr)
            cumulativeRelativePassingRate = np.sum(self.prob_list[success_arr])/np.sum(self.prob_list[cumulativeSuccessList]) if np.sum(self.prob_list[cumulativeSuccessList]) > 0 else 0.0
            cumulativeRelativePassingRateList.append(cumulativeRelativePassingRate)

        return cumulativeRelativePassingRateList

    def createPassingRateTable(self, passingRates: list,cumulativePassingRates=None,cumulativeRelativePassingRates=None) -> pd.DataFrame:
        """
        Create a DataFrame table summarizing clinical goals, nominal values, passing rates, and cumulative passing rates if computed.
        
        Parameters
        ----------
        passingRates : list
            A list of passing rates corresponding to each clinical goal.
        cumulativePassingRates : list, optional
            A list of cumulative passing rates corresponding to each clinical goal in order, required if computeCumulativePassingRate is True.
        cumulativeRelativePassingRates : list, optional
            A list of relative cumulative passing rates corresponding to each clinical goal in order, required if computeCumulativePassingRate is True.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing clinical goals, nominal values, passing rates, and cumulative passing rates if computed.
        """

        headers = ["Mask Name", "Clinical Goal", "Nominal value", "Passing Rate"]
        if cumulativePassingRates is not None:
            headers.append("Cumulative Passing Rate")
        if cumulativeRelativePassingRates is not None:
            headers.append("Cumulative Relative Passing Rate")
        headers.append("Success Array")

        rows = []
        for i, goal in enumerate(self.patientData.clinicalGoalsList):
            row = {
                "Mask Name": goal.maskName,
                "Clinical Goal": str(goal),
                "Nominal value": "{:.3f}".format((goal.valueList[self.nominal_index]) if self.nominal_index is not None and len(goal.valueList) > self.nominal_index else "N/A"),
                "Passing Rate": passingRates[i]
            }

            if cumulativePassingRates is not None:
                row["Cumulative Passing Rate"] = cumulativePassingRates[i]

            if cumulativeRelativePassingRates is not None:
                row["Cumulative Relative Passing Rate"] = cumulativeRelativePassingRates[i]
            row["Success Array"] = np.array(goal.successList)
            rows.append(row)

        table = pd.DataFrame(rows, columns=headers)
        return table

    def createHTMLtable(self, table: pd.DataFrame):
        """
        Create an HTML representation of the passing rate table with color coding for better visualization.
            
        Parameters
        ----------
        table : pd.DataFrame
            A DataFrame containing clinical goals, nominal values, passing rates, and cumulative passing rates if computed, along with the success array for each clinical goal.

        Returns
        -------
        str
            An HTML string representing the styled passing rate table.

        """
        # create a new table without the success array column for better visualization

        table = table.drop(columns=["Success Array"])

        passing_cols = [
            c for c in [
                "Passing Rate",
                "Cumulative Passing Rate",
                "Cumulative Relative Passing Rate"
            ] if c in table.columns
        ]

        goals = self.patientData.clinicalGoalsList
        nominal_idx = table.columns.get_loc("Nominal value")

        def nominal_color(row):
            styles = [""] * len(row)
            goal = goals[row.name]

            success = goal.successList[self.nominal_index]

            if success:
                styles[nominal_idx] = "background-color:#8ef58e"
            else:
                styles[nominal_idx] = "background-color:#f58e8e"

            return styles

        styler = (
            table.style
            .apply(nominal_color, axis=1)
            .background_gradient(cmap="RdYlGn", subset=passing_cols, vmin=0, vmax=1)
            .format({col: "{:.1%}" for col in passing_cols})
            .set_table_styles([
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#2f6fed"),
                        ("color", "white"),
                        ("font-weight", "bold")
                    ]
                }
            ])
        )

        html = styler.to_html()
        return html

    def saveTableToHTML(self,table: pd.DataFrame, filepath: str = "passing_rate_table.html"):
        """
        Save the passing rate table as an HTML file with styling.
        
        Parameters
        ----------
        table : pd.DataFrame
            A DataFrame containing clinical goals, nominal values, passing rates, and cumulative passing rates if computed, along with the success array for each clinical goal.
        filepath : str, optional
            The file path where the HTML table will be saved. Default is "passing_rate_table.html".

        Returns
        -------
        None

        """
        html = self.createHTMLtable(table)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

    def saveTableToCSV(self, table: pd.DataFrame, filepath: str = "passing_rate_table.csv",save_success_array: bool = False):
        """
        Save the passing rate table as a CSV file.
        
        Parameters
        ----------
        table : pd.DataFrame
            A DataFrame containing clinical goals, nominal values, passing rates, and cumulative passing rates if computed, along with the success array for each clinical goal.
        filepath : str, optional
            The file path where the CSV table will be saved. Default is "passing_rate_table.csv".
        save_success_array : bool, optional
            Whether to include the success array in the CSV file. Default is False.
        
        Returns
        -------
        None
        """
        if not save_success_array:
            table = table.drop(columns=["Success Array"])
        table.to_csv(filepath, index=False)



    def passingRate_table(self):
        """
        Print a table with clinical goals, prescribed doses, and passing rates.
        """
        data = []
        for goal_name, goal_obj in self.patientData.clinicalGoalsList.items():
            data.append({
                'Clinical Goal': goal_name,
                'Dose threshold': goal_obj.prescription,
                'Passing Rate': goal_obj.passingRate
            })

        df = pd.DataFrame(data)
        print(df.to_string(index=False))

    def calculate_cummulPR(self):
        """
        Compute cumulative passing rates for ordered clinical goals.
        """
      
        cummul_PR = []
        i=1
        cummul_PR_prev = self.PR[0]
        cummul_PR.append(cummul_PR_prev)
        remaining_successList = self.patientData.clinicalGoalsList[0].successList
        while i < len(self.patientData.clinicalGoalsList):
            remaining_successList = [x*y for x,y in zip(self.patientData.clinicalGoalsList[i].successList, remaining_successList)]
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
        while i < len(self.patientData.clinicalGoalsList):
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
        self.computePassingRate()
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
        for i, goal in enumerate(self.patientData.clinicalGoalsList):
            print(goal)
            print(self.PR[i])
            data.append({
                'Clinical Goal': goal.__str__(),
                'Nominal Scenario': "{:.3f}".format(goal.valueList[zero_idx]),
                'Passing Rate': "{:.3f}".format(self.PR[i]),
                'Cumulative Passing Rate': "{:.3f}".format(self.CummulPR[i]),
                'Cumulative Passing Rate (Relative)': "{:.3f}".format(self.CummulPR_rel[i])
            })
        #for each objective; store the success list
        for i, goal in enumerate(self.patientData.clinicalGoalsList):
            for j, success in enumerate(goal.successList):
                data[i][f'Scenario {j+1} Success'] = success

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
            if self.computeVWMax:
                np.save(os.path.join(dir_path, f'VWMax{proba*100:.0f}.npy'), self.VWMax)
            if self.computeVWMin:
                np.save(os.path.join(dir_path, f'VWMin{proba*100:.0f}.npy'), self.VWMin)

        df.to_csv(out_path, index=False)


    def display_tables(self):
        """
        display in a interface the passing rates and cumulative passing rates in a table format, with clinical goals as rows and passing rates as columns.
        """
        self.computePassingRate()
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
        for i, goal in enumerate(self.patientData.clinicalGoalsList):
            row = {
                'Clinical Goal': goal.__str__(),
                'Passing Rate': "{:.3f}".format(self.PR[i])
            }
            if zero_idx is not None and len(goal.valueList) > zero_idx:
                row['Scenario [0,0,0]'] = "{:.3f}".format(goal.valueList[zero_idx])
            else:
                row['Scenario [0,0,0]'] = "N/A"
            if self.computeCumulativePassingRate:
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
                    success = self.patientData.clinicalGoalsList[row_idx - 1].successList[zero_idx]
                except Exception:
                    continue
                if success:
                    cell.set_text_props(color='green')
                else:
                    cell.set_text_props(color='red')

        plt.tight_layout()
        plt.show()

    def displayVminVmax(self,z_idx=None):
        """
        Display voxel-wise minimum and maximum dose images.

        Parameters
        ----------
        z_idx : int, optional
            The index of the z-slice to display. If None, the middle slice will be displayed. Default is None.

        Returns
        -------
        None
        """
        if z_idx is None:
            z_max = self.patientData.doseImage.shape[2]
            z_idx = z_max // 2

        import matplotlib.pyplot as plt

        if self.computeVWMin and self.computeVWMax:
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