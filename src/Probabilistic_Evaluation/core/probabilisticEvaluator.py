from Probabilistic_Evaluation.core.sampling._abstractSamplingMethod import AbstractsamplingMethod
from Probabilistic_Evaluation.core.scenariosGenerator import ScenariosGenerator
from Probabilistic_Evaluation.data import PatientData
from Probabilistic_Evaluation.data import DVH
from Probabilistic_Evaluation.utils import timed, Timer
import os

import threading
import multiprocessing
import time
import numpy as np
import pandas as pd
import scipy as sp

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
    computeCumulativePassingRates : bool, optional
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

    def __init__(self, PatientData:PatientData, sampler:AbstractsamplingMethod,**kwargs):
        self.patientData = PatientData
        self.sampler = sampler
        self.scenarios = ScenariosGenerator(sampling_method=self.sampler).scenarios_list
        self.computeVWMin = kwargs.get('computeVWMin', False)
        self.computeVWMax = kwargs.get('computeVWMax', False)
        self.computeCumulativePassingRates = kwargs.get('computeCumulativePassingRate', False)

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

    @timed
    def evaluate(self)->pd.DataFrame:
        """
        Evaluate scenarios and compute clinical goal values and passing rates.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing clinical goals, nominal values, passing rates, and cumulative passing rates if computed
        """
        if self.sampler.UncertaintyModel.rand_parameters is not None:
            dose_image = self.blur_dose()
            self.patientData.doseImage = dose_image
            print("Updated patient dose image with blurred dose")
        n_scenarios = len(self.scenarios)
        for goal in self.patientData.clinicalGoalsList:
            goal.valueList = np.empty(n_scenarios)
            goal.successList = np.empty(n_scenarios, dtype=bool)

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

        passingRates = self.passingRates()
        cumulativePassingRates = self.cumulativePassingRates() if self.computeCumulativePassingRates else None
        cumulativeRelativePassingRates = self.cumulativeRelativePassingRates() if self.computeCumulativePassingRates else None
        table = self.createPassingRateTable(passingRates, cumulativePassingRates, cumulativeRelativePassingRates)
        return table

    def compute_and_evaluate_scenario(self,scenario_idx, scenario):
        """
            Compute the dose image for a scenario, evaluate clinical goals, and update voxel-wise min/max if enabled.

        Parameters
        ----------
        scenario_idx : int
            The index of the scenario being evaluated.
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
            goal.compute(dvh_dict[goal.maskName], scenario_idx=scenario_idx)
        if self.computeVWMin:
            self.VWMin = np.minimum(self.VWMin, scenario.doseImage)
        if self.computeVWMax:
            self.VWMax = np.maximum(self.VWMax, scenario.doseImage)
        scenario.delete_doseImage()

    def passingRates(self)->list:
        """
        Compute passing rates for each clinical goal.

        Returns
        -------
        list
            A list of passing rates corresponding to each clinical goal.
        """
        passingRates = []
        for goal in self.patientData.clinicalGoalsList:
            p = np.sum(self.prob_list[goal.successList])
            passingRates.append(p)
        return passingRates


    def cumulativePassingRates(self)->list:
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
            cumulativeSuccessList = np.logical_and(cumulativeSuccessList, goal.successList)
            cumulativePassingRate = np.sum(self.prob_list[cumulativeSuccessList])
            cumulativePassingRateList.append(cumulativePassingRate)

        return cumulativePassingRateList

    def cumulativeRelativePassingRates(self)->list:
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
            cumulativeSuccessList = np.logical_and(cumulativeSuccessList, goal.successList)
            cumulativeRelativePassingRate = np.sum(self.prob_list[goal.successList])/np.sum(self.prob_list[cumulativeSuccessList]) if np.sum(self.prob_list[cumulativeSuccessList]) > 0 else 0.0
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
        headers.append("Probability Array")

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
            row["Success Array"] = goal.successList
            row["Probability Array"] = np.array(self.prob_list)

            rows.append(row)

        table = pd.DataFrame(rows, columns=headers)
        return table

    def recomputePassingRateTable(self, df: pd.DataFrame):
        """
        Recompute cumulative passing rates in the table based on the success arrays and probabilities, allowing for dynamic updates if clinical goals are modified.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing clinical goals, nominal values, passing rates, and cumulative passing rates if computed, along with the success array for each clinical goal.

        Returns
        -------
        pd.DataFrame
            An updated DataFrame with recomputed cumulative passing rates based on the success arrays and probabilities.

        """

        cumulative_success = np.ones(len(self.prob_list), dtype=bool)

        cum_rates = []
        cum_rel_rates = []

        for _, row in df.iterrows():
            success = row["Success Array"]

            cumulative_success = np.logical_and(cumulative_success, success)

            cum = np.sum(self.prob_list[cumulative_success])
            cum_rates.append(cum)

            rel = np.sum(self.prob_list[success]) / cum if cum > 0 else 0
            cum_rel_rates.append(rel)

        df["Cumulative Passing Rate"] = cum_rates
        df["Cumulative Relative Passing Rate"] = cum_rel_rates

        return df
    def blur_dose(self):
        """
        Apply Gaussian blurring to the dose map based on the random setup error parameters.

        Parameters
        ----------
        dosemap : np.ndarray
            A 3D array representing the original dose map.
        rand_parameters : dict
            A dictionary containing the standard deviations for blurring in x, y, and z directions (in mm):
            - 'sigma_x': Standard deviation in x direction
            - 'sigma_y': Standard deviation in y direction
            - 'sigma_z': Standard deviation in z direction

        Returns
        -------
        np.ndarray
            A 3D array representing the blurred dose map.
        """
        dosemap = self.patientData.doseImage
        rand_parameters = self.sampler.UncertaintyModel.rand_parameters
        
        sigma_x = rand_parameters['sigma_x']
        sigma_y = rand_parameters['sigma_y']
        sigma_z = rand_parameters['sigma_z']
        blurred_dosemap = sp.ndimage.gaussian_filter(dosemap, sigma=[sigma_x, sigma_y, sigma_z] / self.patientData.spacing)
        return blurred_dosemap
    
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

        table = table.drop(columns=["Success Array", "Probability Array"])

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

    def saveTableToCSV(self, table: pd.DataFrame, filepath: str = "passing_rate_table.csv",save_success_array: bool = True):
        """
        Save the passing rate table as a CSV file.

        Parameters
        ----------
        table : pd.DataFrame
            A DataFrame containing clinical goals, nominal values, passing rates, and cumulative passing rates if computed, along with the success array for each clinical goal.
        filepath : str, optional
            The file path where the CSV table will be saved. Default is "passing_rate_table.csv".
        save_success_array : bool, optional
            Whether to include the success array in the CSV file. Default is True.

        Returns
        -------
        None
        """
        if not save_success_array:
            table = table.drop(columns=["Success Array", "Probability Array"])
        table.to_csv(filepath, index=False)

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