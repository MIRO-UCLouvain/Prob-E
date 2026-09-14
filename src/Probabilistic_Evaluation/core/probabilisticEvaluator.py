from Probabilistic_Evaluation.core.sampling._abstractSamplingMethod import AbstractsamplingMethod
from Probabilistic_Evaluation.core.scenariosGenerator import ScenariosGenerator
from Probabilistic_Evaluation.data import PatientData
from Probabilistic_Evaluation.data import DVH
from Probabilistic_Evaluation.utils import timed, Timer, shift_dose_for_enhanced_sampling
from Probabilistic_Evaluation.logging_utils import logger, log_call, GroupedMemoryHandler

import os
import logging
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
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
    blurred_dose : np.ndarray or None
        The blurred dose image used for evaluation when simulating fractionation, or None if not computed.
    nThreads : int
        The number of threads to use for parallel evaluation, determined based on the number of CPU cores if not specified.
    """

    def __init__(self, PatientData:PatientData, sampler:AbstractsamplingMethod,**kwargs):
        self.patientData = PatientData
        self.sampler = sampler
        self.scenarios = ScenariosGenerator(sampling_method=self.sampler).scenarios_list
        self.computeVWMin = kwargs.get('computeVWMin', False)
        self.computeVWMax = kwargs.get('computeVWMax', False)
        self.computeCumulativePassingRates = kwargs.get('computeCumulativePassingRate', True)

        self.prob_list = np.array([s.probability for s in self.scenarios])

        self.VWMin = None
        self.VWMax = None
        self._vw_lock = threading.Lock()  # VWMin/VWMax are updated from several scenario threads

        self.blurred_dose = None
        self.half_shifted_dose = None  # Store the shifted+blurred dose for evaluation when using enhanced sampling

        self.nominal_index = [i for i, s in enumerate(self.scenarios) if np.allclose(s.displacement, [0, 0, 0])][
            0] if any(np.allclose(s.displacement, [0, 0, 0]) for s in self.scenarios) else None

        if self.computeVWMin:
            self.VWMin = self.patientData.doseImage
        if self.computeVWMax:
            self.VWMax = np.zeros_like(self.patientData.doseImage)

        nThreads = kwargs.get('nThreads', -1)
        if nThreads < 1:
            self.nThreads = max(1, multiprocessing.cpu_count()-1)  # the thread pool needs at least one worker
        else:
            self.nThreads = nThreads
        logger.info(f"ProbabilisticEvaluator initialized with {len(self.scenarios)} scenarios, using {self.nThreads} threads.")
        logger.debug(f"Patient ID: {self.patientData.patientID}, Number of clinical goals: {len(self.patientData.clinicalGoalsList)}, Sampler: {type(self.sampler).__name__}, Compute VWMin: {self.computeVWMin}, Compute VWMax: {self.computeVWMax}, Compute Cumulative Passing Rates: {self.computeCumulativePassingRates}")
        logger.debug(f"Nominal scenario index: {self.nominal_index}, Blurred dose computed: {self.blurred_dose is not None}, Half-shifted dose computed: {self.half_shifted_dose is not None}")
        logger.debug(f"Max value blurred dose: {np.max(self.blurred_dose) if self.blurred_dose is not None else 'N/A'}, Max value half-shifted dose: {np.max(self.half_shifted_dose) if self.half_shifted_dose is not None else 'N/A'}")
        logger.debug(f"Min value blurred dose: {np.min(self.blurred_dose) if self.blurred_dose is not None else 'N/A'}, Min value half-shifted dose: {np.min(self.half_shifted_dose) if self.half_shifted_dose is not None else 'N/A'}")
        logger.debug(f"Nominal index: {self.nominal_index}")

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate scenarios and compute clinical goal values and passing rates.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing clinical goals, nominal values, passing rates, and cumulative passing rates if computed
        """
        if self.sampler.UncertaintyModel.rand_parameters is not None:
            self.blurred_dose = self.blur_dose(self.patientData.doseImage)
            logger.debug(f"Blurred dose computed for patient {self.patientData.patientID}")
            if self.sampler.enhanced:
                self.half_shifted_dose = shift_dose_for_enhanced_sampling(self.blurred_dose)
                logger.debug(f"Half-shifted blurred dose computed for enhanced sampling for patient {self.patientData.patientID}")

        n_scenarios = len(self.scenarios)
        for goal in self.patientData.clinicalGoalsList:
            # NaN / False until a scenario has actually been evaluated (never uninitialised memory)
            goal.valueList = np.full(n_scenarios, np.nan)
            goal.successList = np.zeros(n_scenarios, dtype=bool)

        file_handler = next((h for h in logger.handlers if isinstance(h, logging.FileHandler)), None)

        if file_handler is None:
            # no file handler attached -> nothing to group, just run
            self._execute_scenario_threads()
        else:
            grouped_handler = GroupedMemoryHandler(target_handler=file_handler)
            logger.removeHandler(file_handler)
            logger.addHandler(grouped_handler)

            try:
                self._execute_scenario_threads()
            finally:
                grouped_handler.flush_grouped()
                logger.removeHandler(grouped_handler)
                logger.addHandler(file_handler)

        passingRates = self.passingRates()
        cumulativePassingRates = self.cumulativePassingRates() if self.computeCumulativePassingRates else None
        cumulativeRelativePassingRates = self.cumulativeRelativePassingRates() if self.computeCumulativePassingRates else None
        table = self.createPassingRateTable(passingRates, cumulativePassingRates, cumulativeRelativePassingRates)
        return table

    def _execute_scenario_threads(self):
        """
        Evaluate all scenarios in a thread pool (shared by the grouped and plain logging paths).

        Every worker exception is logged and, once all scenarios have finished, re-raised as a
        RuntimeError naming the failed scenarios, so that a failed scenario can never end up in the
        passing rates.
        """
        logger.debug("Entering multithreading evaluation of scenarios.")
        failures = []
        with ThreadPoolExecutor(max_workers=self.nThreads, thread_name_prefix="Scenario") as executor:
            futures = {
                executor.submit(self.compute_and_evaluate_scenario, i, scenario): i
                for i, scenario in enumerate(self.scenarios)
            }
            for future, i in futures.items():
                try:
                    future.result()
                except Exception as exc:
                    logger.error(
                        f"Scenario {i+1}/{len(self.scenarios)} (displacement {self.scenarios[i].displacement}) failed",
                        exc_info=exc,
                    )
                    failures.append((i, exc))
        if failures:
            failed_ids = ", ".join(str(i + 1) for i, _ in failures)
            raise RuntimeError(
                f"{len(failures)} of {len(self.scenarios)} scenarios failed during evaluation "
                f"(scenario {failed_ids}); see the log for the tracebacks."
            ) from failures[0][1]
        logger.debug("All threads completed.")


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
        # name the pool thread after the scenario so that the grouped log stays readable per scenario
        threading.current_thread().name = f"Scenario-{scenario_idx+1}"
        logger.info(f"Starting evaluation of scenario {scenario_idx+1}/{len(self.scenarios)}")
        if self.blurred_dose is not None:
            if np.allclose(scenario.displacement % 1, 0, atol=1e-6):
                scenario.compute_shifted_image(self.blurred_dose, scenario.displacement)
                logger.debug(f"{scenario.displacement}: Using blurred dose for evaluation")
            elif self.half_shifted_dose is not None:
                shift = np.array(scenario.displacement) - 0.5
                scenario.compute_shifted_image(self.half_shifted_dose, shift)
                logger.debug(f"{scenario.displacement}: Using half-shifted blurred dose for evaluation, with corrected shift: {shift}")
        
        # Fractionation is not considered, we use the original dose image for evaluation
        else:
            scenario.compute_shifted_image(self.patientData.doseImage, scenario.displacement)

        dvh_dict = {}
        for mask in self.patientData.maskDict.keys():
            logger.debug(f"Evaluating scenario {scenario_idx+1}/{len(self.scenarios)} for mask {mask} with displacement {scenario.displacement} and probability {scenario.probability}.")
            dvh_dict[mask] = DVH(dosemap=scenario.doseImage, mask=self.patientData.maskDict[mask], spacing=self.patientData.spacing)
        for goal in self.patientData.probabilisticGoalsList:
            logger.debug(f"computing goal {goal} for mask {goal.maskName} for scenario {scenario_idx+1}/{len(self.scenarios)} with displacement {scenario.displacement} and probability {scenario.probability}.")
            goal.compute(dvh_dict[goal.maskName], scenario_idx=scenario_idx)
        if self.computeVWMin or self.computeVWMax:
            # read-modify-write on shared arrays: serialise across scenario threads
            with self._vw_lock:
                if self.computeVWMin:
                    self.VWMin = np.minimum(self.VWMin, scenario.doseImage)
                if self.computeVWMax:
                    self.VWMax = np.maximum(self.VWMax, scenario.doseImage)
        scenario.delete_doseImage()

    @log_call(log_result=True)
    def computeNominalValues(self):
        """
        Compute nominal values for each clinical goal based on the scenario with displacement closest to [0, 0, 0].
        If fractionation is simulated, the nominal values are computed on a non-blurred dose since it represents
        a 0 setup error and 0 random error at each fraction.

        Returns
        -------
        value_list : list
            A list of nominal values corresponding to each clinical goal, or None if no nominal scenario is found.
        success_list : list
            A list of boolean values indicating whether each clinical goal is achieved in the nominal scenario, or
            None if no nominal scenario is found.
        """
        value_list = []
        success_list = []
        dvh_dict = {}

        nominal_dose = self.patientData.doseImage
        logger.info(f"Computing nominal values for clincal goals, without dose blurring, using nominal index: {self.nominal_index}")
        # We need to recompute the goals for a non_blured_dose
        if self.blurred_dose is not None:

            for maskname in self.patientData.maskDict.keys():

                dvh_dict[maskname] = DVH(dosemap=nominal_dose, mask=self.patientData.maskDict[maskname], spacing=self.patientData.spacing)
            for goal in self.patientData.clinicalGoalsList:
                value, success = goal.compute(dvh_dict[goal.maskName])
                value_list.append(value)
                success_list.append(success)

        # We can use the nominal index to directly get the values from the goal's valueList
        else:
            for maskname in self.patientData.maskDict.keys():

                dvh_dict[maskname] = DVH(dosemap=nominal_dose, mask=self.patientData.maskDict[maskname], spacing=self.patientData.spacing)
            for goal in self.patientData.clinicalGoalsList:
                if goal.probabilistic:
                    value = goal.valueList[self.nominal_index] if self.nominal_index is not None and len(goal.valueList) > self.nominal_index else None
                    value_list.append(value)
                    success = goal.successList[self.nominal_index] if self.nominal_index is not None else None
                    success_list.append(success)
                else:
                    value, success = goal.compute(dvh_dict[goal.maskName])
                    value_list.append(value)
                    success_list.append(success)

        return value_list, success_list

    def passingRates(self)->list:
        """
        Compute passing rates for each clinical goal.

        Returns
        -------
        list
            A list of passing rates corresponding to each clinical goal.
        """
        logger.debug(f"Computing passing rates for {len(self.patientData.clinicalGoalsList)} clinical goals.")
        passingRates = []
        for goal in self.patientData.probabilisticGoalsList:
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

        logger.debug(f"Computing cumulative passing rates for {len(self.patientData.clinicalGoalsList)} clinical goals.")   
        cumulativeSuccessList = np.ones_like(self.scenarios, dtype=bool)
        cumulativePassingRateList = []
        for goal in self.patientData.probabilisticGoalsList:
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
        logger.debug(f"Computing cumulative relative passing rates for {len(self.patientData.clinicalGoalsList)} clinical goals.")
        cumulativeSuccessList = np.ones_like(self.scenarios, dtype=bool)
        cumulativeRelativePassingRateList = []
        for goal in self.patientData.probabilisticGoalsList:
            cumulativeSuccessList = np.logical_and(cumulativeSuccessList, goal.successList)
            cumulativeRelativePassingRate = np.sum(self.prob_list[goal.successList])/np.sum(self.prob_list[cumulativeSuccessList]) if np.sum(self.prob_list[cumulativeSuccessList]) > 0 else 0.0
            cumulativeRelativePassingRateList.append(cumulativeRelativePassingRate)

        return cumulativeRelativePassingRateList

    @log_call(log_result=True)
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
            A DataFrame containing clinical goals, nominal values, nominal success, passing rates, and cumulative passing rates if computed.
        """

        headers = ["Mask Name", "Clinical Goal", "Nominal Value", "Nominal Success", "Passing Rate","Probabilistic Objective"]
        if cumulativePassingRates is not None:
            headers.append("Cumulative Passing Rate")
        if cumulativeRelativePassingRates is not None:
            headers.append("Cumulative Relative Passing Rate")
        headers.append("Success Array")

        nominal_value,nominal_success = self.computeNominalValues()

        # The rates were computed by iterating over probabilisticGoalsList; look them up by goal identity
        # instead of by position in clinicalGoalsList, which also holds the non-probabilistic goals.
        rate_index = {id(goal): k for k, goal in enumerate(self.patientData.probabilisticGoalsList)}

        rows = []
        for i, goal in enumerate(self.patientData.clinicalGoalsList):
            if goal.probabilistic:
                k = rate_index.get(id(goal))
                if k is None:
                    raise ValueError(
                        f"Clinical goal {goal} ({goal.maskName}) is flagged probabilistic but is not part of "
                        "patientData.probabilisticGoalsList; assign clinicalGoalsList after setting the flag."
                    )
                row = {
                    "Mask Name": goal.maskName,
                    "Clinical Goal": str(goal),
                    "Nominal Value": float(nominal_value[i]) if nominal_value[i] is not None else "N/A",
                    "Nominal Success": nominal_success[i] if nominal_success[i] is not None else "N/A",
                    "Passing Rate": passingRates[k],
                    "Probabilistic Objective": goal.probabilistic
                }

                if cumulativePassingRates is not None:
                    row["Cumulative Passing Rate"] = cumulativePassingRates[k]

                if cumulativeRelativePassingRates is not None:
                    row["Cumulative Relative Passing Rate"] = cumulativeRelativePassingRates[k]
                row["Success Array"] = goal.successList
            else:
                row = {
                    "Mask Name": goal.maskName,
                    "Clinical Goal": str(goal),
                    "Nominal Value": float(nominal_value[i]) if nominal_value[i] is not None else "N/A",
                    "Nominal Success": nominal_success[i] if nominal_success[i] is not None else "N/A",
                    "Passing Rate": "N/A",
                    "Probabilistic Objective": goal.probabilistic,
                    "Success Array": "N/A"
                }

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
    
    @log_call(log_result=True)
    def blur_dose(self, dosemap: np.ndarray) -> np.ndarray:
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
      
        rand_parameters = self.sampler.UncertaintyModel.rand_parameters
        
        sigma_x = rand_parameters['sigma_x']
        sigma_y = rand_parameters['sigma_y']
        sigma_z = rand_parameters['sigma_z']
        print(f"Blurring dose map with sigma_x={sigma_x}, sigma_y={sigma_y}, sigma_z={sigma_z}")
        print(f"Voxel spacing: {self.patientData.spacing}")
        blurred_dosemap = sp.ndimage.gaussian_filter(dosemap, sigma=[sigma_x, sigma_y, sigma_z] / self.patientData.spacing)
        return blurred_dosemap
    
    
    @log_call(log_result=True)
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

        nominal_idx = table.columns.get_loc("Nominal Value")

        def nominal_color(row):
            styles = [""] * len(row)

            success = row["Nominal Success"] if "Nominal Success" in row else None

            if success:
                styles[nominal_idx] = "background-color:#8ef58e"
            else:
                styles[nominal_idx] = "background-color:#f58e8e"

            return styles

        formats = {col: "{:.1%}" for col in passing_cols}
        # nominal values are numeric in the table (strings such as "N/A" are passed through)
        formats["Nominal Value"] = lambda v: f"{v:.4f}" if isinstance(v, (float, int, np.floating, np.integer)) else v

        styler = (
            table.style
            .apply(nominal_color, axis=1)
            .background_gradient(cmap="RdYlGn", subset=passing_cols, vmin=0, vmax=1)
            .format(formats)
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

    @log_call(log_result=True)
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

    @log_call(log_result=True)
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

    @log_call(log_result=True)
    def saveTableToJSON(self, table: pd.DataFrame, filepath: str = "passing_rate_table.json"):
        """
        Save passing rate results as a compact but human-readable JSON file.

        Parameters
        ----------
        table : pd.DataFrame
            A DataFrame containing clinical goals, nominal values, passing rates,
            cumulative passing rates if computed, and per-goal arrays.
        filepath : str, optional
            The file path where the JSON table will be saved. Default is "passing_rate_table.json".

        Returns
        -------
        None
        """
        import json

        def _json_safe(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, pd.Series):
                return [_json_safe(v) for v in value.tolist()]
            if isinstance(value, list):
                return [_json_safe(v) for v in value]
            if isinstance(value, tuple):
                return [_json_safe(v) for v in value]
            if isinstance(value, dict):
                return {k: _json_safe(v) for k, v in value.items()}
            return value

        def _is_na(value):
            if value is None:
                return True
            if isinstance(value, str) and value.strip().upper() == "N/A":
                return True
            try:
                return bool(pd.isna(value))
            except Exception:
                return False

        exported_table = table.copy()

        # Probability array is shared across goals; keep once at top-level only.
        if "Probability Array" in exported_table.columns:
            exported_table = exported_table.drop(columns=["Probability Array"])

        table_records = []
        for row in exported_table.to_dict(orient="records"):
            cleaned_row = _json_safe(row)
            is_probabilistic = bool(cleaned_row.get("Probabilistic Objective", False))

            # For non-probabilistic objectives, remove probabilistic metrics.
            if not is_probabilistic:
                for key in [
                    "Success Array",
                    "Passing Rate",
                    "Cumulative Passing Rate",
                    "Cumulative Relative Passing Rate",
                ]:
                    cleaned_row.pop(key, None)

            # Drop remaining empty placeholders to keep the JSON compact.
            cleaned_row = {k: v for k, v in cleaned_row.items() if not _is_na(v)}
            table_records.append(cleaned_row)

        payload = {
            "Probability Mass": float(np.sum(self.prob_list)),
            "Patient ID": self.patientData.patientID,
            "Probability Array": _json_safe(np.array(self.prob_list)),
            "Table": table_records,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

    @log_call(log_result=True)
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
    
    @log_call(log_result=True)
    def displayblurandnominal(self, nominal_dose, blurred_dose, z_idx=None):
        """
        Display the nominal and blurred dose images with CT background,
        requested contours, four synchronized dose profiles, and a z-axis slider.

        Parameters
        ----------
        nominal_dose : np.ndarray
            A 3D array representing the nominal dose map.
        blurred_dose : np.ndarray
            A 3D array representing the blurred dose map.
        z_idx : int, optional
            The index of the z-slice to display. If None, the middle slice is displayed.

        Returns
        -------
        None
        """
        if blurred_dose is None:
            raise ValueError("blurred_dose cannot be None")

        if nominal_dose.shape != blurred_dose.shape:
            raise ValueError("nominal_dose and blurred_dose must have the same shape")

        z_max = nominal_dose.shape[2]
        if z_idx is None:
            z_idx = z_max // 2
        z_idx = int(np.clip(z_idx, 0, z_max - 1))

        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        from matplotlib.widgets import Slider
        from matplotlib.lines import Line2D

        dose_vmin = float(min(np.min(nominal_dose), np.min(blurred_dose)))
        dose_vmax = float(max(np.max(nominal_dose), np.max(blurred_dose)))
        dose_norm = colors.PowerNorm(gamma=4.0, vmin=dose_vmin, vmax=dose_vmax)
        profile_ymin = float(np.min([np.min(nominal_dose), np.min(blurred_dose)]))
        profile_ymax = float(np.max([np.max(nominal_dose), np.max(blurred_dose)]))

        display_xmin = 150
        display_xmax = 400
        display_ymin = 150
        display_ymax = 400

        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        plt.subplots_adjust(bottom=0.12, hspace=0.4, wspace=0.25)

        image_axes = [axes[0, 0], axes[0, 1]]
        profile_axes = [axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]

        contour_names = ["PTV", "PTV_3mm", "GTV", "Macula"]
        contour_colors = {
            "PTV": "orange",
            "PTV_3mm": "magenta",
            "GTV": "cyan",
            "Macula": "lime",
        }

        contour_aliases = {
            "PTV": ["PTV"],
            "PTV_3mm": ["PTV_3mm"],
            "GTV": ["GTV"],
            "Macula": ["Macula", "Macula_R"],
        }

        available_contours = []
        for display_name in contour_names:
            aliases = [alias.lower() for alias in contour_aliases.get(display_name, [display_name])]
            matching_keys = [
                key for key in self.patientData.maskDict.keys()
                if key.lower() in aliases
            ]

            if display_name == "Macula" and not matching_keys:
                matching_keys = [
                    key for key in self.patientData.maskDict.keys()
                    if "macula" in key.lower()
                ]

            for mask_name in matching_keys:
                available_contours.append((display_name, mask_name))

        legend_handles = [
            Line2D([0], [0], color=contour_colors[display_name], lw=2, label=display_name)
            for display_name, _ in available_contours
        ]

        colorbar_nominal = None
        colorbar_blurred = None

        x_profile_fixed_y = [
            int(np.clip(228, 0, nominal_dose.shape[1] - 1)),
            int(np.clip(236, 0, nominal_dose.shape[1] - 1)),
        ]
        y_profile_fixed_x = [
            int(np.clip(223, 0, nominal_dose.shape[0] - 1)),
            int(np.clip(236, 0, nominal_dose.shape[0] - 1)),
        ]

        def redraw_slice(slice_idx):
            nonlocal colorbar_nominal, colorbar_blurred

            if colorbar_nominal is not None:
                colorbar_nominal.remove()
                colorbar_nominal = None
            if colorbar_blurred is not None:
                colorbar_blurred.remove()
                colorbar_blurred = None

            for axis in axes.flat:
                axis.clear()

            ct_slice = self.patientData.ctImage[:, :, slice_idx]

            nominal_image = image_axes[0].imshow(ct_slice, cmap='gray', zorder=0)
            nominal_dose_image = image_axes[0].imshow(nominal_dose[:, :, slice_idx], cmap='jet', norm=dose_norm, alpha=0.5, zorder=1)
            image_axes[0].set_title('Nominal Dose Distribution (Slice {0})'.format(slice_idx))
            image_axes[0].set_xlabel('x')
            image_axes[0].set_ylabel('y')
            image_axes[0].set_xlim(display_xmin, display_xmax)
            image_axes[0].set_ylim(display_ymax, display_ymin)
            for fixed_y in x_profile_fixed_y:
                image_axes[0].axhline(fixed_y, color='white', linestyle='--', linewidth=1.0, alpha=0.9)
            for fixed_x in y_profile_fixed_x:
                image_axes[0].axvline(fixed_x, color='white', linestyle=':', linewidth=1.0, alpha=0.9)

            blurred_image = image_axes[1].imshow(ct_slice, cmap='gray', zorder=0)
            blurred_dose_image = image_axes[1].imshow(blurred_dose[:, :, slice_idx], cmap='jet', norm=dose_norm, alpha=0.5, zorder=1)
            image_axes[1].set_title('Blurred Dose Distribution (Slice {0})'.format(slice_idx))
            image_axes[1].set_xlabel('x')
            image_axes[1].set_ylabel('y')
            image_axes[1].set_xlim(display_xmin, display_xmax)
            image_axes[1].set_ylim(display_ymax, display_ymin)
            for fixed_y in x_profile_fixed_y:
                image_axes[1].axhline(fixed_y, color='white', linestyle='--', linewidth=1.0, alpha=0.9)
            for fixed_x in y_profile_fixed_x:
                image_axes[1].axvline(fixed_x, color='white', linestyle=':', linewidth=1.0, alpha=0.9)

            for axis in image_axes:
                for display_name, mask_name in available_contours:
                    axis.contour(
                        self.patientData.maskDict[mask_name][:, :, slice_idx],
                        levels=[0.5],
                        colors=contour_colors[display_name],
                        linewidths=1.0,
                        zorder=2,
                    )

                if available_contours:
                    axis.legend(handles=legend_handles, loc='upper right')

            colorbar_nominal = fig.colorbar(nominal_dose_image, ax=image_axes[0], label='Dose')
            colorbar_blurred = fig.colorbar(blurred_dose_image, ax=image_axes[1], label='Dose')

            profile_specs = [
                (profile_axes[0], "x", x_profile_fixed_y[0], f"Dose profile: y={x_profile_fixed_y[0]}, z={slice_idx}"),
                (profile_axes[1], "x", x_profile_fixed_y[1], f"Dose profile: y={x_profile_fixed_y[1]}, z={slice_idx}"),
                (profile_axes[2], "y", y_profile_fixed_x[0], f"Dose profile: x={y_profile_fixed_x[0]}, z={slice_idx}"),
                (profile_axes[3], "y", y_profile_fixed_x[1], f"Dose profile: x={y_profile_fixed_x[1]}, z={slice_idx}"),
            ]

            for axis, direction, fixed_index, title in profile_specs:
                if direction == "x":
                    x_values = np.arange(nominal_dose.shape[0])
                    nominal_curve = nominal_dose[:, fixed_index, slice_idx]
                    blurred_curve = blurred_dose[:, fixed_index, slice_idx]
                    axis.set_xlabel("x")
                else:
                    x_values = np.arange(nominal_dose.shape[1])
                    nominal_curve = nominal_dose[fixed_index, :, slice_idx]
                    blurred_curve = blurred_dose[fixed_index, :, slice_idx]
                    axis.set_xlabel("y")

                axis.plot(x_values, nominal_curve, color="tab:blue", label="Nominal")
                axis.plot(x_values, blurred_curve, color="tab:red", linestyle="--", label="Blurred")
                if direction == "x":
                    for fixed_ref in x_profile_fixed_y:
                        axis.axvline(fixed_ref, color='0.5', linestyle=':', linewidth=0.9, alpha=0.7)
                else:
                    for fixed_ref in y_profile_fixed_x:
                        axis.axvline(fixed_ref, color='0.5', linestyle=':', linewidth=0.9, alpha=0.7)
                axis.set_title(title)
                axis.set_ylabel("Dose")
                axis.grid(True, alpha=0.2)
                axis.legend(loc='best')
                axis.set_xlim(display_xmin, display_xmax)
                axis.set_ylim(profile_ymin, profile_ymax)

        redraw_slice(z_idx)

        slider_ax = fig.add_axes((0.15, 0.07, 0.7, 0.04))
        z_slider = Slider(
            ax=slider_ax,
            label='Z Slice',
            valmin=0,
            valmax=z_max - 1,
            valinit=z_idx,
            valstep=1,
        )

        def update(_):
            current_z = int(z_slider.val)
            redraw_slice(current_z)
            fig.canvas.draw_idle()

        z_slider.on_changed(update)
        plt.show()


    