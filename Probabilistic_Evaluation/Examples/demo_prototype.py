from Probabilistic_Evaluation.io import *
from core.scenariosGenerator import ScenariosGenerator
from data._patientData import PatientData
from core.probabilisticEvaluator import ProbabilisticEvaluator


#Read images from DICOM
dicom_path = r"C:\Users\geert\OneDrive - UCL\PhD\Probabilistic eval project St.Luc\ORL_001\ORL_001\test"
clinicalgoalpath = r"C:\Users\geert\OneDrive - UCL\PhD\Probabilistic eval project St.Luc\ORL_001\ORL_001\test\clinicalGoals.json"
reader = DicomReader()
reader.load_dicom_series(dicom_path)
clinicalgoals_reader = clinicalgoalsreader(maskDict=reader.RTSTRUCT)
clinicalgoals_reader.load_JSON_list(clinicalgoalpath)

#Create patient data object
patient = PatientData(ctImage=reader.CT, doseImage=reader.RTDOSE, maskDict=reader.RTSTRUCT, spacing=reader.spacing)
patient.clinicalGoalsList(clinicalgoals_reader.clinical_goals_list)
#Generate probabilistic scenarios
prob_scenarios_generator = ScenariosGenerator(patientData=patient, max_displacement=10)
#initiate scenario calculation
prob_scenarios_generator.compute_scenarios()
#create probabilistic evaluator object
prob_evaluator = ProbabilisticEvaluator(patientData=patient)
prob_evaluator.evaluate()
