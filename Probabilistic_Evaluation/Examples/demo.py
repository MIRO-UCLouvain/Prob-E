from Probabilistic_Evaluation.core.scenariosGenerator import ScenariosGenerator
from Probabilistic_Evaluation.data import PatientData
from Probabilistic_Evaluation.core.probabilisticEvaluator import ProbabilisticEvaluator
from Probabilistic_Evaluation.io.dicomIO import DicomReader
from Probabilistic_Evaluation.io.clinicalgoalsIO import clinicalgoalsreader


#Read images from DICOM
dicom_path = r"C:\Users\geert\OneDrive - UCL\PhD\Probabilistic eval project St.Luc\ORL_001\ORL_001\test"
clinicalgoalpath = r"C:\Users\geert\OneDrive - UCL\PhD\Probabilistic eval project St.Luc\ORL_001\ORL_001\test\clinicalGoals.json"
out_path = r"C:\Users\geert\OneDrive - UCL\PhD\Probabilistic eval project St.Luc\ORL_001\ORL_001\test\results.csv"
reader = DicomReader()
reader.load_dicom_series(dicom_path)
clinicalgoals_reader = clinicalgoalsreader(maskDict=reader.RTSTRUCT)
clinicalgoals_reader.load_JSON_list(clinicalgoalpath)

#Create patient data object
patient = PatientData(ctImage=reader.CT, doseImage=reader.RTDOSE, maskDict=reader.RTSTRUCT, spacing=reader.spacing)
goalslist = clinicalgoals_reader.clinical_goals_list
patient.clinicalGoalsList(goalslist)
#Generate probabilistic scenarios
prob_scenarios_generator = ScenariosGenerator(patientData=patient, max_displacement=10)
#initiate scenario calculation
prob_scenarios_generator.compute_scenarios()
#create probabilistic evaluator object
prob_evaluator = ProbabilisticEvaluator(patientData=patient)
prob_evaluator.evaluate()
prob_evaluator.write_to_csv(out_path)
