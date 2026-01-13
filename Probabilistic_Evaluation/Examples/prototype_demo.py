from io.dicomIO import DicomReader
from core.ProbabilisticScenarios import ProbabilisticScenarios
from data._patientData import PatientData
from core.probabilisticEvaluator import ProbabilisticEvaluator


#Read images from DICOM
reader = DicomReader()
reader.load_dicom_series(r"C:\Users\geert\OneDrive - UCL\PhD\Probabilistic eval project St.Luc\ORL_001\ORL_001\test")

#Create patient data object
patient = patientData = PatientData(ctImage=reader.CT, doseImage=reader.RTDOSE, maskDict=reader.RTSTRUCT, spacing=reader.spacing)

#Generate probabilistic scenarios
prob_scenarios_generator = ProbabilisticScenarios(patientData=patient, max_displacement=10)
scenarios = prob_scenarios_generator.compute_scenarios()
