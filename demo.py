import os

from src.Probabilistic_Evaluation.data._patientData import PatientData
from src.Probabilistic_Evaluation.data.uncertaintyModel._Gaussian3D import Gaussian3DUncertaintyModel

from src.Probabilistic_Evaluation.io.dicomIO import DicomReader
from src.Probabilistic_Evaluation.io.clinicalgoalsIO import clinicalgoalsreader

from src.Probabilistic_Evaluation.core.probabilisticEvaluator import ProbabilisticEvaluator
from src.Probabilistic_Evaluation.core.scenariosGenerator import ScenariosGenerator
from src.Probabilistic_Evaluation.core.sampling.voronoiSampling import VoronoiSampling

import numpy as np

path = r"/home/matijs/NAS/25111201"
patient_path = path + r"/test"
output_path = path + r'/test/demo.csv'
clincialgoals_path = path + r'/test/testgoals.json'


dicomreader = DicomReader()
dicomreader.load_dicom_series(patient_path)
rtstruct = dicomreader.RTSTRUCT
CT = dicomreader.CTImage
ct_image = dicomreader.CT
dose_image = dicomreader.RTDOSE
spacing = dicomreader.spacing
goalreader = clinicalgoalsreader(rtstruct,dicomreader.CTImage)
goalreader.load_JSON_list(clincialgoals_path)
resampledRTstruct = goalreader.resampledMasks

patient = PatientData(ct_image, dose_image, resampledRTstruct, spacing)
print(f"Loaded clinical goals for masks: {list(resampledRTstruct.keys())}")
patient.clinicalGoalsList = goalreader.clinical_goals_list
parameters = {'mu_x': 0, 'mu_y': 0, 'mu_z': 0, 'sigma_x': 10 / (3.2), 'sigma_y': 10 / (3.2), 'sigma_z': 10 / (3.2)}
model = Gaussian3DUncertaintyModel(parameters)
proba = 0.90
sampler = VoronoiSampling(uncertaintyModel=model, max_displacements=np.array([20.0,20.0,20.0]), spacing=np.array([spacing[0]*4,spacing[1]*4,spacing[2]*4]), probabilityMass=proba)
print("Number of scenarios generated:", len(sampler.voronoiPoints))
evaluator = ProbabilisticEvaluator(patient, sampler=sampler,nThreads=-1,computeCumulativePassingRate=True,computeVWMin=False,computeVWMax=False)
table = evaluator.evaluate()
evaluator.saveTableToCSV(table)
evaluator.saveTableToHTML(table)
# print(table.to_string())