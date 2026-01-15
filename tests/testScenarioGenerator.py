import numpy as np
import pytest
import matplotlib.pyplot as plt

from Probabilistic_Evaluation.core.scenariosGenerator import ScenariosGenerator
from Probabilistic_Evaluation.data import PatientData


class TestScenariosGenerator:
    def test_scenarios_generator_initialization(self):
        pass

    def test_probabilistic_scenarios(self, plot=True):
        # Example CT and dose images (3D numpy arrays)
        ctImage = np.zeros((50,50,50))
        ctImage[15:35,15:35,15:35] = 100  # Example high-density region
        doseImage = np.zeros((50,50,50))
        doseImage[20:30,20:30,20:30] = 50  # Example dose distribution
        targetMask = np.zeros((50,50,50),dtype=bool)  # Example target mask
        targetMask[22:28,22:28,22:28] = True  # Define target region
        maskdict = {'Target': targetMask}
        spacing = (1.0, 1.0, 2.0)  # Example spacing

        patientData = PatientData(ctImage, doseImage, maskdict, spacing=spacing)
        sigma = 5/3.2

        # test on patient data
        # assert patient.ctImage.shape == (50,50,50)
        pytest.approx(patientData.ctImage[20,20,20], 100)
        pytest.approx(patientData.doseImage[25,25,25], 50)
        assert patientData.maskDict['Target'][23,23,23] == True
        
        ps = ScenariosGenerator(patientData, max_displacement=10,sigmas=(sigma, sigma, sigma), sampling_method='voronoi')
        
        assert len(ps.scenarios_list) == ps.voronoi_cells.points.shape[0]

        # print("Evaluation results for all scenarios:", results)
        
        #2 plots, one showing the results as afucntion of scenario index and other one show probabilities of voronoi cells as function on cell(scenario) index
        if plot:
            plt.subplot(1, 1, 1)
            plt.plot(ps.voronoi_cells.probabilities_analytical, label='Analytical')
            plt.title("Voronoi Cell Probabilities")
            plt.xlabel("Voronoi Cell (Scenario) Index")
            plt.ylabel("Probability")
            plt.legend()
            plt.show()


            for i, point in enumerate(ps.voronoi_cells.points):
                if point.tolist() == [0,0,0]:
                    print(f"Scenario {i}: Displacement {point}, Probability: {ps.voronoi_cells.probabilities_analytical[i]}")
                if point.tolist() == [0,1,0]:
                    print(f"Scenario {i}: Displacement {point}, Probability: {ps.voronoi_cells.probabilities_analytical[i]}")
                if point.tolist() == [0,0,1]:
                    print(f"Scenario {i}: Displacement {point}, Probability: {ps.voronoi_cells.probabilities_analytical[i]}")
                    from Probabilistic_Evaluation.utils import shift_dose_image
                    shifted_dose = shift_dose_image(ps.doseImage, shift=point)
                    plt.imshow(shifted_dose[:,:,25], cmap='jet')
                    plt.colorbar(label='Dose')
                    plt.contour(ps.targetMask[:,:,25], colors='white', linewidths=0.5)
                    plt.title(f"Dose Distribution for Displacement {point}")
                    plt.show()

if __name__ == "__main__":
    TestScenariosGenerator().test_probabilistic_scenarios(plot=True)