from Probabilistic_Evaluation.data import Scenario
from Probabilistic_Evaluation.core.sampling._abstractSamplingMethod import AbstractsamplingMethod
from Probabilistic_Evaluation.logging_utils import logger


class ScenariosGenerator:
    """
    A class to generate and evaluate probabilistic scenarios based on Voronoi cells.
    
    Attributes
    ----------
    sampling_method : AbstractsamplingMethod
        The sampling method used to generate scenarios.
    scenarios_list : list
        A list of generated scenarios.

    Methods
    -------
    generate_scenarios():
        Generates scenarios based on the sampling method.
    """

    def __init__(self, sampling_method: AbstractsamplingMethod):
        self.sampling_method = sampling_method
        self.scenarios_list = []
        self.generate_scenarios()

    def generate_scenarios(self):
        """
        Generate scenarios based on the sampling method.

        Returns
        -------
        None.
        """
        displacements, probabilities = self.sampling_method.analyticalSampling()
        for displacement, probability in zip(displacements, probabilities):
            scenario = Scenario(displacement=displacement, probability=probability)
            self.scenarios_list.append(scenario)
        logger.debug(f"Generated {len(displacements)} scenarios from {type(self.sampling_method).__name__}.analyticalSampling() with a total probability of {sum(probabilities)}.")
