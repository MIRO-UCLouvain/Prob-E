from opentps.core.io.dicomIO import *
from opentps.core.io.dataLoader import *
from opentps.core.data.images import DoseImage, CTImage
from opentps.core.data import RTStruct
import numpy as np

from Probabilistic_Evaluation.utils import timed, Timer
from Probabilistic_Evaluation.logging_utils import log_call, logger


class DicomReader():
    """
    A class to read and process DICOM files including CT images, RTSTRUCT, and RTDOSE.

    Attributes
    ----------
    CT : np.ndarray
        The CT image data as a 3D numpy array.
    RTSTRUCT : dict
        A dictionary containing the RTSTRUCT data, with structure names as keys and contour data as values
    RTDOSE : np.ndarray
        The RTDOSE image data as a 3D numpy array.
    spacing : tuple
        The voxel spacing for the CT image, typically in the format (x_spacing, y_spacing, z_spacing).
    data : list
        A list to store the raw DICOM data loaded from the specified directory.
    CTImage : CTImage
        An instance of the CTImage class from opentps.core.data.images, representing the CT image and its associated metadata.
    """

    def __init__(self, spacing=None):
        self.RTSTRUCT: dict = None
        self.RTDOSE: np.ndarray = None
        self.spacing: tuple = spacing
        self.origin: tuple = None
        self.gridSize: tuple = None
        self.patientID = None
        self.data = None

    @timed
    def load_dicom_series(self,directory):
        """
        Load a DICOM series from the specified directory.

        Parameters
        ----------
        directory : str
            The path to the directory containing the DICOM files.

        Returns
        -------
            None
        """
        self.data = readData(directory)
        self.RTDOSE = self.readRTDOSE()
        self.RTSTRUCT = self.readRTSTRUCT()

    def readCT(self):
        for key in self.data:
            if isinstance(key, CTImage):
                CT = key.imageArray
                spacing = key.spacing
                origin = key.origin
                gridSize = key.gridSize
                self.patientID = key.seriesInstanceUID
                logger.info(f"CT Image shape: {CT.shape}, Spacing: {spacing}")
                logger.debug(f"CT Image origin: {origin}, Grid Size: {gridSize}")
                if hasattr(key, 'ImagepositionPatient'):
                    logger.debug(f"CT Image Position Patient: {key.ImagepositionPatient}")
                if hasattr(key, 'ImageOrientationPatient'):
                    logger.debug(f"CT Image Orientation Patient: {key.ImageOrientationPatient}")
                return CT, spacing, origin, gridSize
        raise ValueError("No CTImage found in the provided DICOM series.")
    
    def readRTDOSE(self):
        # Only return the first DoseImage found
        for key in self.data:
            if isinstance(key, DoseImage):
                if self.spacing is None :
                    self.spacing = key.spacing
                    self.gridSize = key.gridSize
                else:
                    newGridSize = (int(key.gridSize[0] * key.spacing[0] / self.spacing[0]),
                                   int(key.gridSize[1] * key.spacing[1] / self.spacing[1]),
                                   int(key.gridSize[2] * key.spacing[2] / self.spacing[2]))
                    key.resample(self.spacing,newGridSize,key.origin)
                    self.gridSize = newGridSize

                RTdose = key.imageArray
                self.origin = key.origin
                self.patientID = key.seriesInstanceUID
                logger.info(f"RTDOSE Image shape: {RTdose.shape} and spacing: {key.spacing}")
                logger.debug(f"RTDOSE Image origin: {key.origin}, Grid Size: {key.gridSize}")
                logger.debug("Max dose value in RTDOSE: {:.2f}".format(np.max(RTdose)))
                logger.debug("Min dose value in RTDOSE: {:.2f}".format(np.min(RTdose)))
                if hasattr(key, 'ImagepositionPatient'):
                    logger.debug(f"Dose Image Position Patient: {key.ImagepositionPatient}")
                if hasattr(key, 'ImageOrientationPatient'):
                    logger.debug(f"Dose Image Orientation Patient: {key.ImageOrientationPatient}")
                return RTdose
        raise ValueError("No DoseImage found in the provided DICOM series.")

    
    def readRTSTRUCT(self):
        RTstruct_dictionary = {}
        for key in self.data:
            if isinstance(key, RTStruct):
                RTstruct = key
                logger.info(f"RTSTRUCT loaded with {len(RTstruct.name)} structures.")
                for contour in RTstruct._contours:
                    logger.info(f"Structure: {contour.name}")
                    RTstruct_dictionary[contour.name] = contour
                    
                return RTstruct_dictionary
        raise ValueError("No RTStruct found in the provided DICOM series.")




