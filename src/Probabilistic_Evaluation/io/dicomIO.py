from opentps.core.io.dicomIO import *
from opentps.core.io.dataLoader import *
from opentps.core.data.images import DoseImage, CTImage
from opentps.core.data import RTStruct
import numpy as np

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

    def __init__(self):
        self.CT: np.ndarray = None
        self.RTSTRUCT: dict = None
        self.RTDOSE: np.ndarray = None
        self.spacing: tuple = None
        self.data = None
        self.CTImage = None

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
        print(f"Loaded {len(self.data)} DICOM files from {directory}")
        self.CT,self.spacing = self.readCT()

        self.RTDOSE = self.readRTDOSE()
        self.RTSTRUCT = self.readRTSTRUCT()

    def readCT(self):
        for key in self.data:
            if isinstance(key, CTImage):
                self.CTImage = key
                CT = key.imageArray
                spacing = key.spacing
                print(f"CT Image shape: {CT.shape}, Spacing: {spacing}")
                return CT, spacing
        raise ValueError("No CTImage found in the provided DICOM series.")
    
    def readRTDOSE(self):
        # Only return the first DoseImage found
        for key in self.data:
            if isinstance(key, DoseImage):
                key.resample(self.spacing,self.CTImage.gridSize,self.CTImage.origin)
                RTdose = key.imageArray
                print(f"RTDOSE Image shape: {RTdose.shape} and spacing: {key.spacing}")
                return RTdose
        raise ValueError("No DoseImage found in the provided DICOM series.")

        
    
    def readRTSTRUCT(self):
        RTstruct_dictionary = {}
        for key in self.data:
            if isinstance(key, RTStruct):
                RTstruct = key
                print(f"RTSTRUCT loaded with {len(RTstruct.name)} structures.")
                for contour in RTstruct._contours:
                    print(f"Structure: {contour.name}")
                    RTstruct_dictionary[contour.name] = contour
                    
                return RTstruct_dictionary
        raise ValueError("No RTStruct found in the provided DICOM series.")




