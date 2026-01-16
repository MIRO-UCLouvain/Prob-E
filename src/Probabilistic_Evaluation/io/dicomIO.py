from opentps.core.io.dicomIO import *
from opentps.core.io.dataLoader import *
from opentps.core.data.images import DoseImage, CTImage
from opentps.core.data import RTStruct

class DicomReader():
    """
    A class to read and process DICOM files including CT images, RTSTRUCT, and RTDOSE.

    Attributes
    ----------
        CT (ndarray): The CT image data.
        RTSTRUCT (dict): The RT structure data.
        RTDOSE (ndarray): The RT dose image data.
        spacing (tuple): The spacing of the CT image.
    """

    def __init__(self):
        self.CT: np.ndarray = None
        self.RTSTRUCT: dict = None
        self.RTDOSE: np.ndarray = None
        self.spacing: tuple = None
        self.data = None

    def load_dicom_series(self,directory):
        """
        Load a DICOM series from the specified directory.

        Args:
            directory (str): Path to the directory containing DICOM files.

        Returns
        -------
            list: List of pydicom Dataset objects sorted by InstanceNumber.
        """
        self.data = readData(directory)
        print(f"Loaded {len(self.data)} DICOM files from {directory}")
        self.CT,self.spacing = self.readCT()

        self.RTDOSE = self.readRTDOSE()
        self.RTSTRUCT = self.readRTSTRUCT()
    


        # CTImage = readDicomCT(directory)
        # self.CT = CTImage.imageArray
        # self.spacing = CTImage.spacing()

        # RTdoseImage = readDicomRTDose(directory)
        # self.RTDOSE = RTdoseImage.imageArray

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
                    RTstruct_dictionary[contour.name] = contour.getBinaryMask(origin=self.CTImage.origin, gridSize=self.CTImage.gridSize, spacing=self.spacing).imageArray
                    
                return RTstruct_dictionary
        raise ValueError("No RTStruct found in the provided DICOM series.")

if __name__ == "__main__":
    dicomDir = r"your path"
    reader = DicomReader()
    reader.load_dicom_series(dicomDir)