import copy
import datetime
import os
import pydicom
import numpy as np

from opentps.core.io.dicomIO import *
from opentps.core.io.dataLoader import *
from opentps.core.data.images._doseImage import *
from opentps.core.data.images._ctImage import *
from opentps.core.data._rtStruct import *



class DicomReader():
    def __init__(self):
        self.CT = None
        self.RTSTRUCT = None
        self.RTDOSE = None
        self.spacing = None
        self.data = None

    def load_dicom_series(self,directory):
        """
        Load a DICOM series from the specified directory.

        Args:
            directory (str): Path to the directory containing DICOM files.
        Returns:
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
                RTDOSE = key.imageArray
                print(f"RTDOSE Image shape: {RTDOSE.shape} and spacing: {key.spacing}")
                return RTDOSE
        raise ValueError("No DoseImage found in the provided DICOM series.")
    
    def readRTSTRUCT(self):
        for key in self.data:
            if isinstance(key, RTStruct):
                RTSTRUCT = key
                print(f"RTSTRUCT loaded with {len(RTSTRUCT.name)} structures.")
                return RTSTRUCT
        raise ValueError("No RTStruct found in the provided DICOM series.")

if __name__ == "__main__":
    dicomDir = r"C:\Users\rschyns\OneDrive - UCL\Job MIRO\OpenTPS\dataset_testing\Plan_UCL7"
    reader = DicomReader()
    reader.load_dicom_series(dicomDir)