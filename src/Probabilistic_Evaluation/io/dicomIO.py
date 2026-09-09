import os
from typing import List

import numpy as np
import pydicom

from Probabilistic_Evaluation.utils import timed, Timer
from Probabilistic_Evaluation.logging_utils import log_call, logger
from Probabilistic_Evaluation.data import CTImage, DoseImage, ROIContour, RTStruct


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
        An instance of the CTImage class defined above, representing the CT image and its associated metadata.
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



def readData(inputPaths, maxDepth=-1) -> List[object]:
    """
    Load all data found at the given input path, using pydicom to parse DICOM files.

    Parameters
    ----------
    inputPaths: str or list
        Path or list of paths pointing to the data to be loaded.

    maxDepth: int, optional
        Maximum subfolder depth where the function will check for data to be loaded.
        Default is -1, which implies recursive search over infinite subfolder depth.

    Returns
    -------
    dataList: list of data objects
        The function returns a list of data objects containing the imported data.

    """

    fileLists = listAllFiles(inputPaths, maxDepth=maxDepth)
    dataList = []

    # read Dicom files
    dicomCT = {}

    for d, filePath in enumerate(fileLists["Dicom"]):
        logger.info(f'Loading data {d+1}/{len(fileLists["Dicom"])} Dicom files : {os.path.basename(filePath)}.')
        dcm = pydicom.dcmread(filePath)

        # Dicom CT
        if dcm.SOPClassUID == "1.2.840.10008.5.1.4.1.1.2":
            # Dicom CT are not loaded directly. All slices must first be classified according to SeriesInstanceUID.

            # this checks if a breathingPeriod file is present in the ct folder or in the parent of the ct folder
            # if yes, this ct slice is given a dynamic series index
            dynSeriesIndex = -1
            for txtFilePathIndex, txtFilePath in enumerate(fileLists["txt"]):
                if txtFilePath.endswith('breathingPeriod.txt'):
                    if os.path.dirname(txtFilePath) == os.path.dirname(filePath) or os.path.dirname(txtFilePath) == os.path.dirname(os.path.dirname(filePath)):
                        dynSeriesIndex = txtFilePathIndex
                        ## associer la slice à une série 4D

            newCT = 1
            for key in dicomCT:
                if key == dcm.SeriesInstanceUID:
                    dicomCT[dcm.SeriesInstanceUID].append(filePath)
                    newCT = 0
            if newCT == 1:
                dicomCT[dcm.SeriesInstanceUID] = [dynSeriesIndex, filePath]

        # Dicom dose
        elif dcm.SOPClassUID == "1.2.840.10008.5.1.4.1.1.481.2":
            dose = readDicomDose(filePath)
            dataList.append(dose)

        # Dicom RT Photon and Ion plan
        elif dcm.SOPClassUID in ("1.2.840.10008.5.1.4.1.1.481.8","1.2.840.10008.5.1.4.1.1.481.5"):
            plan = readDicomPlan(filePath)
            dataList.append(plan)

        # Dicom struct
        elif dcm.SOPClassUID == "1.2.840.10008.5.1.4.1.1.481.3":
            struct = readDicomStruct(filePath)
            dataList.append(struct)

        else:
            logger.warning("WARNING: Unknown SOPClassUID " + dcm.SOPClassUID + " for file " + filePath)
    
    # import Dicom CT images
    for key in dicomCT:
        logger.debug('in dataLoader readData, for key in dicomCT {}'.format(key))
        logger.debug(dicomCT[key][0])
        ct = readDicomCT(dicomCT[key][1:])
        dataList.append(ct)
                        
    return dataList

def listAllFiles(inputPaths, maxDepth=-1):
    """
    List all files of compatible data format from given input paths.

    Parameters
    ----------
    inputPaths: str or list
        Path or list of paths pointing to the data to be listed.

    maxDepth: int, optional
        Maximum subfolder depth where the function will check for files to be listed.
        Default is -1, which implies recursive search over infinite subfolder depth.

    Returns
    -------
    fileLists: dictionary
        The function returns a dictionary containing lists of data files classified according to their file format (Dicom, MHD).

    """

    fileLists = {
        "Dicom": [],
        "MHD": [],
        "Serialized": [],
        "txt": []
    }
    # if inputPaths is a list of path, then iteratively call this function with each path of the list
    if(isinstance(inputPaths, list)):
        for path in inputPaths:
            lists = listAllFiles(path, maxDepth=maxDepth)
            for key in fileLists:
                fileLists[key] += lists[key]

        return fileLists


    # check content of the input path
    if os.path.isdir(inputPaths):
        inputPathContent = sorted(os.listdir(inputPaths))
    else:
        inputPathContent = [inputPaths]
        inputPaths = ""


    for fileName in inputPathContent:
        filePath = os.path.join(inputPaths, fileName)

        # folders
        if os.path.isdir(filePath):
            if(maxDepth != 0):
                subfolderFileList = listAllFiles(filePath, maxDepth=maxDepth-1)
                for key in fileLists:
                    fileLists[key] += subfolderFileList[key]

        # files
        elif os.path.isfile(filePath):
            filetype = get_file_type(filePath)
            if filetype is None:
                logger.info("INFO: cannot recognize file format of " + filePath)
            else:
                fileLists[filetype].append(filePath)

    return fileLists


def get_file_type(filePath):
    # Is Dicom file ?
    dcm = None
    try:
        dcm = pydicom.dcmread(filePath)
    except:
        pass
    if(dcm != None):
        return 'Dicom'

    # Is MHD file ?
    with open(filePath, 'rb') as fid:
        data = fid.read(50*1024)  # read 50 kB, which should be more than enough for MHD header
        if data.isascii():
            if("ElementDataFile" in data.decode('ascii')): # recognize key from MHD header
                return 'MHD'

    # Is serialized file ?
    if filePath.endswith('.p') or filePath.endswith('.pbz2') or filePath.endswith('.pkl') or filePath.endswith('.pickle'):
        return "Serialized"

    # Is txt file ?
    if filePath.endswith('.txt'):
        return 'txt'

    logger.info("INFO: cannot recognize file format of " + filePath)
    return None


def readDicomCT(fileList) -> CTImage:
    """
    Read a CT series from a list of DICOM slice file paths using pydicom.

    Parameters
    ----------
    fileList : list of str
        Paths to the DICOM files of a single CT series.

    Returns
    -------
    CTImage
        The reconstructed 3D CT image with axis order (x, y, z).
    """
    slices = [pydicom.dcmread(filePath) for filePath in fileList]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    rowSpacing, colSpacing = (float(v) for v in slices[0].PixelSpacing)
    if len(slices) > 1:
        zSpacing = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))
    else:
        zSpacing = float(getattr(slices[0], 'SliceThickness', 1.0))
    spacing = np.array([colSpacing, rowSpacing, zSpacing], dtype=float)

    origin = np.asarray(slices[0].ImagePositionPatient, dtype=float)

    # pixel_array per slice has shape (rows, cols) i.e. (y, x)
    stack = np.stack(
        [s.pixel_array.astype(np.float32) * float(getattr(s, 'RescaleSlope', 1.0))
         + float(getattr(s, 'RescaleIntercept', 0.0)) for s in slices],
        axis=-1,
    )  # shape (y, x, z)
    imageArray = np.transpose(stack, (1, 0, 2))  # -> (x, y, z)

    gridSize = np.array(imageArray.shape, dtype=int)

    return CTImage(
        imageArray=imageArray,
        spacing=spacing,
        origin=origin,
        gridSize=gridSize,
        seriesInstanceUID=getattr(slices[0], 'SeriesInstanceUID', None),
    )


def readDicomDose(filePath) -> DoseImage:
    """
    Read an RTDOSE DICOM file using pydicom.

    Parameters
    ----------
    filePath : str
        Path to the RTDOSE DICOM file.

    Returns
    -------
    DoseImage
        The dose distribution with axis order (x, y, z), in Gy.
    """
    ds = pydicom.dcmread(filePath)

    doseScaling = float(getattr(ds, 'DoseGridScaling', 1.0))
    # pixel_array shape is (frames, rows, cols) i.e. (z, y, x)
    stack = ds.pixel_array.astype(np.float32) * doseScaling
    imageArray = np.transpose(stack, (2, 1, 0))  # -> (x, y, z)

    rowSpacing, colSpacing = (float(v) for v in ds.PixelSpacing)
    frameOffsets = getattr(ds, 'GridFrameOffsetVector', None)
    if frameOffsets is not None and len(frameOffsets) > 1:
        zSpacing = abs(float(frameOffsets[1]) - float(frameOffsets[0]))
    else:
        zSpacing = float(getattr(ds, 'SliceThickness', 1.0))
    spacing = np.array([colSpacing, rowSpacing, zSpacing], dtype=float)

    origin = np.asarray(ds.ImagePositionPatient, dtype=float)
    gridSize = np.array(imageArray.shape, dtype=int)

    return DoseImage(
        imageArray=imageArray,
        spacing=spacing,
        origin=origin,
        gridSize=gridSize,
        seriesInstanceUID=getattr(ds, 'SeriesInstanceUID', None),
    )


def readDicomStruct(filePath) -> RTStruct:
    """
    Read an RTSTRUCT DICOM file using pydicom.

    Parameters
    ----------
    filePath : str
        Path to the RTSTRUCT DICOM file.

    Returns
    -------
    RTStruct
        The set of ROI contours, each holding a name and a polygonMesh
        (list of flat DICOM ContourData arrays [x0, y0, z0, x1, y1, z1, ...]).
    """
    ds = pydicom.dcmread(filePath)

    roiNames = {roi.ROINumber: roi.ROIName for roi in getattr(ds, 'StructureSetROISequence', [])}

    contours = []
    for roiContour in getattr(ds, 'ROIContourSequence', []):
        roiNumber = roiContour.ReferencedROINumber
        name = roiNames.get(roiNumber, f'ROI_{roiNumber}')
        polygonMesh = [
            np.asarray(contourSeq.ContourData, dtype=float)
            for contourSeq in getattr(roiContour, 'ContourSequence', [])
        ]
        contours.append(ROIContour(name=name, polygonMesh=polygonMesh))

    return RTStruct(contours=contours, seriesInstanceUID=getattr(ds, 'SeriesInstanceUID', None))


def readDicomPlan(filePath):
    """
    Read an RT plan DICOM file using pydicom.

    Parameters
    ----------
    filePath : str
        Path to the RTPLAN DICOM file.

    Returns
    -------
    pydicom.dataset.FileDataset
        The raw pydicom dataset. Plan data is not otherwise consumed by this
        pipeline; it is kept for potential future use.
    """
    return pydicom.dcmread(filePath)
