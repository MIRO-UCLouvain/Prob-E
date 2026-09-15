import os
from concurrent.futures import ThreadPoolExecutor, wait
from typing import List

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from Probabilistic_Evaluation.utils import timed, Timer, resample_image, resampling_grid
from Probabilistic_Evaluation.logging_utils import log_call, logger
from Probabilistic_Evaluation.data import CTImage, DoseImage, ROIContour, RTStruct

_CT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.2"
_RTDOSE_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.2"
_RTSTRUCT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.3"
_RTPLAN_SOP_CLASSES = ("1.2.840.10008.5.1.4.1.1.481.5", "1.2.840.10008.5.1.4.1.1.481.8")
# modalities that often share a patient folder but are not used here: skipped without a warning
_IGNORED_MODALITIES = ("MR", "REG")
# the only tags read while scanning a folder; pixel data and all other elements are skipped
_HEADER_TAGS = ["SOPClassUID", "Modality", "SeriesInstanceUID"]


class DicomReader():
    """
    A class to read and process DICOM files including CT images, RTSTRUCT, and RTDOSE.

    The directory is scanned once, reading only the DICOM headers, and the RTDOSE and RTSTRUCT to use are
    selected before any image data is read. The CT is only read when ``loadCT`` is True.

    Attributes
    ----------
    RTSTRUCT : dict
        A dictionary containing the RTSTRUCT data, with structure names as keys and contour data as values
    RTDOSE : np.ndarray
        The RTDOSE image data as a 3D numpy array.
    spacing : tuple
        The voxel spacing for the CT image, typically in the format (x_spacing, y_spacing, z_spacing).
    data : list
        The data objects loaded from the specified directory: the selected DoseImage and RTStruct, followed by
        one CTImage per CT series when ``loadCT`` is True.
    loadCT : bool
        Whether the CT series are read. The evaluation itself only needs the dose and the structures. Default True.
    nThreads : int
        Number of threads reading the DICOM headers while scanning the directory. Values below 1 use all logical
        CPUs. Default -1.
    """

    def __init__(self, spacing=None, loadCT=True, nThreads=-1):
        self.RTSTRUCT: dict = None
        self.RTDOSE: np.ndarray = None
        self.spacing: tuple = spacing
        self.origin: tuple = None
        self.gridSize: tuple = None
        self.patientID = None
        self.data = None
        self.loadCT = loadCT
        self.nThreads = nThreads

    @timed
    def load_dicom_series(self, directory, doseFile=None, structFile=None):
        """
        Load the RTDOSE, the RTSTRUCT and, with ``loadCT``, the CT series from the specified directory.

        Parameters
        ----------
        directory : str
            The path to the directory containing the DICOM files. Subfolders are searched too.
        doseFile : str, optional
            Path or file name of the RTDOSE to load. Required when the directory holds more than one RTDOSE.
        structFile : str, optional
            Path or file name of the RTSTRUCT to load. Required when the directory holds more than one RTSTRUCT.

        Returns
        -------
            None

        Raises
        ------
        ValueError
            If no RTDOSE or RTSTRUCT is found, if several are found and none is selected, or if the selection
            does not match exactly one file. Raised after the header scan, before any image data is read.
        """
        dicomFiles = scanDicomFiles(directory, nThreads=self.nThreads)
        dosePath = _selectFile(dicomFiles["RTDOSE"], doseFile, "RTDOSE", "doseFile")
        structPath = _selectFile(dicomFiles["RTSTRUCT"], structFile, "RTSTRUCT", "structFile")
        logger.info(f"Loading RTDOSE {dosePath} and RTSTRUCT {structPath}")

        self.data = [readDicomDose(dosePath), readDicomStruct(structPath)]
        if self.loadCT:
            self.data += [readDicomCT(ctFiles) for ctFiles in dicomFiles["CT"].values()]
        self.RTDOSE = self.readRTDOSE()
        self.RTSTRUCT = self.readRTSTRUCT()

    def readCT(self):
        if not self.loadCT:
            raise ValueError("The CT was not loaded, create the DicomReader with loadCT=True to read it.")
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
                    # target grid covering the same physical (voxel-edge) extent as the original dose grid
                    newGridSize, newOrigin = resampling_grid(key.gridSize, key.spacing, key.origin, self.spacing)
                    key.imageArray = resample_image(
                        key.imageArray, key.spacing, key.origin, self.spacing, newGridSize, newOrigin
                    )
                    key.spacing = np.asarray(self.spacing, dtype=float)
                    key.gridSize = np.asarray(newGridSize, dtype=int)
                    key.origin = np.asarray(newOrigin, dtype=float)
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


def _selectFile(filePaths, requested, label, argumentName):
    """
    Return the one file of a type to load, or raise before any image data is read.

    ``requested`` is matched against the full path or the file name of each candidate.
    """
    names = ", ".join(os.path.basename(filePath) for filePath in filePaths)
    if requested is not None:
        target = os.path.normcase(os.path.abspath(requested))
        matches = [
            filePath for filePath in filePaths
            if os.path.basename(filePath) == requested or os.path.normcase(os.path.abspath(filePath)) == target
        ]
        if len(matches) != 1:
            raise ValueError(f"{argumentName}={requested!r} matches {len(matches)} of the {label} files found: [{names}].")
        return matches[0]
    if not filePaths:
        raise ValueError(f"No {label} found in the provided DICOM series.")
    if len(filePaths) > 1:
        raise ValueError(f"Found {len(filePaths)} {label} files, pass {argumentName}= to choose one: [{names}].")
    return filePaths[0]


def readData(inputPaths, maxDepth=-1, nThreads=-1) -> List[object]:
    """
    Load all DICOM CT, RTDOSE, RTPLAN and RTSTRUCT data found at the given input path.

    Parameters
    ----------
    inputPaths: str or list
        Path or list of paths pointing to the data to be loaded.

    maxDepth: int, optional
        Maximum subfolder depth where the function will check for data to be loaded.
        Default is -1, which implies recursive search over infinite subfolder depth.

    nThreads: int, optional
        Number of threads reading the DICOM headers. Values below 1 use all logical CPUs. Default -1.

    Returns
    -------
    dataList: list of data objects
        The function returns a list of data objects containing the imported data.

    """
    dicomFiles = scanDicomFiles(inputPaths, maxDepth=maxDepth, nThreads=nThreads)
    dataList = [readDicomDose(filePath) for filePath in dicomFiles["RTDOSE"]]
    dataList += [readDicomPlan(filePath) for filePath in dicomFiles["RTPLAN"]]
    dataList += [readDicomStruct(filePath) for filePath in dicomFiles["RTSTRUCT"]]
    dataList += [readDicomCT(ctFiles) for ctFiles in dicomFiles["CT"].values()]
    return dataList


def scanDicomFiles(inputPaths, maxDepth=-1, nThreads=-1) -> dict:
    """
    Classify the DICOM files found at the given input path, reading only their headers.

    The headers are read in parallel, because opening each file is the dominant cost on network shares, and pixel
    data is never read. MR and registration files are skipped silently, other unsupported DICOM files with one
    warning per SOP class, and files that are not DICOM at debug level.

    Parameters
    ----------
    inputPaths: str or list
        Path or list of paths pointing to the data to be scanned.

    maxDepth: int, optional
        Maximum subfolder depth where the function will check for files.
        Default is -1, which implies recursive search over infinite subfolder depth.

    nThreads: int, optional
        Number of threads reading the headers. Values below 1 use all logical CPUs. Default -1.

    Returns
    -------
    dicomFiles: dict
        ``{"CT": {SeriesInstanceUID: [paths]}, "RTDOSE": [paths], "RTSTRUCT": [paths], "RTPLAN": [paths]}``,
        with the paths in file listing order.
    """
    filePaths = listFiles(inputPaths, maxDepth=maxDepth)
    headers = readDicomHeaders(filePaths, nThreads=nThreads)

    dicomFiles = {"CT": {}, "RTDOSE": [], "RTSTRUCT": [], "RTPLAN": []}
    ignored, unsupported, nonDicom = {}, {}, 0
    for filePath, header in zip(filePaths, headers):
        if header is None:
            nonDicom += 1
            continue
        sopClassUID = str(header.get("SOPClassUID", ""))
        modality = str(header.get("Modality", ""))
        if sopClassUID == _CT_SOP_CLASS:
            dicomFiles["CT"].setdefault(str(header.get("SeriesInstanceUID", "")), []).append(filePath)
        elif sopClassUID == _RTDOSE_SOP_CLASS:
            dicomFiles["RTDOSE"].append(filePath)
        elif sopClassUID == _RTSTRUCT_SOP_CLASS:
            dicomFiles["RTSTRUCT"].append(filePath)
        elif sopClassUID in _RTPLAN_SOP_CLASSES or modality == "RTPLAN":
            dicomFiles["RTPLAN"].append(filePath)
        elif modality in _IGNORED_MODALITIES:
            ignored[modality] = ignored.get(modality, 0) + 1
        else:
            unsupported.setdefault(sopClassUID, []).append(filePath)

    for sopClassUID, paths in unsupported.items():
        logger.warning(f"Skipped {len(paths)} DICOM file(s) with unsupported SOPClassUID {sopClassUID}, e.g. {paths[0]}")
    nSlices = sum(len(ctFiles) for ctFiles in dicomFiles["CT"].values())
    logger.info(
        f"Scanned {len(filePaths)} files: {nSlices} CT slices in {len(dicomFiles['CT'])} series, "
        f"{len(dicomFiles['RTDOSE'])} RTDOSE, {len(dicomFiles['RTSTRUCT'])} RTSTRUCT, {len(dicomFiles['RTPLAN'])} RTPLAN, "
        f"ignored {ignored if ignored else 'none'}, {nonDicom} non-DICOM files."
    )
    return dicomFiles


def readDicomHeaders(filePaths, nThreads=-1) -> list:
    """
    Read the classification tags of many files in parallel.

    Parameters
    ----------
    filePaths : list of str
        Paths of the files to read.
    nThreads : int, optional
        Number of threads reading the headers. Values below 1 use all logical CPUs. Default -1.

    Returns
    -------
    list
        For each file, in the order of ``filePaths``, a pydicom dataset holding only SOPClassUID, Modality and
        SeriesInstanceUID, or None if the file is not a readable DICOM file.
    """
    if nThreads is None or int(nThreads) < 1:
        nThreads = os.cpu_count() or 1
    executor = ThreadPoolExecutor(max_workers=int(nThreads))
    futures = [executor.submit(_readDicomHeader, filePath) for filePath in filePaths]
    try:
        # wait in short slices: a blocking wait on Windows would hold back Ctrl+C until every header is read
        pending = futures
        while pending:
            pending = wait(pending, timeout=0.5).not_done
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return [future.result() for future in futures]


def _readDicomHeader(filePath):
    try:
        return pydicom.dcmread(filePath, stop_before_pixels=True, specific_tags=_HEADER_TAGS)
    except InvalidDicomError:
        logger.debug(f"Not a DICOM file, skipped: {filePath}")
    except Exception as error:
        logger.warning(f"Could not read the DICOM header of {filePath}, skipped: {type(error).__name__}: {error}")
    return None


def listFiles(inputPaths, maxDepth=-1) -> List[str]:
    """
    List all files found at the given input paths.

    Parameters
    ----------
    inputPaths: str or list
        Path or list of paths pointing to the files or folders to be listed.

    maxDepth: int, optional
        Maximum subfolder depth where the function will check for files to be listed.
        Default is -1, which implies recursive search over infinite subfolder depth.

    Returns
    -------
    filePaths: list of str
        The file paths, sorted by name within each folder.

    Raises
    ------
    FileNotFoundError
        If an input path does not exist.
    """
    if isinstance(inputPaths, (list, tuple)):
        return [filePath for inputPath in inputPaths for filePath in listFiles(inputPath, maxDepth=maxDepth)]
    if os.path.isfile(inputPaths):
        return [os.fspath(inputPaths)]
    if not os.path.isdir(inputPaths):
        raise FileNotFoundError(f"No such file or directory: {inputPaths}")

    filePaths = []
    # scandir entries carry the file type from the directory listing, avoiding a stat call per file
    with os.scandir(inputPaths) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)
    for entry in entries:
        if entry.is_dir():
            if maxDepth != 0:
                filePaths += listFiles(entry.path, maxDepth=maxDepth - 1)
        elif entry.is_file():
            filePaths.append(entry.path)
    return filePaths


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
