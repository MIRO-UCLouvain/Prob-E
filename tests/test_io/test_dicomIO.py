"""DICOM import: header-only scan, RTDOSE/RTSTRUCT selection and dose resampling.

The DICOM files are tiny synthetic datasets written to a temporary folder.
"""

import logging

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from scipy.ndimage import map_coordinates

import Probabilistic_Evaluation.io.dicomIO as dicomIO
from Probabilistic_Evaluation.io.dicomIO import DicomReader, listFiles, scanDicomFiles
from Probabilistic_Evaluation.utils import resample_image, resampling_grid

logging.getLogger("ProbEval").addHandler(logging.NullHandler())

CT, MR, REG = "1.2.840.10008.5.1.4.1.1.2", "1.2.840.10008.5.1.4.1.1.4", "1.2.840.10008.5.1.4.1.1.66.1"
RTDOSE, RTSTRUCT = "1.2.840.10008.5.1.4.1.1.481.2", "1.2.840.10008.5.1.4.1.1.481.3"
PIXEL_MODULE = dict(SamplesPerPixel=1, PhotometricInterpretation="MONOCHROME2", BitsAllocated=16, BitsStored=16, HighBit=15, PixelRepresentation=0)


# ----------------------------------------------------------------------------- synthetic DICOM files
def _write(path, sopClassUID, modality, **elements):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sopClassUID
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), Dataset(), file_meta=meta, preamble=b"\0" * 128)
    if int(pydicom.__version__.split(".")[0]) < 3:  # pydicom 2.x takes the encoding from the dataset
        ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.SOPClassUID, ds.SOPInstanceUID, ds.Modality = sopClassUID, meta.MediaStorageSOPInstanceUID, modality
    ds.SeriesInstanceUID = generate_uid()
    for keyword, value in elements.items():
        setattr(ds, keyword, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path))
    return path


def _dose(path):
    frames = np.arange(24, dtype=np.uint16).reshape(4, 2, 3)  # (z, y, x)
    return _write(path, RTDOSE, "RTDOSE", Rows=2, Columns=3, NumberOfFrames=4, PixelSpacing=[2.0, 2.0],
                  GridFrameOffsetVector=[0.0, 2.0, 4.0, 6.0], ImagePositionPatient=[0.0, 0.0, 0.0],
                  DoseGridScaling=0.01, PixelData=frames.tobytes(), **PIXEL_MODULE)


def _struct(path):
    roi = Dataset()
    roi.ROINumber, roi.ROIName = 1, "GTV"
    contour = Dataset()
    contour.ContourData = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0]
    roiContour = Dataset()
    roiContour.ReferencedROINumber, roiContour.ContourSequence = 1, [contour]
    return _write(path, RTSTRUCT, "RTSTRUCT", StructureSetROISequence=[roi], ROIContourSequence=[roiContour])


@pytest.fixture
def patientFolder(tmp_path):
    series = generate_uid()
    for k in range(3):
        _write(tmp_path / "CT" / f"CT{k}.dcm", CT, "CT", SeriesInstanceUID=series, Rows=2, Columns=3,
               PixelSpacing=[1.0, 1.0], ImagePositionPatient=[0.0, 0.0, 2.0 * k], RescaleSlope=1, RescaleIntercept=-1000,
               PixelData=np.zeros((2, 3), np.uint16).tobytes(), **PIXEL_MODULE)
    _dose(tmp_path / "RD.dcm")
    _struct(tmp_path / "RS.dcm")
    _write(tmp_path / "MR" / "MR0.dcm", MR, "MR")
    _write(tmp_path / "REG.dcm", REG, "REG")
    (tmp_path / "notes.txt").write_text("not DICOM")
    return tmp_path


# ----------------------------------------------------------------------------- scan and selection
def test_scan_classifies_on_headers_and_skips_mr_reg_and_other_files(patientFolder, monkeypatch):
    warnings = []
    monkeypatch.setattr(dicomIO.logger, "warning", lambda message, *args, **kwargs: warnings.append(message))
    files = scanDicomFiles(patientFolder, nThreads=4)
    assert [len(ctFiles) for ctFiles in files["CT"].values()] == [3]
    assert len(files["RTDOSE"]) == 1 and len(files["RTSTRUCT"]) == 1 and files["RTPLAN"] == []
    assert warnings == []  # MR, REG and the text file are skipped silently
    assert len(listFiles(patientFolder, maxDepth=0)) == 4  # RD, RS, REG and notes.txt, subfolders not searched


def test_reader_skips_the_ct_by_default_and_loads_it_on_request(patientFolder):
    reader = DicomReader()  # the evaluation only needs the dose and the structures
    reader.load_dicom_series(patientFolder)
    assert np.allclose(reader.RTDOSE, np.transpose(np.arange(24).reshape(4, 2, 3) * 0.01, (2, 1, 0)))
    assert list(reader.RTSTRUCT) == ["GTV"]
    assert not any(isinstance(item, dicomIO.CTImage) for item in reader.data)
    with pytest.raises(ValueError, match="loadCT=True"):
        reader.readCT()

    reader = DicomReader(loadCT=True)
    reader.load_dicom_series(patientFolder)
    ct, spacing, _, gridSize = reader.readCT()
    assert list(gridSize) == [3, 2, 3] and np.allclose(spacing, [1.0, 1.0, 2.0]) and np.allclose(ct, -1000.0)


def test_header_scan_uses_all_cpus_when_the_thread_count_is_not_positive(patientFolder, monkeypatch):
    import os

    workers = []
    executor = dicomIO.ThreadPoolExecutor
    monkeypatch.setattr(dicomIO, "ThreadPoolExecutor", lambda max_workers: workers.append(max_workers) or executor(max_workers=max_workers))
    for nThreads in (-1, 0, 3):
        scanDicomFiles(patientFolder, nThreads=nThreads)
    assert workers == [os.cpu_count(), os.cpu_count(), 3]


def test_ambiguous_or_missing_files_raise_before_any_image_is_read(patientFolder, monkeypatch):
    def unexpectedRead(*args, **kwargs):
        raise AssertionError("image data read before the file selection was checked")

    _dose(patientFolder / "plan2" / "RD.dcm")
    monkeypatch.setattr(dicomIO, "readDicomDose", unexpectedRead)
    with pytest.raises(ValueError, match="doseFile="):
        DicomReader().load_dicom_series(patientFolder)
    with pytest.raises(ValueError, match="matches 2"):  # the same file name in two folders
        DicomReader().load_dicom_series(patientFolder, doseFile="RD.dcm")
    (patientFolder / "RS.dcm").unlink()
    with pytest.raises(ValueError, match="No RTSTRUCT"):
        DicomReader().load_dicom_series(patientFolder, doseFile=str(patientFolder / "plan2" / "RD.dcm"))
    monkeypatch.undo()

    _struct(patientFolder / "RS.dcm")
    reader = DicomReader()
    reader.load_dicom_series(patientFolder, doseFile=str(patientFolder / "plan2" / "RD.dcm"))
    assert reader.RTDOSE.shape == (3, 2, 4)


# ----------------------------------------------------------------------------- resampling
def _pointSampled(image, spacing, origin, newSpacing, newGridSize, newOrigin):
    """Reference: trilinear interpolation at explicitly listed target voxel centres."""
    axes = [(newOrigin[a] + np.arange(newGridSize[a]) * newSpacing[a] - origin[a]) / spacing[a] for a in range(3)]
    return map_coordinates(image, np.meshgrid(*axes, indexing="ij"), order=1, mode="nearest")


@pytest.mark.parametrize("newSpacing", [[1.0, 1.0, 1.0], [0.8, 1.3, 2.0], [3.0, 3.0, 3.0]])
def test_resample_image_matches_point_sampling(newSpacing):
    image = np.random.default_rng(0).random((20, 16, 12))
    spacing, origin, newSpacing = np.array([2.0, 2.0, 2.5]), np.array([-10.0, 3.0, 7.5]), np.array(newSpacing)
    newGridSize, newOrigin = resampling_grid(image.shape, spacing, origin, newSpacing)
    resampled = resample_image(image, spacing, origin, newSpacing, newGridSize, newOrigin, antialias=False)
    assert resampled.shape == tuple(newGridSize) and resampled.dtype == image.dtype
    assert np.allclose(resampled, _pointSampled(image, spacing, origin, newSpacing, newGridSize, newOrigin), rtol=0, atol=1e-12)


def test_resample_image_returns_the_input_when_the_grid_is_unchanged():
    image = np.ones((4, 5, 6), dtype=np.float32)
    spacing, origin = np.array([2.0, 2.0, 3.0]), np.array([-1.0, 0.5, 2.0])
    newGridSize, newOrigin = resampling_grid(image.shape, spacing, origin, spacing)
    assert resample_image(image, spacing, origin, spacing, newGridSize, newOrigin) is image
