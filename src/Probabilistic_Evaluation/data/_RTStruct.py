class RTStruct:
    """
    A collection of ROI contours extracted from an RTSTRUCT DICOM file.

    Attributes
    ----------
    name : list of str
        The names of all ROIs contained in the RTSTRUCT.
    seriesInstanceUID : str
        The DICOM SeriesInstanceUID referenced by the RTSTRUCT.
    """

    def __init__(self, contours, seriesInstanceUID=None):
        self._contours = contours
        self.name = [contour.name for contour in contours]
        self.seriesInstanceUID = seriesInstanceUID
