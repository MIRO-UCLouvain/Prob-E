import streamlit as st
from tkinter import Tk, filedialog

st.set_page_config(
    page_title="Load DICOM Data",
    page_icon="📂"
)
st.title("📂 Load DICOM Data")

files = None

@st.dialog("Load DICOM files")
def load_dicom_dialog():
    files = st.file_uploader("Upload DICOM files", type=["dcm"], accept_multiple_files=True)
    if files:
        return files

def pick_folder():
    root = Tk()
    root.withdraw()  # Hide the Tkinter main window
    folder_selected = filedialog.askdirectory()
    root.destroy()  # Destroy the Tkinter main window
    return folder_selected


load_data_container = st.container(border=True)
load_data_container.markdown("### Upload DICOM Data")
col1, col2 = load_data_container.columns(2)

with col1:
    if st.button("Load DICOM files",width="stretch"):
        files = load_dicom_dialog()


with col2:
    if st.button("load DICOM Folder",width="stretch"):
        files = pick_folder()


if files is not None:
    try:
        with load_data_container.spinner("Loading DICOM Data..."):
          from Probabilistic_Evaluation.io.dicomIO import DicomReader
          dicomreader = DicomReader()
          dicomreader.load_dicom_series(files)
          st.session_state.data = dicomreader
          load_data_container.success("DICOM Data Loaded Successfully!")
    except Exception as e:
        pass

