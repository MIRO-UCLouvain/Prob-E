import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder

Goal_types = ["DMAX", "DMEAN", "DMIN", "VX", "VXCC", "DX", "DXCC"]
comparison_operators = {'≤': True, '≥': False}

st.set_page_config(
    page_title="Clinical Goals",
    page_icon="🎯"
)

st.title("🎯 Define Clinical Goals")

# Check if DICOM data is loaded
try:
    _ = st.session_state.data.RTSTRUCT
except AttributeError:
    st.warning("Please load DICOM data first in the 'Load DICOM Data' page.")
    st.stop()

# Initialize clinical goals
if "clinical_goals" not in st.session_state:
    st.session_state.clinical_goals = pd.DataFrame(
        columns=["structure", "goal_type", "value", "condition", "prescription"]
    )

# --- Clinical Goal Dialog ---
@st.dialog("Create Clinical Goal")
def create_clinical_goal(available_structures):
    selected_structure = st.selectbox(
        "Structure Name",
        options=available_structures,
        placeholder="Select structure"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        goal_type = st.selectbox("Goal Type", Goal_types)

    with col2:
        if goal_type == "VX":
            value = st.number_input("Value (Gy)", min_value=0.0, step=0.1)
        elif goal_type == "DX":
            value = st.number_input("Value (%)", min_value=0.0, max_value=100.0, step=0.1)
        elif goal_type == "DXCC":
            value = st.number_input("Value (cc)", min_value=0.0, step=0.1)
        elif goal_type == "VXCC":
            value = st.number_input("Value (Gy)", min_value=0.0, step=0.1)
        else:
            value = st.number_input("Value", disabled=True, value=0.0)

    with col3:
        comparison_operator = st.selectbox("Operator", list(comparison_operators.keys()))

    with col4:
        if goal_type in ["DMAX", "DMEAN", "DMIN", "DXCC", "DX"]:
            prescription = st.number_input("Dose (Gy)", min_value=0.0, step=0.1)
        elif goal_type == "VXCC":
            prescription = st.number_input("Volume (cc)", min_value=0.0, step=0.1)
        elif goal_type == "VX":
            prescription = st.number_input("Volume (%)", min_value=0.0, max_value=100.0, step=0.1)

    if st.button("Validate"):
        goal_data = {
            "structure": selected_structure,
            "goal_type": goal_type,
            "value": value,
            "condition": comparison_operator,
            "prescription": prescription
        }
        new_goal_df = pd.DataFrame([goal_data])
        st.session_state.clinical_goals = pd.concat(
            [st.session_state.clinical_goals, new_goal_df],
            ignore_index=True
        )
        st.rerun()


# --- Buttons to create goal ---
goal_definer_container = st.container()
col1, col2 = goal_definer_container.columns(2)

with col1:
    if st.button("Create Clinical Goal"):
        available_structures = list(st.session_state.data.RTSTRUCT.keys())
        create_clinical_goal(available_structures)

# --- Display AG Grid ---
goal_table_container = st.container()

gb = GridOptionsBuilder.from_dataframe(st.session_state.clinical_goals)

# Configure columns
gb.configure_column(
    "structure",
    editable=True,
    cellEditor="agSelectCellEditor",
    cellEditorParams={"values": list(st.session_state.data.RTSTRUCT.keys())},
    rowDrag=True
)
gb.configure_column(
    "goal_type",
    editable=True,
    cellEditor="agSelectCellEditor",
    cellEditorParams={"values": Goal_types}
)
gb.configure_column(
    "condition",
    editable=True,
    cellEditor="agSelectCellEditor",
    cellEditorParams={"values": list(comparison_operators.keys())}
)
gb.configure_column(
    "value",
    editable=True,
    type=["numericColumn", "numberColumnFilter"]
)
gb.configure_column(
    "prescription",
    editable=True,
    type=["numericColumn", "numberColumnFilter"]
)

# Enable column drag-and-drop
gb.configure_grid_options(
    enableSorting=False,
    enableFilter=False,
    enableColumnReorder=False,  # prevent column reorder by headers
    rowDragManaged=True         # allow only manual row drag
)

grid_options = gb.build()

grid_response = AgGrid(
    st.session_state.clinical_goals,
    gridOptions=grid_options,
    editable=True,
    fit_columns_on_grid_load=True
)

# Update session state with edited goals
updated_df = pd.DataFrame(grid_response["data"])
st.session_state.clinical_goals = updated_df