import pandas
import streamlit as st
import pandas as pd

Goal_types = ["DMAX", "DMEAN", "DMIN", "VX", "VXCC", "DX", "DXCC"]
comparison_operators = {'≤': True, '≥': False}

st.set_page_config(
    page_title="Clinical Goals",
    page_icon="🎯"
)

st.title("🎯 Define Clinical Goals")

try:
    _ = st.session_state.data.RTSTRUCT
except AttributeError:
    st.warning("Please load DICOM data first in the 'Load DICOM Data' page.")
    st.stop()

@st.dialog("Create Clinical Goal")
def create_clinical_goal(available_structures,index=None):

    # get the data if goal to edit if index is provided
    if index is not None:
        goal_to_edit = st.session_state.clinical_goals.iloc[index]
        selected_structure = goal_to_edit['structure']
        goal_type = goal_to_edit['goal_type']
        value = goal_to_edit['value']
        comparison_operator = goal_to_edit['condition']
        prescription = goal_to_edit['prescription']
        structure_index = list(available_structures).index(selected_structure)


    selected_structure = st.selectbox(
        "Structure Name",
        options=available_structures,
        index= structure_index if index is not None else None,
        placeholder="Select structure"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        goal_type = st.selectbox("Goal Type", Goal_types, index=Goal_types.index(goal_type) if index is not None else None)

    with col2:
        if goal_type == "VX":
            value = st.number_input("Value (Gy)", min_value=0.0, step=0.1,value = value if index is not None else 0.0)
        elif goal_type == "DX":
            value = st.number_input("Value (%)", min_value=0.0, max_value=100.0, step=0.1, value = value if index is not None else 0.0)
        elif goal_type == "DXCC":
            value = st.number_input("Value (cc)", min_value=0.0, step=0.1, value = value if index is not None else 0.0)
        elif goal_type == "VXCC":
            value = st.number_input("Value (Gy)", min_value=0.0, step=0.1, value = value if index is not None else 0.0)
        else:
            value = st.number_input("Value", disabled=True, value = None)

    with col3:
        comparison_operator = st.selectbox("Operator", list(comparison_operators.keys()), index=list(comparison_operators.keys()).index(comparison_operator) if index is not None else None)

    with col4:
        if goal_type in ["DMAX", "DMEAN", "DMIN", "DXCC", "DX"]:
            prescription = st.number_input("Dose (Gy)", min_value=0.0, step=0.1, value = prescription if index is not None else 0.0)
        elif goal_type == "VXCC":
            prescription = st.number_input("Volume (cc)", min_value=0.0, step=0.1, value = prescription if index is not None else 0.0)
        elif goal_type == "VX":
            prescription = st.number_input("Volume (%)", min_value=0.0, max_value=100.0, step=0.1, value = prescription if index is not None else 0.0)

    if st.button("Validate", type="primary", use_container_width=True):

        goal_data = {
            "structure": selected_structure,
            "goal_type": goal_type,
            "value": value,
            "condition": comparison_operator,
            "prescription": prescription
        }
        if index is not None:
            st.session_state.clinical_goals.iloc[index] = goal_data
            st.rerun()
        else:
            new_goal_df = pd.DataFrame([goal_data])

            st.session_state.clinical_goals = pd.concat(
                [st.session_state.clinical_goals, new_goal_df],
                ignore_index=True
            )

            st.rerun()


@st.dialog("Load Clinical Goals from CSV")
def load_clinical_goals_from_csv():

    uploaded_file = st.file_uploader("Upload Clinical Goals CSV", type="csv")

    if uploaded_file is not None:
        st.success("CSV loaded")



goal_definer_container = st.container()
col1, col2 = goal_definer_container.columns(2)

if "clinical_goals" not in st.session_state:
    st.session_state.clinical_goals = pandas.DataFrame(
        columns=["structure", "goal_type", "value", "condition", "prescription"])

with col1:
    if st.button("Create Clinical Goal", width="stretch"):
        available_structures = st.session_state.data.RTSTRUCT.keys()
        create_clinical_goal(available_structures)

with col2:
    if st.button("Load Clinical Goals from CSV", width="stretch"):
        load_clinical_goals_from_csv()


goal_list_container = st.container(border=True)
goal_list_container.markdown("### Goal list")
for i, goal in st.session_state.clinical_goals.iterrows():

    with goal_list_container.container(border=True):

        col1,col2,col3,col4,col5,col6,col7 = st.columns([2,1,1,1,1,1,1],gap="small")

        col1.write(goal["structure"])
        col2.write(goal["goal_type"])
        col3.write(goal["value"] if goal["value"] is not None else " ")
        col4.write(goal["condition"])
        col5.write(goal["prescription"])

        if col6.button("✏️", key=f"edit_{i}"):
            available_structures = st.session_state.data.RTSTRUCT.keys()
            create_clinical_goal(available_structures,index=i)

        if col7.button("🗑", key=f"delete_{i}"):

            st.session_state.clinical_goals = (
                st.session_state.clinical_goals
                .drop(i)
                .reset_index(drop=True)
            )

            st.rerun()
