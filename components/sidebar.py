import streamlit as st


def render_sidebar():
    """
    Render the ForgeSight application sidebar.
    """

    with st.sidebar:
        st.markdown("## ◈ FORGESIGHT")
        st.caption("Industrial Machine Intelligence")

        st.divider()

        st.markdown("### COMMAND CENTER")

        page = st.radio(
            "Navigation",
            [
                "Overview",
                "Operations",
                "Machine Intelligence",
                "Analytics",
                "Alerts",
                "System",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown("### SYSTEM")

        st.caption("Model")
        st.write("Random Forest v1.0")

        st.caption("Status")
        st.write("Operational")

        st.caption("Dataset")
        st.write("AI4I 2020")

    return page