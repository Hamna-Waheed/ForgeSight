import streamlit as st


def render_machine_profile(result: dict):
    """
    Display prediction results for a machine.
    """

    if not result.get("success"):
        st.error("Prediction could not be completed.")

        for error in result.get("errors", []):
            st.warning(error)

        return

    st.subheader("Machine Health")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Health Index",
            f"{result['health_index']}/100",
        )

    with col2:
        st.metric(
            "Failure Probability",
            f"{result['failure_probability']}%",
        )

    with col3:
        st.metric(
            "Risk Level",
            result["risk_level"],
        )

    st.divider()

    st.subheader("Recommended Action")

    st.info(result["recommendation"])

    st.subheader("Derived Indicators")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Temperature Difference",
            f"{result['temperature_difference']} K",
        )

    with col2:
        st.metric(
            "Power Proxy",
            f"{result['power_proxy']}",
        )