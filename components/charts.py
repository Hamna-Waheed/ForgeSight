import plotly.express as px
import streamlit as st


def risk_distribution_chart(data):
    """
    Display machine risk distribution.
    """

    if data.empty:
        st.info("No machine data available.")
        return

    if "risk_level" not in data.columns:
        st.info("Risk level data is not available.")
        return

    counts = (
        data["risk_level"]
        .value_counts()
        .rename_axis("Risk Level")
        .reset_index(name="Machines")
    )

    figure = px.bar(
        counts,
        x="Risk Level",
        y="Machines",
        title="Machine Risk Distribution",
    )

    figure.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )