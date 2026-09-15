import streamlit as st


def metric_card(
    title: str,
    value: str,
    subtitle: str = "",
):
    """
    Display a simple ForgeSight metric card.
    """

    st.metric(
        label=title,
        value=value,
        delta=subtitle if subtitle else None,
    )