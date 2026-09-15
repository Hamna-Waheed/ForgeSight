import streamlit as st
import pandas as pd
import plotly.express as px

from services.predictor import (
    predict_machine_safe,
    load_model,
    MODEL_FEATURES,
)
from services.health_score import (
    calculate_health_index,
    determine_risk_level,
)
from utils.data_loader import load_processed_data
from utils.config import APP_NAME, APP_VERSION


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ForgeSight | Machine Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #080B0D;
        color: #E8EEF0;
    }

    [data-testid="stSidebar"] {
        background-color: #10161A;
        border-right: 1px solid #263139;
    }

    [data-testid="stMetric"] {
        background-color: #10161A;
        border: 1px solid #263139;
        padding: 18px;
        border-radius: 8px;
    }

    .main-title {
        font-size: 34px;
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #8D9AA1;
        font-size: 15px;
        margin-bottom: 28px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 14px;
    }

    .status-good {
        color: #3DDC97;
        font-weight: 600;
    }

    .status-warning {
        color: #F2C94C;
        font-weight: 600;
    }

    .status-danger {
        color: #EB5757;
        font-weight: 600;
    }

    .info-box {
        background-color: #10161A;
        border: 1px solid #263139;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

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
    st.markdown(
        '<span class="status-good">● Operational</span>',
        unsafe_allow_html=True,
    )

    st.caption("Dataset")
    st.write("AI4I 2020")


# ============================================================
# DATA + MODEL HELPERS
# ============================================================

@st.cache_data
def get_dataset():
    return load_processed_data()


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def generate_fleet_analysis():

    data = load_processed_data().copy()
    model = load_model()

    missing_columns = [
        column
        for column in MODEL_FEATURES
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    model_input = data[MODEL_FEATURES].copy()

    probabilities = model.predict_proba(model_input)[:, 1]

    data["failure_probability"] = probabilities

    data["health_index"] = [
        calculate_health_index(probability)
        for probability in probabilities
    ]

    data["risk_level"] = [
        determine_risk_level(health)
        for health in data["health_index"]
    ]

    data["machine_id"] = [
        f"FS-{index + 1:04d}"
        for index in range(len(data))
    ]

    return data


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="main-title">Command Center</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Fleet-wide machine health and predictive maintenance intelligence'
        '</div>',
        unsafe_allow_html=True,
    )

    try:

        fleet = generate_fleet_analysis()

        total_machines = len(fleet)

        healthy = (fleet["risk_level"] == "Healthy").sum()
        monitor = (fleet["risk_level"] == "Monitor").sum()
        at_risk = (fleet["risk_level"] == "At Risk").sum()
        critical = (fleet["risk_level"] == "Critical").sum()

        average_health = fleet["health_index"].mean()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Machines Analyzed",
                f"{total_machines:,}",
            )

        with col2:
            st.metric(
                "Average Health",
                f"{average_health:.1f}/100",
            )

        with col3:
            st.metric(
                "At Risk",
                f"{at_risk + critical:,}",
            )

        with col4:
            st.metric(
                "Critical",
                f"{critical:,}",
            )

        st.markdown(
            '<div class="section-title">Fleet Risk Distribution</div>',
            unsafe_allow_html=True,
        )

        risk_data = pd.DataFrame(
            {
                "Risk Level": [
                    "Healthy",
                    "Monitor",
                    "At Risk",
                    "Critical",
                ],
                "Machines": [
                    healthy,
                    monitor,
                    at_risk,
                    critical,
                ],
            }
        )

        col1, col2 = st.columns([1.5, 1])

        with col1:

            fig = px.bar(
                risk_data,
                x="Risk Level",
                y="Machines",
                text="Machines",
            )

            fig.update_layout(
                template="plotly_dark",
                height=400,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="#080B0D",
                plot_bgcolor="#080B0D",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with col2:

            fig = px.pie(
                risk_data,
                names="Risk Level",
                values="Machines",
                hole=0.55,
            )

            fig.update_layout(
                template="plotly_dark",
                height=400,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="#080B0D",
                plot_bgcolor="#080B0D",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.markdown(
            '<div class="section-title">Model Status</div>',
            unsafe_allow_html=True,
        )

        st.info(
            "Random Forest v1.0 is operational. "
            "Predictions are generated using the trained ForgeSight model."
        )

        st.caption(
            "Fleet analysis is based on the UCI AI4I 2020 synthetic dataset "
            "and should not be interpreted as live industrial telemetry."
        )

    except Exception as error:

        st.error(
            f"Unable to load the ForgeSight fleet analysis: {error}"
        )


# ============================================================
# OPERATIONS
# ============================================================

elif page == "Operations":

    st.markdown(
        '<div class="main-title">Operations</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine fleet health overview'
        '</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "This view represents an analyzed dataset fleet, not live factory telemetry."
    )

    try:

        fleet = generate_fleet_analysis()

        display_columns = [
            "machine_id",
            "Type",
            "health_index",
            "failure_probability",
            "risk_level",
        ]

        table = fleet[display_columns].copy()

        table.columns = [
            "Machine",
            "Type",
            "Health Index",
            "Failure Probability",
            "Risk Level",
        ]

        table["Health Index"] = table["Health Index"].round(1)

        table["Failure Probability"] = (
            table["Failure Probability"] * 100
        ).round(2)

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

    except Exception as error:

        st.error(
            f"Unable to load operations data: {error}"
        )


# ============================================================
# MACHINE INTELLIGENCE
# ============================================================

elif page == "Machine Intelligence":

    st.markdown(
        '<div class="main-title">Machine Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Analyze individual machine operating conditions'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Machine Conditions</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"],
        )

        air_temperature = st.number_input(
            "Air Temperature [K]",
            min_value=295.0,
            max_value=305.0,
            value=300.0,
            step=0.1,
        )

        process_temperature = st.number_input(
            "Process Temperature [K]",
            min_value=305.0,
            max_value=315.0,
            value=310.0,
            step=0.1,
        )

    with col2:

        rotational_speed = st.number_input(
            "Rotational Speed [rpm]",
            min_value=1000.0,
            max_value=3000.0,
            value=1500.0,
            step=10.0,
        )

        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.1,
            max_value=80.0,
            value=40.0,
            step=0.1,
        )

    with col3:

        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0.0,
            max_value=300.0,
            value=50.0,
            step=1.0,
        )

    st.divider()

    analyze = st.button(
        "Analyze Machine",
        type="primary",
        use_container_width=True,
    )

    if analyze:

        result = predict_machine_safe(
            machine_type,
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear,
        )

        if not result["success"]:

            st.error("Machine analysis could not be completed.")

            for error in result["errors"]:
                st.warning(error)

        else:

            st.markdown(
                '<div class="section-title">Machine Health</div>',
                unsafe_allow_html=True,
            )

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

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Recommended Action")

                if result["risk_level"] == "Critical":
                    st.error(result["recommendation"])

                elif result["risk_level"] == "At Risk":
                    st.warning(result["recommendation"])

                else:
                    st.info(result["recommendation"])

            with col2:

                st.subheader("Derived Indicators")

                st.write(
                    f"**Temperature Difference:** "
                    f"{result['temperature_difference']} K"
                )

                st.write(
                    f"**Power Proxy:** "
                    f"{result['power_proxy']}"
                )

            st.divider()

            st.subheader("Primary Risk Drivers")

            try:

                importance = pd.read_csv(
                    "models/feature_importance.csv"
                )

                if "feature" in importance.columns and "importance" in importance.columns:

                    importance = importance.sort_values(
                        "importance",
                        ascending=False,
                    ).head(5)

                    for _, row in importance.iterrows():

                        st.write(
                            f"**{row['feature']}** — "
                            f"{row['importance']:.3f}"
                        )

                else:

                    st.caption(
                        "Feature importance data is available but uses an unexpected format."
                    )

            except Exception:

                st.caption(
                    "Feature importance information is unavailable."
                )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.markdown(
        '<div class="main-title">Analytics</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore machine operating conditions and model-derived risk'
        '</div>',
        unsafe_allow_html=True,
    )

    try:

        fleet = generate_fleet_analysis()

        col1, col2 = st.columns(2)

        with col1:

            fig = px.scatter(
                fleet.sample(
                    min(1500, len(fleet)),
                    random_state=42,
                ),
                x="Torque [Nm]",
                y="Rotational speed [rpm]",
                color="risk_level",
                hover_data=[
                    "health_index",
                    "failure_probability",
                ],
                title="Torque vs Rotational Speed",
            )

            fig.update_layout(
                template="plotly_dark",
                height=450,
                paper_bgcolor="#080B0D",
                plot_bgcolor="#080B0D",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with col2:

            fig = px.scatter(
                fleet.sample(
                    min(1500, len(fleet)),
                    random_state=42,
                ),
                x="Tool wear [min]",
                y="failure_probability",
                color="risk_level",
                title="Tool Wear vs Failure Probability",
            )

            fig.update_layout(
                template="plotly_dark",
                height=450,
                paper_bgcolor="#080B0D",
                plot_bgcolor="#080B0D",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.subheader("Health Index Distribution")

        fig = px.histogram(
            fleet,
            x="health_index",
            nbins=30,
            title="Machine Health Distribution",
        )

        fig.update_layout(
            template="plotly_dark",
            height=400,
            paper_bgcolor="#080B0D",
            plot_bgcolor="#080B0D",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    except Exception as error:

        st.error(
            f"Unable to generate analytics: {error}"
        )


# ============================================================
# ALERTS
# ============================================================

elif page == "Alerts":

    st.markdown(
        '<div class="main-title">Alerts</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'High-risk machines identified by the predictive model'
        '</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "These are model-generated alerts from the analyzed dataset. "
        "They are not live production alerts."
    )

    try:

        fleet = generate_fleet_analysis()

        alerts = fleet[
            fleet["risk_level"].isin(
                ["At Risk", "Critical"]
            )
        ].copy()

        alerts = alerts.sort_values(
            "failure_probability",
            ascending=False,
        )

        if alerts.empty:

            st.success(
                "No At Risk or Critical machines were identified."
            )

        else:

            display = alerts[
                [
                    "machine_id",
                    "Type",
                    "health_index",
                    "failure_probability",
                    "risk_level",
                ]
            ].copy()

            display.columns = [
                "Machine",
                "Type",
                "Health Index",
                "Failure Probability",
                "Risk Level",
            ]

            display["Failure Probability"] = (
                display["Failure Probability"] * 100
            ).round(2)

            display["Health Index"] = (
                display["Health Index"].round(1)
            )

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
            )

    except Exception as error:

        st.error(
            f"Unable to generate alerts: {error}"
        )


# ============================================================
# SYSTEM
# ============================================================

elif page == "System":

    st.markdown(
        '<div class="main-title">System</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'ForgeSight model, dataset and methodology'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Model")

        st.write("**Algorithm:** Random Forest")
        st.write("**Version:** 1.0")
        st.write("**Training:** Completed offline")
        st.write("**Inference:** Local trained model")

        st.success("Model status: Operational")

    with col2:

        st.subheader("Dataset")

        st.write(
            "**UCI AI4I 2020 Predictive Maintenance Dataset**"
        )

        st.write(
            "10,000 instances representing synthetic "
            "industrial machine behavior."
        )

        st.info(
            "The dataset is synthetic and does not represent "
            "live factory telemetry."
        )

    st.divider()

    st.subheader("ForgeSight Health Index")

    st.write(
        "Health Index = (1 − failure probability) × 100"
    )

    st.write(
        "80–100 → Healthy"
    )

    st.write(
        "60–79 → Monitor"
    )

    st.write(
        "40–59 → At Risk"
    )

    st.write(
        "0–39 → Critical"
    )

    st.caption(
        "The Health Index is a project-defined indicator "
        "for communicating model-derived risk."
    )

    st.divider()

    st.subheader("Data Science Pipeline")

    st.code(
        """
Machine Data
     ↓
Feature Engineering
     ↓
Preprocessing
     ↓
Random Forest
     ↓
Failure Probability
     ↓
Health Index
     ↓
Risk Classification
     ↓
Maintenance Recommendation
        """,
        language="text",
    )

    st.caption(
        f"{APP_NAME} v{APP_VERSION}"
    )