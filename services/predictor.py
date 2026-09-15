import math
from typing import Any

import joblib
import pandas as pd

from services.health_score import (
    calculate_health_index,
    determine_risk_level,
)
from services.recommendations import generate_recommendation
from utils.config import MODEL_PATH, MACHINE_TYPES


# Expected model features
MODEL_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temperature Difference [K]",
    "Power Proxy",
]


_model = None


def load_model():
    """
    Load the trained ForgeSight model once.
    """
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        _model = joblib.load(MODEL_PATH)

    return _model


def validate_machine_input(
    machine_type: Any,
    air_temperature: Any,
    process_temperature: Any,
    rotational_speed: Any,
    torque: Any,
    tool_wear: Any,
) -> list[str]:
    """
    Validate machine input before prediction.
    Returns a list of validation errors.
    """

    errors = []

    # Machine type
    if machine_type not in MACHINE_TYPES:
        errors.append(
            f"Machine Type must be one of: {', '.join(MACHINE_TYPES)}."
        )

    # Numeric conversion helper
    numeric_values = {
        "Air temperature": air_temperature,
        "Process temperature": process_temperature,
        "Rotational speed": rotational_speed,
        "Torque": torque,
        "Tool wear": tool_wear,
    }

    converted = {}

    for name, value in numeric_values.items():
        if value is None or str(value).strip() == "":
            errors.append(f"{name} is required.")
            continue

        try:
            converted[name] = float(value)

            if not math.isfinite(converted[name]):
                errors.append(f"{name} must be a valid finite number.")

        except (TypeError, ValueError):
            errors.append(f"{name} must be a valid number.")

    if errors:
        return errors

    air = converted["Air temperature"]
    process = converted["Process temperature"]
    rpm = converted["Rotational speed"]
    torque_value = converted["Torque"]
    wear = converted["Tool wear"]

    # Dataset-informed validation ranges
    if not 295 <= air <= 305:
        errors.append(
            "Air temperature must be between 295 K and 305 K."
        )

    if not 305 <= process <= 315:
        errors.append(
            "Process temperature must be between 305 K and 315 K."
        )

    if process <= air:
        errors.append(
            "Process temperature must be higher than air temperature."
        )

    if not 1000 <= rpm <= 3000:
        errors.append(
            "Rotational speed must be between 1000 and 3000 rpm."
        )

    if not 0 < torque_value <= 80:
        errors.append(
            "Torque must be greater than 0 and no more than 80 Nm."
        )

    if not 0 <= wear <= 300:
        errors.append(
            "Tool wear must be between 0 and 300 minutes."
        )

    return errors


def predict_machine(
    machine_type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
) -> dict:
    """
    Validate machine data and generate a ForgeSight prediction.
    """

    errors = validate_machine_input(
        machine_type,
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear,
    )

    if errors:
        return {
            "success": False,
            "errors": errors,
        }

    air_temperature = float(air_temperature)
    process_temperature = float(process_temperature)
    rotational_speed = float(rotational_speed)
    torque = float(torque)
    tool_wear = float(tool_wear)

    temperature_difference = (
        process_temperature - air_temperature
    )

    power_proxy = torque * rotational_speed

    machine_data = pd.DataFrame(
        [
            {
                "Type": machine_type,
                "Air temperature [K]": air_temperature,
                "Process temperature [K]": process_temperature,
                "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
                "Temperature Difference [K]": temperature_difference,
                "Power Proxy": power_proxy,
            }
        ]
    )

    try:
        model = load_model()

        failure_probability = float(
            model.predict_proba(machine_data)[0][1]
        )

        prediction = int(
            model.predict(machine_data)[0]
        )

        health_index = calculate_health_index(
            failure_probability
        )

        risk_level = determine_risk_level(
            health_index
        )

        recommendation = generate_recommendation(
            risk_level
        )

        return {
            "success": True,
            "prediction": prediction,
            "failure_probability": round(
                failure_probability * 100, 2
            ),
            "health_index": health_index,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "temperature_difference": round(
                temperature_difference, 2
            ),
            "power_proxy": round(
                power_proxy, 2
            ),
        }

    except Exception as error:
        return {
            "success": False,
            "errors": [
                f"Prediction could not be completed: {error}"
            ],
        }


def predict_machine_safe(
    machine_type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
) -> dict:
    """
    Safe wrapper around the prediction engine.
    Prevents application crashes from unexpected errors.
    """

    try:
        return predict_machine(
            machine_type,
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear,
        )

    except Exception as error:
        return {
            "success": False,
            "errors": [
                f"Unexpected prediction error: {error}"
            ],
        }