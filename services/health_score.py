def calculate_health_index(failure_probability: float) -> float:
    """
    Convert failure probability into a 0-100 ForgeSight Health Index.
    """
    probability = max(0.0, min(1.0, float(failure_probability)))
    return round((1 - probability) * 100, 1)


def determine_risk_level(health_index: float) -> str:
    """
    Determine machine risk level from the ForgeSight Health Index.
    """
    if health_index >= 80:
        return "Healthy"
    elif health_index >= 60:
        return "Monitor"
    elif health_index >= 40:
        return "At Risk"
    else:
        return "Critical"