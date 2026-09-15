def generate_recommendation(risk_level: str) -> str:
    """
    Generate a maintenance recommendation based on machine risk.
    """

    recommendations = {
        "Healthy": (
            "Machine operating within a healthy range. "
            "Continue normal operation and routine maintenance."
        ),
        "Monitor": (
            "Continue operation with increased monitoring. "
            "Schedule a routine inspection."
        ),
        "At Risk": (
            "Machine shows elevated failure risk. "
            "Inspect critical components and schedule maintenance soon."
        ),
        "Critical": (
            "High failure risk detected. "
            "Inspect the machine immediately and consider controlled shutdown."
        ),
    }

    return recommendations.get(
        risk_level,
        "Unable to determine a maintenance recommendation."
    )