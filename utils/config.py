from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "ai4i2020.csv"
PROCESSED_DATA_PATH = DATA_DIR / "ai4i2020_processed.csv"

# Model paths
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "forgeSight_machine_failure_model.pkl"

# Application settings
APP_NAME = "ForgeSight"
APP_VERSION = "1.0.0"

# Health Index thresholds
HEALTHY_MIN = 80
MONITOR_MIN = 60
AT_RISK_MIN = 40
CRITICAL_MIN = 0

# Supported machine types
MACHINE_TYPES = ["L", "M", "H"]