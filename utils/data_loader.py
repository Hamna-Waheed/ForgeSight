import pandas as pd

from utils.config import PROCESSED_DATA_PATH


def load_processed_data() -> pd.DataFrame:
    """
    Safely load the processed ForgeSight dataset.
    """

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {PROCESSED_DATA_PATH}"
        )

    try:
        return pd.read_csv(PROCESSED_DATA_PATH)

    except Exception as error:
        raise RuntimeError(
            f"Unable to load processed dataset: {error}"
        ) from error