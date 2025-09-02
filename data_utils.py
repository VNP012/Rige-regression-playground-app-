import pandas as pd

def summarize_data(df: pd.DataFrame) -> dict:
    """Return basic summary stats of dataset (numeric only)."""
    # keep only numeric columns
    df_numeric = df.select_dtypes(include=["number"])

    summary = {
        "shape": df_numeric.shape,
        "columns": list(df_numeric.columns),
        "missing_values": df_numeric.isnull().sum().to_dict(),
        "dtypes": df_numeric.dtypes.astype(str).to_dict(),
        "correlation": df_numeric.corr(numeric_only=True).to_dict()
    }
    return summary

def get_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned numeric-only DataFrame (drops text/IDs)."""
    return df.select_dtypes(include=["number"]).dropna()

def validate_csv(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate input CSV for model training.
    Returns (is_valid, message).
    """
    if df.empty:
        return False, "CSV file is empty."

    # Drop non-numeric cols
    df_numeric = df.select_dtypes(include=["number"])

    if df_numeric.shape[1] < 2:
        return False, "Dataset must have at least 1 feature column and 1 target column."

    if df_numeric.isnull().any().any():
        return False, "Dataset contains missing values. Please clean or impute."

    if len(df_numeric) < 5:
        return False, "Dataset too small. Need at least 5 rows."

    return True, "✅ CSV is valid and ready for training."
