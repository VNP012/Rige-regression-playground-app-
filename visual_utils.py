import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from utils.data_utils import get_numeric_data

def _split(df: pd.DataFrame):
    """Helper: split features/target using cleaned numeric data."""
    df = get_numeric_data(df)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def plot_predictions(model: Ridge, df: pd.DataFrame):
    """Scatter plot: predicted vs actual values."""
    X, y = _split(df)
    preds = model.predict(X)

    fig, ax = plt.subplots()
    ax.scatter(y, preds, alpha=0.6)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    return fig

def plot_residuals(model: Ridge, df: pd.DataFrame):
    """Residuals plot."""
    X, y = _split(df)
    preds = model.predict(X)
    residuals = y - preds

    fig, ax = plt.subplots()
    ax.scatter(preds, residuals, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals Plot")
    return fig
