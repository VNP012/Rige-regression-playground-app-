import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from utils.data_utils import get_numeric_data

def _split(df: pd.DataFrame):
    """Helper: split into features/target."""
    df = get_numeric_data(df)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_default(df: pd.DataFrame):
    """Train baseline Ridge regression with default params."""
    df = get_numeric_data(df)
    X_train, X_test, y_train, y_test = _split(df)
    model = Ridge()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "MSE": float(mean_squared_error(y_test, preds)),
        "R2": float(r2_score(y_test, preds)) if len(y_test) > 1 else None
    }
    return metrics, model

def tune_ridge(df: pd.DataFrame):
    """Hypertune Ridge regression using GridSearchCV."""
    df = get_numeric_data(df)
    X_train, X_test, y_train, y_test = _split(df)

    cv_folds = min(5, len(y_train)) if len(y_train) > 1 else 2
    params = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(Ridge(), params, cv=cv_folds, scoring="neg_mean_squared_error")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)
    best_score = mean_squared_error(y_test, preds) if len(y_test) > 0 else None

    return grid.best_params_, float(best_score) if best_score is not None else None, best_model

def cross_validate(df: pd.DataFrame):
    """Run K-fold cross-validation on Ridge regression."""
    df = get_numeric_data(df)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    cv_folds = min(5, len(y)) if len(y) > 1 else 2
    model = Ridge()
    scores = cross_val_score(model, X, y, cv=cv_folds, scoring="neg_mean_squared_error")

    return {
        "CV_MSE_mean": float(np.mean(-scores)),
        "CV_MSE_std": float(np.std(-scores)),
        "CV_folds": cv_folds
    }

def add_predictions(model, df: pd.DataFrame) -> pd.DataFrame:
    """Return dataset with model predictions added as a new column."""
    df_clean = get_numeric_data(df).copy()
    X = df_clean.iloc[:, :-1].values
    preds = model.predict(X)
    df_out = df.copy()
    df_out["prediction"] = preds
    return df_out
