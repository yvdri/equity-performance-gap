import numpy as np
from sklearn import metrics


def calculate_adjusted_r2(r2, n, p):
    """Calculate adjusted R² that penalizes model complexity."""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def calculate_aic(n, mse, p):
    """Calculate AIC. Lower is better."""
    return n * np.log(mse) + 2 * p


def calculate_bic(n, mse, p):
    """Calculate BIC. Lower is better."""
    return n * np.log(mse) + p * np.log(n)


def evaluate_model(y_true, y_pred):
    """Calculate R², MSE, RMSE, MAE."""
    return {
        'R²': metrics.r2_score(y_true, y_pred),
        'MSE': metrics.mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(metrics.mean_squared_error(y_true, y_pred)),
        'MAE': metrics.mean_absolute_error(y_true, y_pred)
    }


def calculate_overfitting_gap(r2_train, r2_test):
    """Calculate gap between train and test R²."""
    if isinstance(r2_train, (list, np.ndarray)):
        return [r2_train[i] - r2_test[i] for i in range(len(r2_train))]
    return r2_train - r2_test
