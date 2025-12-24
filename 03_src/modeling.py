from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np


def fit_linear_model(X_train, y_train):
    """Fit simple linear regression."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def fit_ridge(X_train, y_train, alphas=[0.001, 0.01, 0.1, 1, 10, 100]):
    """Fit Ridge regression with cross-validation."""
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_train, y_train)
    return model


def fit_lasso(X_train, y_train, alphas=[0.0001, 0.001, 0.01, 0.1, 1]):
    """Fit Lasso regression with cross-validation."""
    model = LassoCV(alphas=alphas, cv=5, max_iter=10000)
    model.fit(X_train, y_train)
    return model


def create_polynomial_features(X, degree=2):
    """Add polynomial features (squared terms)."""
    X_poly = np.column_stack([X, X[:, 0]**degree, X[:, 2]**degree])
    return X_poly


def create_lagged_features(df, cols, lag=1):
    """Create lagged features for time series."""
    df_lags = df.copy()
    for col in cols:
        df_lags[f'{col}_lag{lag}'] = df_lags[col].shift(lag)
    return df_lags.dropna()


def fit_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """Fit Random Forest regressor."""
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def fit_gradient_boosting(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
    """Fit Gradient Boosting regressor."""
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model
