import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, params=None):
    model = RandomForestRegressor(**(params or {}))
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, params=None):
    model = GradientBoostingRegressor(**(params or {}))
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, params=None):
    default_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror"
    }

    if params:
        default_params.update(params)

    model = xgb.XGBRegressor(**default_params)
    model.fit(X_train, y_train)
    return model