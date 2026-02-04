import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def rmse(y_true, y_pred):
    """
    Calcola l'RMSE (Root Mean Squared Error).
    È la metrica principale del progetto PriceMyHouse.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r2(y_true, y_pred):
    """
    Calcola il coefficiente di determinazione R².
    Indica quanto bene il modello spiega la variabilità del target.
    """
    return r2_score(y_true, y_pred)


def evaluate_model(model, X_test, y_test):
    """
    Valuta un modello calcolando RMSE e R².
    Restituisce un dizionario con i risultati.
    """
    y_pred = model.predict(X_test)

    return {
        "RMSE": rmse(y_test, y_pred),
        "R2": r2(y_test, y_pred)
    }


def compare_models(results_dict):
    """
    Confronta più modelli.
    results_dict deve essere un dizionario del tipo:
    {
        "Linear Regression": {"RMSE": ..., "R2": ...},
        "XGBoost": {"RMSE": ..., "R2": ...},
        ...
    }
    Restituisce una tabella ordinata per RMSE crescente.
    """
    import pandas as pd

    df = pd.DataFrame(results_dict).T
    df = df.sort_values(by="RMSE", ascending=True)
    return df