import joblib
import os


def save_model(model, path):
    """
    Salva un modello addestrato nel percorso indicato.
    Crea automaticamente le cartelle mancanti.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Modello salvato in: {path}")


def load_model(path):
    """
    Carica un modello precedentemente salvato.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Il file del modello non esiste: {path}")

    model = joblib.load(path)
    print(f"Modello caricato da: {path}")
    return model