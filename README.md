# 🏠 PriceMyHouse
==============

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)
![PyCharm](https://img.shields.io/badge/IDE-PyCharm-green)

**PriceMyHouse** è un progetto di **Machine Learning** dedicato alla **predizione dei prezzi delle case**, sviluppato come lavoro universitario per il corso di **Machine Learning (A.A. 2024/2025)**.

Il progetto affronta un problema di **regressione**, applicando l’intera pipeline di Machine Learning:  
analisi esplorativa, preprocessing dei dati, feature engineering, addestramento dei modelli e valutazione delle performance.

---

## 📌 Obiettivo del progetto

L’obiettivo del progetto è prevedere il **prezzo di vendita di una casa** a partire da un insieme eterogeneo di feature strutturali, qualitative e quantitative, utilizzando tecniche di **Machine Learning supervisionato**.

---

## 📊 Dataset

- **Nome:** House Prices – Advanced Regression Techniques  
- **Fonte:** Kaggle  
- **Tipologia:** Dataset reale ad alta dimensionalità  

Il dataset presenta diverse problematiche tipiche dei dati reali, tra cui:
- valori mancanti
- variabili categoriche
- feature ridondanti
- presenza di outlier

Queste caratteristiche lo rendono particolarmente adatto allo studio di **feature engineering** e modelli di regressione.

---

## 🧠 Modelli utilizzati

Nel progetto sono stati sperimentati e confrontati diversi modelli di regressione, tra cui:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor (opzionale)

La selezione finale del modello è basata sulle performance ottenute sul test set.

---

## ⚙️ Tecnologie utilizzate

- **Python**
- **PyCharm**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**

---

## 📁 Struttura del progetto

```text
PriceMyHouse/
├── data/
│   ├── raw/                # dataset originale
│   └── processed/          # dataset preprocessato
├── notebook/               # analisi esplorativa e sperimentazioni
├── preprocessing/          # pulizia dati e feature engineering
├── models/                 # training e valutazione modelli
│   └── artifacts/          # modelli salvati (opzionale)
├── utils/                  # funzioni di supporto
├── requirements.txt
├── run_preprocess.sh       # runner per il preprocessing con venv
├── run_train.sh            # runner per il training con venv
└── main.py                 # entry point del progetto
```
---

## 🚀 Avvio locale

1. Clona la repository:
   ```bash
   git clone https://github.com/ludpotesta/PriceMyHouse.git
   ```

2. Apri il progetto con PyCharm

3. Crea un ambiente virtuale e installa le dipendenze:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. Scarica il dataset Kaggle `train.csv` e posizionalo in `data/raw/train.csv`.

5. Preprocessing (genera `data/processed/train_processed.csv`):
   ```bash
   ./run_preprocess.sh
   ```

6. Training modelli:
   ```bash
   ./run_train.sh
   ```

   In alternativa:
   ```bash
   python models/train_pipeline.py
   ```

   Il miglior modello viene salvato automaticamente in `models/artifacts/`.

### Nota su XGBoost (macOS)
Se vuoi usare XGBoost su macOS, serve anche la libreria OpenMP:
```bash
brew install libomp
```

---

## 📈 Valutazione delle performance

I modelli sono valutati utilizzando le seguenti metriche di regressione:
	•	Root Mean Squared Error (RMSE)
	•	R² Score

I risultati sono presentati tramite analisi numeriche e visualizzazioni grafiche.

---

## 👥 Team
- Luigi Potestà [github.com/ludpotesta](https://github.com/ludpotesta)
- Giulia Buonafine [github.com/giub29](https://github.com/giub29)

---

## 🎓 Contesto accademico

📘 Progetto universitario per il corso di Machine Learning
🏫 Università degli Studi di Salerno
📅 Anno Accademico 2024/2025
