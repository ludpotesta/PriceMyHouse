import matplotlib.pyplot as plt


def plot_feature_importance(importance_df, top_n=20):
    """
    Grafico delle feature più importanti.
    importance_df deve contenere due colonne:
    - 'feature'
    - 'importance'
    """
    df = importance_df.sort_values(by="importance", ascending=True).tail(top_n)

    plt.figure(figsize=(8, 10))
    plt.barh(df["feature"], df["importance"])
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()