from sklearn.model_selection import train_test_split


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Divide il dataset in train e test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)