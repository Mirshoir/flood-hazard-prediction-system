from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_test_split_data(X, y, test_size=0.3, random_state=42):
    """
    Perform train-test split.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(model_type, X_train, y_train, X_test):
    """
    Train model based on selected type, return model and predictions.
    """
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not implemented yet.")

    return model, y_train_pred, y_test_pred
