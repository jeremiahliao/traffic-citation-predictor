from classifiers.classifier_base import BaseClassifier
import pandas as pd
import numpy as np

class RandomClassifier(BaseClassifier):
    def __init__(self):
        super().__init__("random")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return [1 if prob > 0.5 else 0 for prob in probs]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # create arrat of length of X with random values
        return np.random.rand(X.shape[0])