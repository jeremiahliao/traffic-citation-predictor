from classifiers.classifier_base import BaseClassifier
import pandas as pd
import numpy as np


class AlwaysCitationClassifier(BaseClassifier):
    def __init__(self):
        super().__init__("always_citation")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # create arrat of length of X with random values
        return np.ones(X.shape[0])