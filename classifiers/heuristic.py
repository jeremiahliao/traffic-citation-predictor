import numpy as np
import pandas as pd
from classifiers.classifier_base import BaseClassifier

class HeuristicClassifier(BaseClassifier):
    """
    Heuristic binary classifier:
    For each Charge, compute P(citation=1). Predict 1 if >= 0.5 else 0.
    """

    def __init__(self, charge_col: str = "Charge"):
        super().__init__(name="HeuristicCitationRate")
        self.charge_col = charge_col
        self.charge_to_rate = {}
        self.global_rate = 0.5  # fallback if unseen charge

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        # Compute per-charge mean
        df = pd.DataFrame({self.charge_col: X_train[self.charge_col], "y": y_train})
        grouped = df.groupby(self.charge_col)["y"].mean()

        # Store mapping
        self.charge_to_rate = grouped.to_dict()

        # Compute fallback rate for unseen charges
        self.global_rate = y_train.mean()

    def _lookup_rate(self, charges: pd.Series) -> np.ndarray:
        # Return rate if exists, else fallback
        return charges.map(self.charge_to_rate).fillna(self.global_rate).values

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Probability is the empirical rate
        return self._lookup_rate(X[self.charge_col])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Label = 1 if rate >= 0.5
        return (self.predict_proba(X) >= 0.5).astype(int)
