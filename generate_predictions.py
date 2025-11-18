import pandas as pd
from sklearn.model_selection import train_test_split

from classifiers.random_classifier import RandomClassifier
from classifiers.alwayscitation_classifier import AlwaysCitationClassifier
from classifiers.alwayswarning_classifier import AlwaysWarningClassifier
from classifiers.rf_classifier import RFClassifier

from data_cleaning import FEATURE_COLS

INPUT_FILE = "Traffic_Violations.parquet"
OUTPUT_FILE = "Traffic_Violations_test_predictions.parquet"
MODEL_NAME = "Random Forest"   # choose model here
TEST_SIZE = 0.2
RANDOM_STATE = 42


def get_model(name):
    if name == "Random":
        return RandomClassifier()
    elif name == "Always Citation":
        return AlwaysCitationClassifier()
    elif name == "Always Warning":
        return AlwaysWarningClassifier()
    elif name == "Random Forest":
        return RFClassifier()
    else:
        raise ValueError(f"Unknown model name: {name}")


def main():
    print(f"Loading dataset from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)

    X = df[FEATURE_COLS]
    y = (df["Violation Type"] == "Citation").astype(int)

    print(f"Splitting train/test...")
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Loading model: {MODEL_NAME}")
    model = get_model(MODEL_NAME)

    print("Training model on training split...")
    model.fit(X_train, y_train)

    print("Generating predictions on *test split*...")
    y_pred = model.predict(X_test)

    # Add numeric predictions + readable labels
    df_test = df_test.copy()
    df_test["y_true"] = y_test
    df_test["y_pred"] = y_pred
    df_test["Prediction"] = df_test["y_pred"].map({1: "Citation", 0: "Warning"})

    print(f"Saving test predictions to {OUTPUT_FILE}...")
    df_test.to_parquet(OUTPUT_FILE, index=False)

    print("Done. File saved:")
    print(f" -> {OUTPUT_FILE}")
    print(f"Rows in test set: {len(df_test)}")


if __name__ == "__main__":
    main()
