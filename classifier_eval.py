import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
import plotly.graph_objects as go

from classifiers.alwayscitation_classifier import AlwaysCitationClassifier
from classifiers.alwayswarning_classifier import AlwaysWarningClassifier
from classifiers.random_classifier import RandomClassifier
from classifiers.rf_classifier import RFClassifier
from classifiers.hgb_classifier import HGBClassifier, HGBClassifier2
from classifiers.heuristic import HeuristicClassifier
from data_cleaning import FEATURE_COLS


FILE_PATH = "Traffic_Violations.parquet"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def plot_roc_curves(y_true, scores_dict, out_html="roc_curve.html"):
    fig = go.Figure()
    for name, y_score in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             name="Chance", line=dict(dash="dash")))
    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        legend_title_text=None,
        width=900, height=550
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def plot_pr_curves(y_true, scores_dict, out_html="pr_curve.html"):
    fig = go.Figure()
    pos_rate = (sum(y_true) / len(y_true)) if len(y_true) else 0.0
    for name, y_score in scores_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                                 name=f"{name} (AP={ap:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[pos_rate, pos_rate], mode="lines",
                             name=f"Baseline (pos rate={pos_rate:.3f})",
                             line=dict(dash="dash")))
    fig.update_layout(
        title="Precision–Recall Curve Comparison",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        legend_title_text=None,
        width=900, height=550
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def print_report(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n--- {name} ---")
    print("Confusion Matrix [TN FP; FN TP]:")
    print(cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC:  {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

def plot_confusion_matrices(y_true, pred_dict, out_html="confusion_matrices.html"):
    models = list(pred_dict.keys())
    heatmaps = []
    annotations = []

    for name in models:
        cm = confusion_matrix(y_true, pred_dict[name]).astype(int)
        tn, fp, fn, tp = cm.ravel()

        z = [[tn, fp],
             [fn, tp]]

        annotation_text = [[str(z[i][j]) for j in range(2)] for i in range(2)]

        heatmaps.append(
            go.Heatmap(
                z=z,
                x=["Pred 0 (Warning)", "Pred 1 (Citation)"],
                y=["True 0 (Warning)", "True 1 (Citation)"],
                colorscale="Blues",
                zmin=0,
                zmax=max(tn, fp, fn, tp),
                showscale=False,
                visible=False
            )
        )

        annotations.append([
            dict(
                x=j,
                y=i,
                text=annotation_text[i][j],
                showarrow=False,
                font=dict(color="black", size=16)
            )
            for i in range(2)
            for j in range(2)
        ])

    heatmaps[0].visible = True
    current_ann = annotations[0]

    buttons = []
    for i, name in enumerate(models):
        visibility = [False] * len(models)
        visibility[i] = True
        buttons.append(
            dict(
                label=name,
                method="update",
                args=[
                    {"visible": visibility},
                    {"annotations": annotations[i],
                     "title": f"Confusion Matrix — {name}"}
                ]
            )
        )

    fig = go.Figure(data=heatmaps)
    fig.update_layout(
        title=f"Confusion Matrix — {models[0]}",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        template="plotly_white",
        annotations=current_ann,
        updatemenus=[
            dict(
                type="dropdown",
                x=1.15, y=1.15,
                buttons=buttons,
                showactive=True
            )
        ],
        width=650,
        height=550
    )

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Saved confusion matrices: {out_html}")


def main():
    print("Loading data...")
    df = pd.read_parquet(FILE_PATH)
    top_255 = top_255 = df["Charge"].value_counts().nlargest(255).index
    df = df[df["Charge"].isin(top_255)].copy()

    X = df[FEATURE_COLS]
    y = (df["Violation Type"] == "Citation").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        #"Random": RandomClassifier(),
        "Always Citation": AlwaysCitationClassifier(),
        #"Always Warning": AlwaysWarningClassifier(),
        "Heuristic": HeuristicClassifier(),
        # "Random Forest": RFClassifier(),
        "HGB (all)": HGBClassifier(),
        "HGB (no race/gender)": HGBClassifier2()
    }

    predictions = {}
    probabilities = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        probabilities[name] = model.predict_proba(X_test)

    print(f"\n=== Binary Classifier Comparison (Violation Type = Citation) ===")
    for name in models.keys():
        print_report(name, y_test, predictions[name], probabilities[name])

    plot_roc_curves(y_test, probabilities, out_html="roc_curve.html")
    plot_pr_curves(y_test, probabilities, out_html="pr_curve.html")
    plot_confusion_matrices(y_test, predictions, out_html="confusion_matrices.html")

    print("\nSaved interactive plots:")
    print(" - roc_curve.html")
    print(" - pr_curve.html")


if __name__ == "__main__":
    main()