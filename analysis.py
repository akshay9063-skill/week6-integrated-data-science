"""
Week 6 integrated data science pipeline.
Dataset: Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn.
Positive class: malignant. This is an educational benchmark, not a clinical tool.
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, OUT_DIR, VIZ_DIR = ROOT / "data", ROOT / "outputs", ROOT / "visualizations"
OUT_DIR.mkdir(exist_ok=True); VIZ_DIR.mkdir(exist_ok=True)
csv_path = DATA_DIR / "breast_cancer_diagnostic.csv"

if csv_path.exists():
    data = pd.read_csv(csv_path)
else:
    ds = load_breast_cancer(as_frame=True)
    data = ds.frame.copy()
    data["diagnosis"] = data["target"].map({0: "malignant", 1: "benign"})
    data.drop(columns=["target"], inplace=True)
    data.to_csv(csv_path, index=False)

# Basic validation and preparation
data = data.drop_duplicates().copy()
data["diagnosis"] = data["diagnosis"].astype(str).str.strip().str.lower()
data = data[data["diagnosis"].isin(["benign", "malignant"])]
X = data.drop(columns=["diagnosis"])
y = (data["diagnosis"] == "malignant").astype(int)
X = X.apply(pd.to_numeric, errors="coerce")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Pipeline prevents preprocessing leakage: imputation/scaling are fitted only on training data.
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000, random_state=42))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "precision_malignant": precision_score(y_test, pred, zero_division=0),
    "recall_malignant": recall_score(y_test, pred, zero_division=0),
    "f1_malignant": f1_score(y_test, pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, prob),
}
with open(OUT_DIR / "model_metrics.json", "w") as f:
    json.dump({k: round(float(v), 4) for k, v in metrics.items()}, f, indent=2)

predictions = pd.DataFrame({
    "actual_diagnosis": y_test.map({1: "malignant", 0: "benign"}).values,
    "predicted_diagnosis": pd.Series(pred, index=y_test.index).map({1: "malignant", 0: "benign"}).values,
    "predicted_malignant_probability": prob,
}, index=y_test.index)
predictions.to_csv(OUT_DIR / "test_predictions.csv", index_label="row_id")

# Visualizations
data["diagnosis"].value_counts().reindex(["benign", "malignant"]).plot(kind="bar")
plt.title("Diagnosis class distribution"); plt.xlabel("Diagnosis"); plt.ylabel("Samples")
plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(VIZ_DIR / "01_class_distribution.png", dpi=180); plt.close()

for label in ["benign", "malignant"]:
    plt.hist(data.loc[data["diagnosis"] == label, "mean radius"], bins=20, alpha=0.55, label=label)
plt.title("Mean radius by diagnosis"); plt.xlabel("Mean radius"); plt.ylabel("Frequency")
plt.legend(); plt.tight_layout(); plt.savefig(VIZ_DIR / "02_mean_radius_distribution.png", dpi=180); plt.close()

cm = confusion_matrix(y_test, pred, labels=[1, 0])
disp = ConfusionMatrixDisplay(cm, display_labels=["malignant", "benign"])
disp.plot(values_format="d"); plt.title("Confusion matrix (test set)")
plt.tight_layout(); plt.savefig(VIZ_DIR / "03_confusion_matrix.png", dpi=180); plt.close()

RocCurveDisplay.from_predictions(y_test, prob)
plt.title("ROC curve (test set)"); plt.tight_layout()
plt.savefig(VIZ_DIR / "04_roc_curve.png", dpi=180); plt.close()

print("Finished. Test metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
print("See outputs/model_metrics.json, outputs/test_predictions.csv, and visualizations/.")
