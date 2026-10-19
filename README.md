# Week 6: Integrated Project and Comprehensive Reporting

## Project title
**Breast Cancer Diagnosis Classification: An End-to-End Data Science Pipeline**

## Problem statement
Build and evaluate a reproducible baseline classifier that distinguishes benign from malignant cases in the Wisconsin Diagnostic Breast Cancer benchmark dataset. This is for education only and is not a clinical diagnostic tool.

## Dataset
The dataset is bundled with scikit-learn and is based on the Wisconsin Diagnostic Breast Cancer dataset. It contains 569 samples and 30 numeric features computed from digitized images of fine needle aspirates. Target labels are benign and malignant. The CSV snapshot is included in `data/breast_cancer_diagnostic.csv`.

Dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

## Pipeline
1. Load data and validate target labels.
2. Check duplicates, missing values and class distribution.
3. Explore the target balance and the distribution of mean radius.
4. Split into training (80%) and testing (20%) sets using stratification and `random_state=42`.
5. Impute missing numeric values using training-set medians and standardize features in a scikit-learn Pipeline.
6. Train logistic regression.
7. Evaluate using accuracy, malignant-class precision, malignant-class recall, F1 and ROC-AUC.
8. Export predictions, metrics and visualizations.

## Test-set metrics
These values are generated from the fixed 80/20 stratified split. Re-run `src/analysis.py` to reproduce:
- Accuracy: 0.9649
- Precision (malignant): 0.9750
- Recall (malignant): 0.9286
- F1 (malignant): 0.9512
- ROC-AUC: 0.9960

## Key files
- `report.docx`: full project report.
- `data/breast_cancer_diagnostic.csv`: dataset snapshot.
- `src/analysis.py`: reproducible pipeline.
- `notebooks/week6_integrated_project.ipynb`: notebook with analysis and explanations.
- `visualizations/`: EDA and evaluation plots.
- `outputs/model_metrics.json`: metrics.
- `outputs/test_predictions.csv`: held-out test predictions.

## Run locally
Python 3.10+ recommended.
```bash
python -m pip install pandas matplotlib scikit-learn jupyter nbformat
python src/analysis.py
```
To open the notebook:
```bash
jupyter notebook notebooks/week6_integrated_project.ipynb
```

## Limitations and responsible use
This is a small, curated benchmark and may not reflect real clinical populations or workflows. A test split gives one estimate and does not establish clinical validity. Accuracy is not the only important metric; false negatives and false positives have different consequences. External validation, cross-validation, calibration, fairness analysis and clinician oversight would be necessary before any real-world use. Never use this educational model to make medical decisions.

## References
- scikit-learn developers. `load_breast_cancer` dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- Wolberg, W. H., Street, W. N., & Mangasarian, O. L. Wisconsin Diagnostic Breast Cancer dataset, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
