# Predicting Hospital Readmissions with Machine Learning

## Project Overview
Hospital readmissions within 30 days are a major challenge for healthcare providers, impacting both patient care and operational costs. This project focuses on predicting patient readmissions using machine learning techniques. Using a publicly available healthcare dataset, multiple models were trained, tuned, and compared using advanced preprocessing, class balancing, and hyperparameter optimization.

**Key Contributions:**
- Comprehensive data cleaning and feature engineering  
- Encoding categorical variables and handling missing values  
- Comparison of multiple models: XGBoost, Random Forest, Logistic Regression  
- Implementation of SMOTE oversampling and class weighting to handle imbalanced data  
- Hyperparameter tuning using randomized search and trial-error method  
- Evaluation using accuracy, ROC-AUC, precision, recall, and F1-score  

---

## Dataset
- The dataset used in this project was downloaded from the UCI Machine Learning Repository: [Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008).
- Patient admissions data including demographics, lab results, diagnoses, medications, and hospital information  
- Target variable: `readmitted` (YES / NO within 30 days)  

---

## Models and Methods
| Model | Method | Best Hyperparameters | Accuracy | ROC-AUC |
|-------|--------|-------------------|---------|---------|
| XGBoost | Class Weight | `{'n_estimators': 500, 'max_depth': 7, 'learning_rate': 0.01}` | 0.642 | 0.697 |
| XGBoost | SMOTE | `{'n_estimators': 500, 'max_depth': 7, 'learning_rate': 0.01}` | 0.630 | 0.695 |
| Random Forest | Class Weight | `{'n_estimators': 400, 'max_depth': 15, 'min_samples_leaf': 2}` | 0.637 | 0.688 |
| Random Forest | SMOTE | `{'n_estimators': 400, 'max_depth': 15, 'min_samples_leaf': 2}` | 0.636 | 0.687 |
| Logistic Regression | Class Weight | `{'C': 1, 'penalty': 'l2'}` | 0.621 | 0.657 |
| Logistic Regression | SMOTE | `{'C': 1, 'penalty': 'l2'}` | 0.619 | 0.654 |

> Detailed metrics for precision, recall, and F1-score for both classes are available in `results.csv`.

---

## Visualizations

### 1. ROC Curves
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

models = [best_xgb_model, best_rf_model, best_lr_model]
model_names = ["XGBoost", "Random Forest", "Logistic Regression"]

plt.figure(figsize=(8,6))
for model, name in zip(models, model_names):
    y_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Best Models')
plt.legend()
plt.show()
````

### 2. Feature Importance (XGBoost)

```python
import seaborn as sns
import pandas as pd

feat_importance = pd.Series(best_xgb_model.feature_importances_, index=X_train.columns)
feat_importance.nlargest(15).plot(kind='barh', figsize=(8,6))
plt.title("Top 15 Features - XGBoost")
plt.show()
```

### 3. Confusion Matrix (XGBoost)

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, best_xgb_model.predict(X_test))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

---

## Project Impact

* Enables hospitals to **identify high-risk patients** early
* Helps in **resource allocation** and proactive interventions
* Demonstrates **ML best practices** for imbalanced healthcare datasets

---

## Repository Contents

* `notebook.ipynb` – Full analysis with preprocessing, model training, tuning, and evaluation
* `results.csv` – Final results of all model trials with metrics
* `README.md` – Project overview and instructions

---

## How to Run

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run `notebook.ipynb` to reproduce results and visualizations

---

**Author:** Aditya Subramanian
**LinkedIn:** \https://www.linkedin.com/in/aditya-subramanian07/
**Contact:** \aditya.subramanian7@gmail.com
