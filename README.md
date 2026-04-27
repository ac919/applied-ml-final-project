# Predicting NFL Defense Pass Coverage Based On Presnap Alignment
Using transformer models to predict what pass coverage an NFL defense is in on a particular play based on presnap player tracking data

## What it Does


## Quick Start
- make sure requirements.txt are satisfied
- load/prep the data by running data_loading.py
- train the 2 different classifier models by running man_zone_transformer.py and coverage_classifier_transformer.py
- evaluate the models using man_zone_eval.py & coverage_classifier_eval.py

## Video Links

## Evaluation
**Model 1: Man vs Zone Coverage**

Accuracy: 0.8757

|  | Precision | Recall | F1-Score |
|---|---|---|---|
| Zone | 0.91 | 0.91 | 0.91 |
| Man | 0.81 | 0.79 | 0.80 |

**Confusion Matrix** (Rows=Actual, Cols=Predicted)

|  | Predicted Zone | Predicted Man |
|---|---|---|
| **Actual Zone** | 14,913 | 1,403 |
| **Actual Man** | 1,555 | 5,923 |

---

**Model 2: Multiclass Coverage Classifier**

Accuracy: 0.7232

|  | Precision | Recall | F1-Score |
|---|---|---|---|
| Cover-0 | 0.79 | 0.70 | 0.74 |
| Cover-1 | 0.72 | 0.78 | 0.75 |
| Cover-2 | 0.75 | 0.76 | 0.76 |
| Cover-3 | 0.75 | 0.78 | 0.77 |
| Quarters | 0.66 | 0.58 | 0.62 |
| 2-Man | 0.61 | 0.64 | 0.62 |
| Cover-6 | 0.65 | 0.56 | 0.60 |

**Confusion Matrix** (Rows=Actual, Cols=Predicted)

|  | Cover-0 | Cover-1 | Cover-2 | Cover-3 | Quarters | 2-Man | Cover-6 |
|---|---|---|---|---|---|---|---|
| **Cover-0** | 809 | 187 | 7 | 61 | 83 | 0 | 16 |
| **Cover-1** | 139 | 4,515 | 96 | 758 | 153 | 78 | 48 |
| **Cover-2** | 2 | 134 | 2,533 | 245 | 104 | 66 | 233 |
| **Cover-3** | 31 | 1,137 | 151 | 5,946 | 243 | 14 | 92 |
| **Quarters** | 35 | 113 | 142 | 644 | 1,756 | 21 | 311 |
| **2-Man** | 0 | 61 | 89 | 1 | 3 | 322 | 26 |
| **Cover-6** | 2 | 87 | 371 | 252 | 321 | 30 | 1,327 |

**Per-Class Accuracy:**
- Cover-0: 809/1,163 (69.6%)
- Cover-1: 4,515/5,787 (78.0%)
- Cover-2: 2,533/3,317 (76.4%)
- Cover-3: 5,946/7,614 (78.1%)
- Quarters: 1,756/3,022 (58.1%)
- 2-Man: 322/502 (64.1%)
- Cover-6: 1,327/2,390 (55.5%)



