# Predicting NFL Defense Pass Coverage Based On Presnap Alignment
Using transformer models to predict what pass coverage an NFL defense is in on a particular play based on presnap player tracking data. 

## What it Does
Before the ball is snapped and a play begins, offenses and defenses are in an elaborate cat and mouse game each trying to figure out what the other is about to do. Based on things like personnel (which players are on the field), formation (how the players are lined up), and presnap movement, it is possible to identify tells that can reveal what a defense may be about to do. This project, which is inspired by a 2025 NFL Big Data Bowl submission, uses transformers to predict what coverage an NFL defense is in based on player tracking data recorded from thousands of frames of presnap video, as well as labeled postsnap coverage data. 


## Quick Start
- prerequisites: check that requirements.txt are satisfied
- open the project in google colab to utilize their GPU (more info found in SETUP.md)
- run `run.ipynb` which calls all the necessary data processing, model training, and evaluation scripts

## Video Links

## Evaluation
Evaluation set: Week 8 validation tensors (2740 frames of data)
**Model 1: Man vs Zone Coverage**

Accuracy: 0.8004

|  | Precision | Recall | F1-Score |
|---|---|---|---|
| Zone | 0.80 | 0.94 | 0.87 |
| Man | 0.78 | 0.49 | 0.60 |

AUC-ROC (Man=positive class): 0.8479

**Confusion Matrix** (Rows=Actual, Cols=Predicted)

|  | Predicted Zone | Predicted Man |
|---|---|---|
| **Actual Zone** | 1,778 | 116 |
| **Actual Man** | 431 | 415 |

---

**Model 2: Multiclass Coverage Classifier**

Accuracy: 0.5745

|  | Precision | Recall | F1-Score |
|---|---|---|---|
| Cover-0 | 0.58 | 0.52 | 0.54 |
| Cover-1 | 0.67 | 0.59 | 0.62 |
| Cover-2 | 0.63 | 0.42 | 0.50 |
| Cover-3 | 0.58 | 0.77 | 0.66 |
| Quarters | 0.44 | 0.59 | 0.50 |
| 2-Man | 0.23 | 0.10 | 0.14 |
| Cover-6 | 0.54 | 0.26 | 0.35 |

**Confusion Matrix** (Rows=Actual, Cols=Predicted)

|  | Cover-0 | Cover-1 | Cover-2 | Cover-3 | Quarters | 2-Man | Cover-6 |
|---|---|---|---|---|---|---|---|
| **Cover-0** | 68 | 18 | 2 | 35 | 9 | 0 | 0 |
| **Cover-1** | 26 | 385 | 13 | 178 | 43 | 5 | 5 |
| **Cover-2** | 1 | 28 | 157 | 113 | 44 | 10 | 24 |
| **Cover-3** | 14 | 92 | 13 | 677 | 72 | 2 | 13 |
| **Quarters** | 7 | 21 | 5 | 94 | 208 | 3 | 17 |
| **2-Man** | 0 | 12 | 15 | 16 | 8 | 6 | 2 |
| **Cover-6** | 2 | 21 | 46 | 45 | 92 | 0 | 73 |

**Per-Class Accuracy:**
- Cover-0: 68/132 (51.5%)
- Cover-1: 385/655 (58.8%)
- Cover-2: 157/377 (41.6%)
- Cover-3: 677/883 (76.7%)
- Quarters: 208/355 (58.6%)
- 2-Man: 6/59 (10.2%)
- Cover-6: 73/279 (26.2%)



