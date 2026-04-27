# Predicting NFL Defense Pass Coverage Based On Presnap Alignment
Using transformer models to predict what pass coverage an NFL defense is in on a particular play based on presnap player tracking data. 

## What it Does
Before the ball is snapped and a play begins, offenses and defenses are in an elaborate cat and mouse game each trying to figure out what the other is about to do. Based on things like personnel (which players are on the field), formation (how the players are lined up), and presnap movement, it is possible to identify tells that can reveal what a defense may be about to do. This project, which is inspired by a 2025 NFL Big Data Bowl submission, uses transformers to predict what coverage an NFL defense is in based on player tracking data recorded from thousands of frames of presnap video, as well as labeled postsnap coverage data. 


## Quick Start
- prerequisites: check that requirements.txt are satisfied
- use git lfs to pull the data
- run `run.ipynb` which calls all the necessary data processing, model training, and evaluation scripts

## Video Links

## Evaluation
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

Accuracy: 0.5500

|  | Precision | Recall | F1-Score |
|---|---|---|---|
| Cover-0 | 0.68 | 0.38 | 0.49 |
| Cover-1 | 0.60 | 0.64 | 0.62 |
| Cover-2 | 0.57 | 0.40 | 0.47 |
| Cover-3 | 0.59 | 0.71 | 0.65 |
| Quarters | 0.40 | 0.53 | 0.46 |
| 2-Man | 0.43 | 0.17 | 0.24 |
| Cover-6 | 0.41 | 0.21 | 0.28 |

**Confusion Matrix** (Rows=Actual, Cols=Predicted)

|  | Cover-0 | Cover-1 | Cover-2 | Cover-3 | Quarters | 2-Man | Cover-6 |
|---|---|---|---|---|---|---|---|
| **Cover-0** | 50 | 39 | 2 | 21 | 20 | 0 | 0 |
| **Cover-1** | 8 | 420 | 15 | 158 | 37 | 7 | 10 |
| **Cover-2** | 1 | 49 | 150 | 103 | 37 | 1 | 36 |
| **Cover-3** | 7 | 122 | 12 | 630 | 92 | 2 | 18 |
| **Quarters** | 8 | 21 | 8 | 113 | 188 | 0 | 17 |
| **2-Man** | 0 | 28 | 10 | 7 | 0 | 10 | 4 |
| **Cover-6** | 0 | 18 | 65 | 37 | 97 | 3 | 59 |

**Per-Class Accuracy:**
- Cover-0: 50/132 (37.9%)
- Cover-1: 420/655 (64.1%)
- Cover-2: 150/377 (39.8%)
- Cover-3: 630/883 (71.3%)
- Quarters: 188/355 (53.0%)
- 2-Man: 10/59 (16.9%)
- Cover-6: 59/279 (21.1%)



