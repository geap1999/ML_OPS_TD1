# TP02 selection note

## Criterion, fixed before the comparison

Written in `configs/train.json` before any candidate was trained:

- **Primary metric:** validation **average precision (AP)**, computed from
  positive-class (`Churn = Yes`) probabilities.
- **Rule:** highest validation AP among learned candidates. Any candidate
  within **0.01** AP of the best counts as tied, and ties go to the simpler
  model (logistic regression < decision tree < random forest).
- **Data allowed to influence selection:** validation only. The test
  partition is allocated and recorded but has not been evaluated.

Why AP rather than accuracy: churners are about 26.5% of customers. A model
that never predicts churn already gets 73.5% accuracy (the majority baseline
below) while finding zero churners. AP measures how well the model *ranks*
customers by churn risk, across all thresholds, which is what a retention
team needs to prioritise calls. Precision, recall and F1 at threshold 0.5
are reported as supporting evidence.

## Results (validation, 1 409 customers, 374 churners)

| Model | AP | Precision | Recall | F1 | Accuracy | TP | FN | FP | Fit (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| majority baseline | 0.265 | 0.000 | 0.000 | 0.000 | 0.735 | 0 | 374 | 0 | ~0.02 |
| logistic regression | 0.648 | 0.664 | 0.529 | 0.589 | 0.804 | 198 | 176 | 100 | ~0.05 |
| decision tree | 0.617 | 0.636 | 0.602 | 0.618 | 0.803 | 225 | 149 | 129 | ~0.03 |
| random forest | **0.664** | 0.653 | 0.484 | 0.556 | 0.795 | 181 | 193 | 96 | ~0.9 |

Source: `reports/model_comparison.json`. Same split (seed 42), same six
features, same preprocessing recipe (a fresh instance per candidate) and same
metric code for every row. Fit times are descriptive, from one Linux machine.

## Decision

**Selected: `random_forest`.** Its validation AP (0.664) is the highest, and
the closest simpler candidate, logistic regression (0.648), is 0.017 behind.
That is outside the 0.01 tolerance, so the simplicity tie-break does not
apply.

Points to keep in mind about this choice:

- **The margin is small.** 0.017 AP on one validation split of 1 409
  customers may well be within split-to-split noise. The tolerance was fixed
  in advance and the rule is applied as written, but cross-validation (an
  optional extension) would show whether the ranking is stable.
- **Highest AP is not highest accuracy.** Logistic regression has better
  accuracy (0.804 vs 0.795) and recall at 0.5 (0.529 vs 0.484). The forest
  ranks customers better overall, but at the 0.5 threshold it flags fewer
  churners. If the team cares most about catching churners, the threshold
  should be chosen on validation for that goal (optional extension 2), not
  left at 0.5.
- **Cost of the choice:** the forest is about 20× slower to fit and its
  artifact is about 4 MB, against a few KB for logistic regression. It is
  also harder to explain to the business.
- **Not an algorithm comparison with milestone 1.** The two-feature logistic
  regression (`reports/first_model.json`, accuracy 0.782) used a different
  feature set. The gain to 0.804 mixes the effect of the new features with
  everything else, so it does not prove an algorithm improvement.
- **Probabilities are not calibrated.** A forest score of 0.8 ranks a
  customer as high risk; it is not shown to mean an 80% chance of churning.
