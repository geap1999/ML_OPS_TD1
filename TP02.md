# TP02 - From a Working Model to a Reusable Training Pipeline

**ESILV - Machine Learning Operations**  
**Dataset:** IBM Telco Customer Churn  
**Starting point:** your own TP01 repository, environment and original CSV  
**Format:** independent implementation, with instructor support  
**Duration:** 90-minute core; complete the third model and extensions afterwards if needed

> First make a model work. Then make the whole prediction process reusable by another engineer.

## Your mission

The telecom team wants a first churn model. A single notebook result will not be enough: another engineer must be able to repeat training, compare alternatives and load the selected artifact to predict new customers without rebuilding preprocessing by hand.

Build your solution from your own project. You receive the dataset and requirements, not a starter repository or completed implementation. You may explore in a notebook, but the final training workflow must also run from a terminal.

The progression is deliberate: **working model → preprocessing pipeline → fair model comparison → reusable artifact**. The objective is a reliable ML implementation, not the highest possible leaderboard score.

### How to use the code hints

Each milestone includes a short, optional starting point. **The snippets are incomplete:** replace `...`, add the necessary imports, and connect the variables in your own project before running them. They are not a complete script. You still choose configurations, implement evaluation and checks, and explain your decisions. Try the task first, then use the hint that addresses your blocker.

## 1. Get a first model working

Define the prediction task: `Churn = Yes` is the positive class. Keep `customerID` for identification and split checks; never use it or `Churn` as an input feature.

Create a reproducible stratified train/validation/test split, approximately 60/20/20. Use one seed and retain the same customer allocation throughout the exercise. Reserve the test set; use validation to compare candidates.

For this first milestone, you may use only `tenure` and `MonthlyCharges`. Train one simple classifier and obtain predictions on validation customers. Add a majority-class baseline, for example a `DummyClassifier`, to give the result context.

**Your checkpoint:** a first learned model runs from data to validation predictions. Record at least accuracy and a confusion matrix, with the positive class and matrix labels stated.

### Code hint - split first, then fit

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

train_df, remaining_df = train_test_split(
    df, test_size=0.4, stratify=df["Churn"], random_state=42
)
# TODO: split remaining_df equally into validation and test, also stratified.
valid_df, test_df = ...

first_features = ["tenure", "MonthlyCharges"]
X_train = train_df[first_features]  # Ensure numeric types before fitting.
y_train = train_df["Churn"].map({"No": 0, "Yes": 1})
first_model = LogisticRegression(max_iter=1000)
first_model.fit(..., ...)
# TODO: build X_valid/y_valid and obtain validation predictions.
```

Research `DummyClassifier(strategy="most_frequent")` for the baseline and `confusion_matrix(..., labels=[0, 1])` for an explicitly ordered matrix. Which dataset should the baseline be fitted on?

**Think about it**

- What would a system that always predicts “No churn” achieve?
- Can high accuracy coexist with failing to identify any churners?
- Which data are permitted to influence model selection?

## 2. Connect preprocessing and the model

Expand to this shared feature set for the remaining comparison:

| Numeric inputs | Categorical inputs |
|---|---|
| `tenure` | `Contract` |
| `MonthlyCharges` | `InternetService` |
| `TotalCharges` | `PaymentMethod` |

Convert numeric representations explicitly and normalize whitespace-only `TotalCharges` to missing values. Do not silently turn malformed nonblank text into an allowed missing value. Keep the original CSV unchanged.

Build a scikit-learn `Pipeline` that combines a `ColumnTransformer` with a classifier. Define separate preprocessing branches for numeric and categorical inputs. Include a documented missing-value strategy, appropriate numeric scaling and categorical encoding. Decide what happens when an inference record contains a category absent from training.

Start with a simple, consistent preprocessing recipe. Research `SimpleImputer`, `StandardScaler` or `MinMaxScaler`, and `OneHotEncoder`; these are pointers, not a required implementation template.

All learned preprocessing statistics must come from training data. Validation and new records reuse the fitted transformations. Stateless input parsing may precede the pipeline, provided its rules are documented and consistent at inference.

**Your checkpoint:** one call to fit the complete pipeline trains the preprocessing and classifier; prediction accepts original, unencoded feature columns in the documented input types. You do not separately refit an encoder or scaler for validation.

### Code hint - two preprocessing branches, one prediction object

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# TODO: define numeric_features and categorical_features from the table.
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", ...),
])
categorical_transformer = Pipeline([
    ("imputer", ...),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", ..., ...),
])
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", ...),
])
# TODO: fit on the six original training feature columns, then predict validation.
```

Decide the categorical imputation strategy and the classifier. The initial two-feature `X_train` must now be rebuilt with all six features. Most subsequent estimators accept a transformed numeric array; check sparse/dense compatibility for your choices. For this small exercise, `sparse_output=False` is an available OneHotEncoder option if dense output is needed.

For numeric parsing, first identify truly blank values, then compare them with conversion failures. `pandas.to_numeric` provides `errors="raise"` and `errors="coerce"`; investigate their different behavior before choosing one.

**Think about it**

- What do `fit`, `transform` and `predict` each do?
- What goes wrong if you fit a scaler on validation data?
- Does using a Pipeline automatically prevent leakage if you give it the wrong training data?
- Which operations are ordered by `Pipeline`, and which columns are routed by `ColumnTransformer`?

## 3. Try different models fairly

Compare **at least three learned classifiers**, plus the majority baseline, using the same six input features, same split and same metric definitions. Possible candidates include logistic regression, a decision tree and a random forest; you may choose other suitable classifiers and explain why.

Reuse your pipeline-building logic instead of copying the entire workflow for each model. Create a fresh preprocessing instance for each candidate. Start with modest configurations; large hyperparameter searches are unnecessary.

Before inspecting the comparison, choose a primary selection metric and justify it. A suitable classroom choice is **average precision** computed from positive-class scores. Also report precision, recall and F1 at a fixed threshold such as 0.5; retain accuracy and a confusion matrix as supporting evidence. If a chosen model has no probability method, document its score API and the threshold policy instead of pretending a decision score is a probability.

Record configuration and approximate fit duration as well as validation metrics. Timing is descriptive on your machine, not a rigorous deployment benchmark.

**Your checkpoint:** a machine-readable comparison table and a short selection decision. Do not claim that a change from the initial two-feature model proves an algorithm improvement: the feature set also changed.

### Code hint - swap the estimator, reuse your workflow

```python
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "second_model": ...,  # Research a different classifier family.
    "third_model": ...,
}

results = []
for name, classifier in models.items():
    candidate = build_pipeline(classifier)  # Your function; fresh state each time.
    candidate.fit(X_train, y_train)        # The same six features and split.
    metrics = evaluate_model(candidate, X_valid, y_valid)  # Your function.
    results.append({"model": name, **metrics})
# TODO: include the baseline, save the table and apply your selection rule.
```

For classifiers supporting probabilities, `predict_proba` returns one column per class. Find the column corresponding to class `1` in `classes_` before computing `average_precision_score`. Use continuous scores for average precision; use thresholded class predictions for precision, recall and F1. Neither helper function above is supplied: design and implement it.

**Think about it**

- Would changing the split for every classifier make the comparison easier to interpret?
- Does the model with the highest accuracy necessarily meet the team's needs?
- If two candidates are close, what would justify choosing the simpler one?
- Is a predicted probability of 0.8 automatically a well-calibrated 80% risk?

## 4. Refactor for reuse, then save the whole fitted pipeline

Separate responsibilities into small functions or modules: loading/validation, splitting, constructing an unfitted pipeline, training, evaluation and saving results. Choose your own names and structure.

A pipeline builder should construct and return an unfitted object; it should not secretly read data or train it. Remember that estimators are stateful: fitting changes the estimator. Reusable functions do not make training mathematically pure or free of state.

Save the selected **fitted preprocessing plus classifier together**. In a fresh Python process, reload it and predict a small batch without calling fit. Use only trusted artifacts and record compatible library versions.

Alongside the artifact, retain the source hash, split seed and customer allocations, feature roles, preprocessing settings, model parameters, metrics, positive-class definition, threshold and code/environment identity. A compact JSON record is enough; experiment-tracking infrastructure will come in the next TP.

**Your checkpoint:** loaded and original predictions agree within a stated tolerance on the same examples. The artifact consumes the documented feature schema; CSV parsing and input-type validation remain explicit boundary responsibilities.

### Code hint - save the fitted object, not the recipe alone

```python
import joblib

joblib.dump(selected_pipeline, "pipeline.joblib")

# In a separate Python process, with a compatible environment:
loaded_pipeline = joblib.load("pipeline.joblib")  # Trusted artifact only.
predictions = loaded_pipeline.predict(new_customers)
```

`selected_pipeline` is your already-fitted chosen candidate. `new_customers` is a DataFrame you must construct with the agreed six feature names and input types, without `Churn`. Compare like with like: original versus loaded class predictions, or original versus loaded probability scores. No `fit` call should appear in the inference-only process.

**Think about it**

- Why is saving only the classifier insufficient?
- Which fitted information lives inside an imputer or encoder?
- Could two files called `model.joblib` contain different models?
- How does a sklearn Pipeline differ from a workflow DAG such as Kubeflow Pipelines?

## 5. Demonstrate reliability with a few focused checks

Provide evidence that:

- Customer IDs do not overlap across train, validation and test.
- A learned imputation/scaling statistic matches training data, rather than the full dataset.
- A prediction call does not refit or change that statistic.
- A permitted missing numeric input and a previously unseen category follow your documented policy. Handling unknown categories does not prove predictions are reliable for them.
- A missing required feature produces a visible failure, rather than a silent incorrect prediction.
- Saving/reloading preserves predictions; a separate inference process does not need training labels or training execution.

**Inspection hint:** if you used the step names above, start with `selected_pipeline.named_steps["preprocessor"]`. After fitting, its `named_transformers_["num"]` exposes the numeric branch; that branch's `named_steps["imputer"].statistics_` contains learned fill values. Compare them with the training-only medians for the median strategy, and verify that a prediction call leaves them unchanged. Adapt these names if your design differs.

Exchange projects with a peer. Ask them to reproduce training or load a trusted artifact using only your README, then record one issue and its resolution.

## Deliverables

| Deliverable | Required evidence |
|---|---|
| Runnable training workflow | Documented terminal entry point; source and configuration identified |
| Complete sklearn Pipeline | Numeric/categorical preprocessing linked to the estimator |
| Comparison table | Three learned classifiers plus baseline; same split/features and clear metrics |
| Selection note | Criterion chosen in advance; decision based on validation only |
| Selected fitted artifact and metadata | Accessible artifact, version/environment information and reproduction record |
| Reload/inference evidence | Agreement with original predictions and no refitting |
| Updated README and Git history | A peer can follow the workflow and understand its limitations |

Use an appropriate artifact submission location; do not commit a virtual environment. The original dataset remains immutable. Keep the test set reserved: repeated test evaluation during candidate selection is not accepted.

### Assessment

| Criterion | Points |
|---|---:|
| Working model, baseline and reproducible disjoint splits | 2 |
| Correct preprocessing Pipeline and training-only fitting | 3 |
| Fair comparison and justified model choice | 2 |
| Complete artifact, fresh-process reload and focused checks | 2 |
| Reusable code, clear evidence and peer feedback | 1 |
| **Total** | **10** |

Scores depend on correctness and engineering quality, not on beating a prescribed accuracy target.

## Suggested progression

| Around | Milestone |
|---|---|
| 20 minutes | One simple model and a baseline produce validation predictions |
| 45 minutes | Full preprocessing and model work as a Pipeline |
| 65 minutes | Candidate comparison runs; finish the third learned model afterwards if necessary |
| 80 minutes | Selected pipeline saved and reloaded |
| Final 10 minutes | Focused checks, peer review and remaining-work record |

## Optional extensions - choose one

1. **Cross-validation without leakage:** compare a small number of configurations using folds within training data. Fit the entire pipeline within each fold. Do not turn the test set into a tuning set.
2. **Threshold policy:** select a threshold using validation and a stated cost/recall objective. Explain how this differs from changing the trained classifier; evaluate a frozen policy on test only after selection is finished.
3. **Failure experiment:** drop a required feature or alter a category in a copied inference batch. Describe the difference between schema failure, supported missingness and distribution shift.

Data-job publication protocols, historical backfill and full orchestration are outside this TP's core. The next TP will build on these candidates and artifacts to introduce experiment tracking and model versioning.

## Documentation starting points

- [scikit-learn: Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html)
- [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [Common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
- [Evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Model persistence](https://scikit-learn.org/stable/model_persistence.html)

Choose documentation matching your installed versions. The hints support your implementation; they do not replace understanding it. Be ready to explain the behavior of every preprocessing step and the evidence behind your selection.
