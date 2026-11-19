# TP04 summary: testing and code quality

Everything below was run on 2026-11-19 in the project environment
(Python 3.12.11, Linux under WSL): pytest 9.1.1, pytest-cov 7.1.0, Ruff
0.17.5, pre-commit 4.7.0, all resolved in `uv.lock`. Raw terminal logs are
in [`evidence/tp04/`](../evidence/tp04/); nothing here is edited or
invented. No model score was changed: this TP only adds checks.

## 1. Contract and the defects the checks must detect

Input/output contract (also in the README):

- The six TP02 features are required; `Churn` and `customerID` are never
  model inputs.
- Numbers may be numeric or numeric text; blank text is missing; other
  non-numeric text is an error.
- A missing number is median-imputed and an unseen category is encoded as
  all zeros (`OneHotEncoder(handle_unknown="ignore")`).
- A non-empty batch gets one probability and one class per customer, in
  order. Class `1` is `Churn = Yes`; `classes_` is `[0, 1]`.
- A saved and reloaded pipeline gives the same predictions.

Four realistic defects ("the program runs" would not notice any of them):

1. A required column is renamed or dropped upstream and prediction goes on
   without it.
2. Someone changes the encoder to reject unknown categories, so one new
   contract type makes a whole batch fail in production.
3. The churn probability is read from a fixed column position and silently
   becomes the no-churn probability.
4. The save helper stores only the classifier, so the file loads but cannot
   score raw customer records.

## 2. Tests

46 tests pass in about 6 seconds; the 30 synthetic ones take 0.9 s
([`evidence/tp04/pytest_output.txt`](../evidence/tp04/pytest_output.txt)).

```text
$ uv run pytest -q
..............................................                           [100%]
46 passed in 5.82s
```

Fixture ([`tests/conftest.py`](../tests/conftest.py)): 16 invented customers
with the six columns, 7 churners and 9 non-churners, never read from the
Telco CSV. `sample_pipeline` is fitted with the project's real
`ChurnPipelineBuilder`, the committed preprocessing configuration and a
logistic regression with a fixed seed. Tests go through the project's
inference boundary (`ChurnPredictor`), with no network, no CSV and no
MLflow. Fixtures are function-scoped, so a test that modifies its frame
cannot affect another.

### Test-to-risk mapping

| Test | Brief | Real defect it would catch |
|---|---|---|
| `test_missing_required_feature_is_rejected` (parametrized over the six features), `test_input_contract.py` | A | Prediction continuing without a required column. Asserts `KeyError` and the message naming the column, not a broad `Exception`. |
| `test_id_and_target_do_not_influence_predictions` | A | The target or the ID leaking into the inputs. The labels are flipped and the scores must not move. |
| `test_missing_numeric_and_unseen_category_are_scored`, with `test_unseen_category_was_really_absent_from_training` | B | The permissive policy turned into a rejection, or a crash on `NaN`. The second test proves `"Three year"` is unknown to the fitted encoder, so the first cannot pass by accident. |
| `test_variation_in_one_row_leaves_other_rows_unchanged` | B | A bad value in one customer changing the scores of the others. |
| `test_malformed_numeric_text_is_rejected`, `test_blank_text_is_treated_like_a_missing_number` | B | A parser switched to `errors="coerce"`, which would impute typos as if they were blanks. |
| `test_probability_matrix_is_well_formed`, `test_prediction.py` | C | Wrong shape, non-finite values, values outside `[0, 1]`, rows not summing to 1 (tolerance `1e-9`). |
| `test_positive_class_is_looked_up_in_classes` | C | Reading the churn probability from a fixed column instead of from `classes_`. |
| `test_predictions_follow_the_threshold_rule`, `test_one_output_row_per_customer_in_the_same_order` | C | A changed threshold rule; rows lost, duplicated or reordered. |
| `test_prediction_leaves_learned_statistics_unchanged` | extension | Inference refitting the imputer or scaler on the batch being scored. |
| `test_reload_preserves_predictions`, `test_persistence.py` | D | A joblib round trip changing probabilities (tolerance `1e-12`) or classes. |
| `test_reloaded_object_still_contains_the_preprocessing`, `test_loaded_artifact_reproduces_reference_predictions` | D | Saving only the classifier, in the test or in the project's `ArtifactStore`. |
| `tests/test_selection.py` (6 tests) | coverage follow-up | The selection rule choosing the wrong candidate, see section 5. |

Test D checks serialization inside the test process. TP03's
`scripts/verify_registered.py` covers the separate case of a fresh process.

**Does a successful prediction on an unknown category prove it is
accurate?** No. With `handle_unknown="ignore"` the unknown contract is
encoded as "none of the known contracts", a combination the model never saw
in training. The test proves the batch is accepted and the output is well
formed, not that the score is right. Accuracy on such customers would need
labelled examples of them.

## 3. Regression demonstration

Log: [`evidence/tp04/regression_unknown_category.txt`](../evidence/tp04/regression_unknown_category.txt).

**The defect.** In `telco_churn/pipeline.py` I replaced the configured
policy with a hard-coded rejection, a change someone could plausibly make
"to be strict":

```diff
 OneHotEncoder(
-    handle_unknown=self.preprocessing.unknown_categories,
+    handle_unknown="error",
     sparse_output=False,
 ),
```

**The failure.** Training still works and 21 of the 24 tests then present
still pass, because no training row has an unknown category. Three tests
fail:

```text
FAILED tests/test_input_contract.py::TestSupportedInputVariation::test_missing_numeric_and_unseen_category_are_scored
FAILED tests/test_input_contract.py::TestSupportedInputVariation::test_variation_in_one_row_leaves_other_rows_unchanged
FAILED tests/test_prediction.py::TestInferenceIsStateless::test_prediction_leaves_learned_statistics_unchanged
3 failed, 21 passed in 1.29s
```

The traceback ends inside scikit-learn's encoder, at the call
`sample_predictor.predict(varied)` in the test:

```text
>       result = sample_predictor.predict(varied)
tests/test_input_contract.py:85:
...
E   ValueError: Found unknown categories ['Three year'] in column 0 during transform
```

**The reason.** The contract says an unseen category is scored; with
`handle_unknown="error"` the encoder raises on it, so the whole batch is
refused. The test fails on the behaviour itself, not on an `assert False`
or a changed expected value.

**The fix and the rerun.** `git checkout -- telco_churn/pipeline.py`
restored the implementation, then the full suite passed again
(`40 passed in 6.20s` at that point).

A second regression is recorded in
[`evidence/tp04/regression_save_helper.txt`](../evidence/tp04/regression_save_helper.txt):
making `ArtifactStore.save` dump only the classifier fails
`test_loaded_artifact_reproduces_reference_predictions` with
`ValueError: could not convert string to float: 'Month-to-month'`, because
the classifier alone cannot read unencoded columns. It was restored the
same way.

## 4. Ruff

Scope: `telco_churn scripts tests` (root-package layout). Configuration in
`pyproject.toml`:

```toml
[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I"]
```

`E4` import placement, `E7` statement problems, `E9` syntax and IO errors,
`F` Pyflakes (unused or undefined names), `I` import sorting. The formatter
is a separate command with no rule codes.

Before ([`evidence/tp04/ruff_before.txt`](../evidence/tp04/ruff_before.txt)):
lint already passed, and `ruff format --check` reported one file,
`telco_churn/dataset.py` (two comprehensions split over three lines that
fit on one). I looked at the diff, applied `ruff format`, and reran the
tests. After ([`evidence/tp04/ruff_after.txt`](../evidence/tp04/ruff_after.txt)):

```text
$ uv run ruff check telco_churn scripts tests
All checks passed!
$ uv run ruff format --check telco_churn scripts tests
30 files already formatted
```

No rule is ignored and no source file is excluded.

## 5. Coverage

Log: [`evidence/tp04/coverage_output.txt`](../evidence/tp04/coverage_output.txt).
Branch coverage is on. Final result: **61 %** of `telco_churn`
(941 statements, 355 missed).

| Module | Cover | Comment |
|---|---:|---|
| `artifacts.py`, `prediction.py` | 97 % | |
| `config.py`, `provenance.py` | 95 %, 94 % | |
| `features.py`, `splitting.py`, `pipeline.py` | 91 %, 91 %, 90 % | |
| `reporting.py`, `evaluation.py`, `training.py` | 89 %, 86 %, 86 % | |
| `dataset.py` | 39 % | TP01 download and inspection code |
| `tracking.py`, `registry.py` | 0 % | need an MLflow store; deliberately not started by the tests |
| `verification.py` | 0 % | executed, but in a subprocess that coverage does not measure |

**An uncovered, relevant behaviour, and the test for it.** The first report
showed line 233 of `training.py` as missed: the tie-break branch of
`ModelSelector.select`, "a simpler candidate within 0.01 of the best is
selected instead". It decides which model gets registered, and it had never
run, because on the real data no candidate is within the tolerance (the gap
is 0.0167). I proposed and then wrote `tests/test_selection.py`: hand-made
validation results, no training, asserting that a simpler candidate 0.005
behind wins, that one 0.02 behind does not, that the baseline is never
selected, and that selecting among baselines only raises. `training.py`
went from 83 % to 86 %; the total stayed at 61 %.

**A gap that is still open, with a proposed test.** `evaluation.py` lines
102-104: the fallback to `decision_function` and the `TypeError` for a
model with neither scoring API. Proposed test: build the pipeline around a
`LinearSVC`, assert that `positive_scores` returns `"decision_function"` as
the score type and that those scores are not forced into `[0, 1]`; and
assert `TypeError` for an estimator with no scoring method. This matters
the day someone adds a candidate without `predict_proba`. For
`tracking.py` and `registry.py`, a useful check would be a slower
integration test against a temporary SQLite store in `tmp_path`, kept out
of the fast suite.

**What does an executed line tell you?** Only that the line ran during the
tests without raising. **Can a test execute a line without asserting the
right behaviour?** Yes, in both directions here: `verification.py` is fully
exercised by the fresh-process test but shows 0 %, and conversely a test
could call `predict` and assert nothing, giving 100 % on those lines while
checking nothing. Coverage shows what is not tested; it does not show that
what is covered is correct. **Which code is outside the intended scope?**
Third-party code (scikit-learn, pandas, MLflow in `.venv`), the tests
themselves and the thin `scripts/` entry points. `--cov=telco_churn`
measures only the project's package.

## 6. pre-commit

Configuration: [`.pre-commit-config.yaml`](../.pre-commit-config.yaml).
Pinned revisions: `astral-sh/ruff-pre-commit` **`v0.17.5`** (same Ruff as
the command line), with `ruff-check --fix` before `ruff-format`, limited to
`^(telco_churn|scripts|tests)/`; `pre-commit/pre-commit-hooks`
**`v6.0.0`** for trailing whitespace, end of file, YAML, TOML and large
files. The virtual environment and the MLflow storage are never scanned.

Setup log ([`evidence/tp04/precommit_setup.txt`](../evidence/tp04/precommit_setup.txt)):
`validate-config` passed, `install` wrote `.git/hooks/pre-commit`, and the
first `run --all-files` fixed real issues in `pyproject.toml` (a line with
trailing spaces and an extra blank line at the end). After re-staging,
every hook passed.

### Blocked commit and successful retry

Log: [`evidence/tp04/precommit_blocked_commit.txt`](../evidence/tp04/precommit_blocked_commit.txt).
I added a correct test to `tests/test_selection.py` with careless
formatting and an unused `import os`, staged it and committed:

```text
$ git commit -m "Test that tied candidates are listed simplest first"
ruff check...............................................................Failed
- hook id: ruff-check
- files were modified by this hook

Found 1 error (1 fixed, 0 remaining).

ruff format..............................................................Failed
- hook id: ruff-format
- files were modified by this hook

1 file reformatted
exit code: 1   <- commit refused
```

`HEAD` had not moved. `git status` showed `MM`: my staged version, and the
hooks' fixes unstaged. I reviewed them with `git diff` (the unused import
removed, the test reformatted, no logic change), reran the test, staged
again and committed again, without `--no-verify`:

```text
ruff check...............................................................Passed
ruff format..............................................................Passed
[feat/tp04-tests-quality 1964b3b] Test that tied candidates are listed simplest first
exit code: 0   <- commit accepted
```

Hooks are local: a peer must run `uv run pre-commit install` after
cloning, and `--no-verify` skips them. pytest stays a manual command in
this TP; TP05 will run Ruff and pytest in CI, where they cannot be skipped.

## 7. README

The README has a TP04 section with the setup, test, lint, format, coverage
and hook commands, a table of which command detects which kind of issue,
the contract and the list of test files. The ignored generated files
(`.venv/`, `.pytest_cache/`, `.ruff_cache/`, `.coverage`, `coverage.xml`,
`htmlcov/`) were already in `.gitignore`.

Not done yet: asking a peer to explain which command detects which kind of
issue.

## 8. Quality Gate reflection

SonarQube was not run for this submission (optional extension); this
section is the reflection only.

| | What it is | What it tells me | What it cannot tell me |
|---|---|---|---|
| Tests (pytest) | Code that runs my code and asserts results | A specific behaviour still holds | Anything not asserted; whether the model is good |
| Static analysis (Ruff, SonarQube issues) | Reading the code without running it | Unused names, likely bugs, smells, with a file and a line | Whether the program computes the right thing |
| Coverage | A measurement of which lines the tests executed | Where there is no test at all | Whether the executed lines were checked |
| Quality Gate | A **policy**: thresholds applied to those measurements, giving pass or fail | Whether this analysis meets the rules the team chose | Whether the rules themselves are the right ones |

A code issue and coverage are different things: an issue is a finding at a
location in the code, while coverage is a number produced by running the
tests. SonarQube does not run pytest; it imports the `coverage.xml` that
pytest-cov wrote (`sonar.python.coverage.reportPaths=coverage.xml`).
"Analysis completed" and "gate passed" are also different: an analysis can
finish normally and the gate still fail, for example on coverage of new
code below the threshold. The result depends on the thresholds and on the
definition of "new code", so the same code can pass one project's gate and
fail another's. Making a failed gate block a merge is a further step in CI
and in the repository settings, and a GitHub-hosted runner could not reach
a SonarQube server on my laptop's `localhost`.

**If the tests pass, what can still be wrong?** Everything they do not
assert: behaviour on real data unlike my 16 invented customers, poor or
drifting model quality, uncalibrated probabilities, a wrong business
threshold, slow or memory-hungry code, a dependency with a known
vulnerability, and my own mistaken expectations written into a test.

**If a Quality Gate passes, is the model useful, fair or secure?** No. The
gate looks at code measurements. It knows nothing about validation
metrics, about whether errors fall more on one group of customers, about
data quality or about how the model is used. Those need their own
evidence: validation and test results, an analysis by customer group,
dependency and secret scanning (TP05), and human review.

## Limits of this submission

- Run on Linux under WSL only; on this Windows machine Smart App Control
  blocks the uv environment, so the hook was installed from WSL and commits
  are made there.
- The synthetic fixture checks the contract, not model quality: no test
  asserts an accuracy or AP value on 16 invented rows.
- `tracking.py` and `registry.py` have no automated tests; they are
  verified by the TP03 scripts against a running server.
- No SonarQube analysis and no peer exchange yet.
