# Detecting Fraudulent Ethereum Accounts Using Machine Learning and the KDD Process

A machine-learning project that analyzes account-level Ethereum
transaction behaviour to classify accounts as **fraudulent** or
**legitimate**. The project follows the Knowledge Discovery in Databases
(KDD) process and includes model evaluation, SHAP-based interpretation,
and a new-account prediction workflow.

> **Important:** The provided CSV files for new-account prediction
> contain synthetic demonstration records. Their predictions are
> intended to test the notebook workflow and are not verified real-world
> fraud findings.

## Table of Contents

-   [Project Overview](#project-overview)
-   [Objectives](#objectives)
-   [KDD Methodology](#kdd-methodology)
-   [Dataset](#dataset)
-   [Technology Stack](#technology-stack)
-   [Machine-Learning Workflow](#machine-learning-workflow)
-   [Model Performance](#model-performance)
-   [Explainability](#explainability)
-   [New Account Prediction](#new-account-prediction)
-   [Project Structure](#project-structure)
-   [Getting Started](#getting-started)
-   [Running the Notebook](#running-the-notebook)
-   [Interpreting Predictions](#interpreting-predictions)
-   [Limitations](#limitations)
-   [Future Improvements](#future-improvements)

## Project Overview

Ethereum's public blockchain contains transaction activity that can be
summarized into account-level behavioural features. This project
investigates whether machine-learning models can use those features to
distinguish fraudulent accounts from legitimate ones.

The workflow covers data selection, exploration, cleaning,
transformation, model development, evaluation, model interpretation, and
prediction on new account records.

**Project type:** Supervised binary classification\
**Target column:** `FLAG`\
**Target meaning:** `0` = Legitimate, `1` = Fraudulent\
**Selected model:** Tuned XGBoost classifier\
**Classification threshold:** `0.60`

## Objectives

-   Explore account-level Ethereum transaction behaviour.
-   Clean and prepare the dataset for supervised learning.
-   Apply suitable encoding and numerical transformations.
-   Compare multiple classification models.
-   Evaluate performance using metrics suitable for an imbalanced
    classification task.
-   Interpret important model features using SHAP.
-   Provide a reusable workflow for predicting new account records from
    a CSV file.

## KDD Methodology

The project follows these stages of the Knowledge Discovery in Databases
(KDD) process:

1.  **Data Selection** --- select the Ethereum account dataset and
    relevant predictor features.
2.  **Data Exploration** --- examine class distribution, feature
    distributions, missing values, duplicates, and relationships between
    features.
3.  **Data Cleaning** --- address duplicate or conflicting records,
    remove identifier columns, and remove constant features where
    appropriate.
4.  **Data Transformation** --- encode categorical token features, apply
    signed-log transformations to selected skewed variables, filter
    highly correlated features, and cap selected outliers using
    training-derived limits.
5.  **Data Mining** --- train and compare supervised classification
    models.
6.  **Evaluation** --- evaluate the selected model on the held-out test
    set and choose a classification threshold.
7.  **Interpretation** --- use SHAP to inspect features associated with
    model predictions.
8.  **Prediction** --- apply the trained model and matching
    preprocessing workflow to new account records.

## Dataset

The project dataset contains aggregated account-level Ethereum
transaction features, including:

-   Transaction counts and timing
-   Ether sent, received, and balance-related values
-   Contract creation and contract transaction activity
-   Unique sent-to and received-from addresses
-   ERC-20 transaction counts and token activity
-   ERC-20 sent and received values
-   Most frequently sent and received token types

The project analysis records the original dataset as **9,841 rows and 51
columns**. After cleaning, the modelling data contains **9,816 rows and
41 columns**, with the target handled separately from the predictor
features.

An 80/20 stratified split with `random_state=42` was used:

  Split        Accounts   Fraudulent accounts
  ---------- ---------- ---------------------
  Training        7,852                 1,743
  Testing         1,964                   436

The modelling input uses 40 original predictor features before
transformation. The final transformed feature matrix contains 50
features after categorical encoding and preprocessing.

**Dataset note:** Include the original dataset in your repository only
if its licence and distribution terms permit it. Otherwise, document
where it can be obtained and how to place it in the expected project
location.

## Technology Stack

-   **Language:** Python
-   **Notebook environment:** Jupyter Notebook / VS Code
-   **Data manipulation:** pandas, NumPy
-   **Machine learning:** scikit-learn, XGBoost, imbalanced-learn (where
    used by the notebook)
-   **Model interpretation:** SHAP
-   **Visualisation:** Matplotlib and the plotting libraries used in the
    notebook

Use the notebook's actual import cells as the source of truth for the
final dependency list. If the project has a `requirements.txt`, install
from that file.

## Machine-Learning Workflow

The notebook compares these candidate models:

-   Logistic Regression
-   Decision Tree
-   Random Forest
-   XGBoost

Model selection and tuning focus on precision-recall performance, which
is useful when fraudulent accounts form a minority class. The selected
model is a tuned XGBoost classifier.

The preprocessing workflow reuses training-derived settings when
preparing new records, including categorical handling, one-hot encoding,
signed-log transformation, correlation-based feature removal, outlier
limits, and final feature alignment. This is important because new data
must be transformed consistently with the data used to train the model.

## Model Performance

The notebook's recorded final test-set results for tuned XGBoost are:

  Metric                         Result
  ---------------------------- --------
  Accuracy                       99.24%
  Precision                      99.30%
  Recall                         97.25%
  F1-score                       98.26%
  ROC-AUC                        99.93%
  PR-AUC / Average Precision     99.80%
  Classification threshold         0.60

Recorded confusion matrix:

                        Predicted legitimate   Predicted fraud
  ------------------- ---------------------- -----------------
  Actual legitimate                    1,525                 3
  Actual fraud                            12               424

These figures describe performance on the project's held-out test set.
They should not be interpreted as a guarantee of performance on future
accounts, different datasets, or live blockchain data.

## Explainability

SHAP is used to examine how model features contribute to predictions.
The project's analysis identified ERC-20 activity, especially
`Total_ERC20_tnxs`, among the most influential features.

Feature importance describes patterns learned by the model; it does not
establish that a feature causes fraud. Categorical one-hot features
should be interpreted as category indicators rather than continuous
numerical scales.

## New Account Prediction

Section 11 of the notebook provides a new-account prediction workflow.

It:

1.  Loads an external CSV file.
2.  Checks that the required original predictor columns are present.
3.  Rejects input containing the target column `FLAG`.
4.  Applies the existing training-derived preprocessing steps.
5.  Uses the trained XGBoost model to estimate fraud probabilities.
6.  classifies accounts using the threshold of `0.60`.
7.  Displays a prediction summary, charts, and account-level results
    directly in the notebook.

The prediction output includes:

-   `Fraud_Probability` --- estimated probability assigned to the fraud
    class.
-   `Prediction` --- `Fraud` or `Legitimate`, based on the selected
    threshold.

The notebook keeps the complete prediction output in the `results_df`
DataFrame. The current dashboard workflow does not need to create a
separate prediction CSV.

### Demonstration CSV files

The project includes two synthetic CSV files for exercising the
prediction workflow:

-   `ethereum_new_accounts_test_1200.csv` --- 1,200 rows.
-   `ethereum_new_accounts_stress_test_1500.csv` --- 1,500 rows.

These files are **synthetic test inputs**, not actual Ethereum account
records. Their predicted classes do not measure model accuracy. A
reliable accuracy assessment requires independently labelled records
with known outcomes.

## Project Structure

A suggested repository layout is:

``` text
ethereum-fraud-detection/
├── data/
│   ├── ethereum_new_accounts_test_1200.csv
│   └── ethereum_new_accounts_stress_test_1500.csv
├── images/
├── notebooks/
│   └── ethereum_fraud_detection.ipynb
├── README.md
└── requirements.txt
```

Adjust the example to match your actual repository. If your notebook is
in the project root rather than `notebooks/`, show that actual location.
The `images/` folder may remain empty if you are not saving plot files.

## Getting Started

### 1. Clone or download the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ethereum-fraud-detection
```

Replace the placeholder with your repository URL. If you downloaded the
project as a ZIP, extract it and open the project folder.

### 2. Create and activate a virtual environment (recommended)

Windows:

``` bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If a `requirements.txt` file is present:

``` bash
python -m pip install -r requirements.txt
```

If it is not present, install the packages required by the notebook's
import statements. Do not assume a package list without checking those
imports.

### 4. Open the notebook

Open the `.ipynb` file in VS Code with the Python and Jupyter extensions
installed, or launch Jupyter Notebook:

``` bash
jupyter notebook
```

Select the environment in which the dependencies are installed.

## Running the Notebook

1.  Ensure the original dataset is available at the location expected by
    the data-loading cell.
2.  Run the notebook from the first cell in order so that preprocessing
    variables and the trained model are created.
3.  Confirm that the training and evaluation sections complete without
    errors.
4.  In Section 11, select one of the demonstration CSV filenames in the
    CSV-loading cell.
5.  Run the input-loading and validation cells.
6.  Continue only when the schema validation passes.
7.  Run the prediction dashboard cell to view the summary, charts, and
    account-level predictions.

The new-account prediction section relies on variables and helper
functions defined earlier in the notebook. Running Section 11 in a fresh
kernel before running the preceding sections will therefore not work.

## Interpreting Predictions

The threshold of `0.60` means an account is classified as `Fraud` when
its predicted fraud probability is at least 0.60; otherwise it is
classified as `Legitimate`.

A model prediction is a risk estimate, not proof that an account is
fraudulent. Before using the system for operational decisions, validate
it on independent, representative, labelled data and assess the
consequences of false positives and false negatives.

## Limitations

-   Results depend on the selected dataset and its labelling quality.
-   Input features are aggregated account-level summaries rather than
    complete transaction sequences or a live transaction graph.
-   Evaluation results may not generalise to newer accounts, other time
    periods, or changing fraud behaviours.
-   Strong performance on one held-out dataset does not guarantee
    production performance.
-   SHAP identifies model associations, not causation.
-   Synthetic demonstration records are useful for testing the pipeline
    but cannot establish real-world detection accuracy.
-   The current notebook is a research and demonstration workflow, not a
    production monitoring service.

## Future Improvements

-   Validate against newer, independently labelled Ethereum datasets.
-   Use time-based evaluation to test generalisation to future accounts.
-   Investigate group-aware validation where repeated behavioural
    patterns may affect evaluation.
-   Package preprocessing and prediction into a single robust pipeline.
-   Calibrate probability estimates and monitor model drift.
-   Explore temporal transaction features and graph-based approaches.
-   Build an API or web interface for controlled model inference.
-   Add model monitoring, versioning, and a documented retraining
    process.

## Conclusion

This project applies the KDD process to Ethereum account-level
transaction behaviour and evaluates several machine-learning classifiers
for fraud detection. Tuned XGBoost was selected based on the notebook's
evaluation, and SHAP was used to interpret influential features. The
notebook also provides a reusable workflow for predicting new account
records and displaying results.

The next important step for any real-world application is independent
validation on representative, labelled data.

------------------------------------------------------------------------

## Acknowledgement and Reproducibility

Document the original dataset source, its licence, and any citation
required by its provider before publishing the repository. Record the
Python and library versions used for the final run so that the analysis
can be reproduced.
