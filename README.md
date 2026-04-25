# Credit_Card_Fraud_Detection
Machine learning project for detecting fraudulent credit card transactions using Random Forest and Logistic Regression on an imbalanced dataset.
# Project Report: Credit Card Fraud Detection

## Introduction
The objective of this project is to build and evaluate machine learning models capable of accurately detecting fraudulent credit card transactions. Fraud detection is notoriously challenging due to massive class imbalance—fraudulent activity is exceedingly rare compared to normal user financial activity. In this project, we processed a raw financial ledger and compared the predictive performance of a baseline Logistic Regression model against an ensemble-based Random Forest model to determine the most effective algorithmic approach.

## Dataset Overview
Our dataset comprised a large-scale transaction ledger tracking over one million records natively. 
* **Total Instances:** 1,296,675 rows, 26 feature columns.
* **Fraud Rate:** 1,289,169 Non-Fraud instances (99.42%) vs. 7,506 Fraud instances (0.58%).
* **Feature Landscape:** Included a varied mix of geographical tracking indicators (mercantile latitude/longitude coordinates), objective financial values (`amt`), user demographics, contextual timestamps, and structural metadata strings.

## Preprocessing
A thorough data cleaning and automated preprocessing pipeline was constructed to ensure strict ML model validity cleanly:
1. **Handling Missing Data:** We detected a ~5% baseline correlation of missing data points across critical numeric columns (`amt`, `city_pop`). These fields were robustly imputed utilizing standard median values instead of damaging overall record scales via droppings.
2. **Feature Pruning:** All text-heavy categorical dimensions (names, job titles, unique IDs) and damaged integrity tracking rows (`merch_zipcode` spanning 15% null drops) were dropped outright. Our model scope was confined to exclusively dense numeric logic structures (`amt`, `lat`, `long`, `unix_time`).
3. **Correcting Class Imbalance:** We utilized analytical **Undersampling**. We gathered the limited minority footprint (7,506 frauds) and aggressively random-sampled an identically equal count out from the vast 1.2M non-fraud grouping. This forced a perfect 50/50 balance footprint—giving models exactly 15,012 perfectly spaced transactions preventing predictive bias.
4. **Train and Test Splits:** The finalized, balanced dataset was securely divided utilizing `scikit-learn` in an 80/20 train-to-test dimension split (stratified by the `is_fraud` label to ensure equal variations cross-boundaries).

## Models Used
1. **Random Forest Classifier**: A highly modular ensemble model perfectly capable of tracing elaborate multidimensional boundaries. Highly effective at bypassing non-linear limitations by factoring evaluations across 100 individual decision trees natively.
2. **Logistic Regression**: A traditional linear-scale baseline classification engine designed to map probabilities through foundational dividing boundaries directly.

## Evaluation Metrics
Since general "Accuracy" metrics falsely inflate success ceilings artificially in heavily imbalanced environments (e.g. guessing safe 100% of the time still nets a 99% accuracy flag), we bypassed it for deep analysis standardizers:
* **Precision:** The false-positive scale detector. When the model ultimately sounds a fraud alarm, how incredibly often is it genuinely accurate?
* **Recall:** The false-negative tracking element. Out of all the truly malicious fraud cases quietly passing the network, how successfully did the framework catch them?
* **F1 Score:** Provides the complex harmonic distribution mean scaling both precision and recall limits interactively.

## Results Comparison

| Metric | Random Forest | Logistic Regression|
| :--- | :--- | :--- |
| **Precision** | 91.66% | **93.06%** |
| **Recall** | **87.87%** | 73.22% | 
| **F1 Score** | **89.73%** | 81.95% |

**Metric Breakdown**: Although Logistic Regression maintained rigidly finer strict precision natively by playing slightly safer with flags, **Random Forest unequivocally swept the core evaluation**. The ensemble engine generated a truly stellar 14% improvement in global system Recall, ultimately pinpointing and blocking an overwhelming percentage of direct fraud completely missed by the regression formula's restricted scope.

## Conclusion
Extreme inherent class imbalance is fundamentally devastating for blind machine learning applications safely passing global data flows. By manually managing constraints via aggressive undersampling protocols organically, we functionally forced our classification algorithms to acknowledge malicious behavior templates instead of exploiting mathematical guessing loops structurally. 

Ultimately, **Random Forest has been determined as the radically superior classification model solution**. Financial fraud systems maintain profoundly volatile, nuanced, and exceptionally complicated scaling bounds. Handling these anomalies reliably requires the branching non-linear problem-solving systems inherently crafted by Random Forest logic. Dynamically tracking variable permutations verifies that the critically catastrophic cost of dropping real security threats (False Negatives) natively collapses—guaranteeing much safer predictive barriers overall.
