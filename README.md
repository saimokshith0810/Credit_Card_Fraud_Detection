# Credit_Card_Fraud_Detection 🚨
Machine learning project for detecting fraudulent credit card transactions using Random Forest and Logistic Regression on an imbalanced dataset.

## 📌 Project Overview
This project aims to detect fraudulent credit card transactions using machine learning techniques. Due to extreme class imbalance, special preprocessing and evaluation strategies are used.

---

## 📊 Dataset Details
- Total records: 1,296,675
- Fraud cases: 7,506 (0.58%)
- Highly imbalanced dataset

---

## ⚙️ Preprocessing Steps
- Selected numerical features only
- Handled missing values using median
- Applied Random Undersampling
- Balanced dataset (Fraud = Non-Fraud)
- Train-Test Split (80/20)

---

## 🤖 Models Used

### 1. Random Forest (Primary Model)
- n_estimators = 100
- Handles non-linear relationships
- Best performance in recall

### 2. Logistic Regression (Baseline)
- Linear model
- Used for comparison

---

## 📈 Evaluation Metrics
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🏆 Results

### Random Forest:
- Precision: 91.66%
- Recall: 87.87%
- F1 Score: 89.73%

### Logistic Regression:
- Precision: 93.06%
- Recall: 73.22%
- F1 Score: 81.95%

---

## 📌 Conclusion
Random Forest outperforms Logistic Regression in detecting fraud cases, especially in recall, making it the better choice.

---

## 🚀 How to Run

```bash
pip install pandas scikit-learn
python preprocess_dataset.py
python train_model.py
python train_logistic_regression.py
