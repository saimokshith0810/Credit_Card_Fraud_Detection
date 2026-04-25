# Credit_Card_Fraud_Detection 🚨


## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. Due to the highly imbalanced nature of the dataset, special preprocessing and evaluation strategies are applied to ensure effective fraud detection.

---

## 📊 Dataset Details
- Total Records: **1,296,675**
- Fraud Cases: **7,506 (0.58%)**
- Dataset is **highly imbalanced**

---

## ⚙️ Data Preprocessing
- Selected relevant numerical features
- Handled missing values using **median**
- Applied **Random Undersampling** to balance dataset
- Performed **train-test split (80:20)**

---

## 🤖 Models Used

### 🥇 Random Forest (Primary Model)
- Handles non-linear relationships effectively
- Performs well on large datasets
- Provides balanced performance

### 🥈 Logistic Regression (Baseline Model)
- Simple and interpretable
- Used for comparison

---

## 📈 Evaluation Metrics
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🏆 Results Comparison

| Metric      | Random Forest | Logistic Regression |
|------------|--------------|---------------------|
| Precision  | 91.66%       | 93.06%              |
| Recall     | 87.87%       | 73.22%              |
| F1 Score   | 89.73%       | 81.95%              |

---

### 📌 Key Insight
Although Logistic Regression achieves slightly higher precision, Random Forest significantly outperforms in recall and F1 score.  
Since fraud detection prioritizes identifying fraudulent transactions, **Random Forest is the better model**.

---

## 📊 Confusion Matrix (Random Forest)

- True Positive: **1319**
- False Positive: **120**
- False Negative: **182**
- True Negative: **1382**

---

## 🚀 How to Run the Project

### 1. Install dependencies
```bash
pip install pandas scikit-learn
