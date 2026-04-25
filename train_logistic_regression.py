from preprocess_dataset import preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def train_and_evaluate_logistic():
    # 1. Get the preprocessed data
    print("Fetching preprocessed data...")
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # 2. Train the Logistic Regression model
    print("\nTraining Logistic Regression model...")
    # max_iter is increased to ensure the algorithm converges on our dataset properly
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # 3. Predict on the test set
    print("Making predictions on the test set...")
    y_pred = lr_model.predict(X_test)
    
    # 4. Evaluate and print results clearly
    print("\n==================================================")
    print("LOGISTIC REGRESSION EVALUATION RESULTS")
    print("==================================================")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n[Confusion Matrix]")
    print(f"True Non-Fraud (Correctly marked harmless): {cm[0][0]:>5}")
    print(f"False Fraud    (Mistaken as fraud):         {cm[0][1]:>5}")
    print(f"False Non-Fraud(Missed fraud cases!):       {cm[1][0]:>5}")
    print(f"True Fraud     (Correctly caught fraud!):   {cm[1][1]:>5}")
    
    # Advanced Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n[Performance Metrics]")
    print(f"Precision: {precision * 100:.2f}%  <- When it flags fraud, how often is it actually fraud?")
    print(f"Recall:    {recall * 100:.2f}%  <- Out of all real fraud cases, how many did it catch?")
    print(f"F1 Score:  {f1 * 100:.2f}%  <- Harmonic mean of Precision & Recall")

if __name__ == "__main__":
    train_and_evaluate_logistic()
