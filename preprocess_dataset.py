import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(filepath='final_dataset.csv'):
    print("1. Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"   Original shape: {df.shape}")
    
    print("\n2. Selecting useful numerical features & dropping text/noise...")
    # Defining only the necessary continuous numerical features
    numeric_features = [
        'amt', 'lat', 'long', 'city_pop', 'unix_time', 
        'merch_lat', 'merch_long'
    ]
    target = 'is_fraud'
    
    # Filter the dataframe
    df = df[numeric_features + [target]].copy()
    
    print("\n3. Handling missing values...")
    # Fill missing values in numeric fields with the median
    for col in numeric_features:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"   Filled missing values in '{col}' with median: {median_val}")
            
    # Drop any leftover missing targets
    df.dropna(subset=[target], inplace=True)
    
    print("\n4. Handling class imbalance (Undersampling)...")
    fraud_df = df[df[target] == 1]
    non_fraud_df = df[df[target] == 0]
    
    print(f"   Before under-sampling: Fraud={len(fraud_df)}, Non-Fraud={len(non_fraud_df)}")
    
    # Randomly select a sample from the majority class equal to the minority class size
    non_fraud_undersampled = non_fraud_df.sample(n=len(fraud_df), random_state=42)
    
    balanced_df = pd.concat([fraud_df, non_fraud_undersampled]).reset_index(drop=True)
    print(f"   After under-sampling: Fraud={len(fraud_df)}, Non-Fraud={len(non_fraud_undersampled)}")
    
    print("\n5. Separating features (X) and target (y)...")
    X = balanced_df.drop(columns=[target])
    y = balanced_df[target]
    
    print("\n6. Splitting dataset into train and test sets (80% train, 20% test)...")
    # stratify=y ensures target proportions belong exactly evenly across train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"   Training Set: {X_train.shape[0]} samples")
    print(f"   Testing Set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    print("\nPreprocessing completed successfully! Ready for ML.")
