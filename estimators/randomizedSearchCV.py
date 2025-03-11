import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np
from scipy.stats import uniform, loguniform
import joblib  # For saving the model

def load_data(file_path):
    """Load dataset from CSV, handle NaN values, and return features and labels."""
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["Label"])  # Remove rows where Label is NaN
    df = df.dropna()  # Remove any other rows with NaN values
    X = df.drop(columns=["Label"])  # Features
    y = df["Label"]  # Labels (0 = Control, 1 = Parkinson's)
    return X, y

# Set up the parameter distributions for RandomizedSearchCV
param_dist = {
    'C': loguniform(0.01, 100),  # Log-uniform distribution for C
    'penalty': ['l1', 'l2'],  # Regularization type
    'solver': ['liblinear', 'saga'],  # Solvers that support L1 or L2 penalties
    'max_iter': [200, 500, 1000, 2000]  # Maximum iterations for convergence
}

# Perform RandomizedSearchCV
def tune_model_randomized(X, y):
    # Scale the features to standardize them
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(class_weight='balanced', random_state=42)
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, 
                                       n_iter=100, cv=5, scoring='balanced_accuracy', n_jobs=-1, random_state=42)
    random_search.fit(X_scaled, y)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best balanced accuracy: {random_search.best_score_:.4f}")
    
    # Save the best model
    best_model = random_search.best_estimator_
    joblib.dump(best_model, "best_model_randomized.pkl")
    print("Best model saved as 'best_model_randomized.pkl'.")
    
    # Return the best model
    return best_model

if __name__ == "__main__":
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//EEICT-2025//extracted_features_csv"  # Change to your folder containing CSVs
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing {file_name}...")
            
            # Load data only once per file
            X, y = load_data(file_path)
            
            # Skip if dataset is empty after dropping NaN values
            if X.empty or y.empty:
                print(f"Skipping {file_name} due to empty dataset after NaN handling.")
                continue

            # Perform hyperparameter tuning
            best_model_randomized = tune_model_randomized(X, y)
