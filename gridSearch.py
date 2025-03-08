import os
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np

# Load your data
def load_data(file_path):
    """Load dataset from CSV, handle NaN values, and return features and labels."""
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["Label"])  # Remove rows where Label is NaN
    df = df.dropna()  # Remove any other rows with NaN values
    X = df.drop(columns=["Label"])  # Features
    y = df["Label"]  # Labels (0 = Control, 1 = Parkinson's)
    return X, y

# Set up the parameter grid for GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],  # Regularization type
    'solver': ['liblinear', 'saga'],  # Solvers that support L1 or L2 penalties
    'max_iter': [200, 500, 1000, 2000]  # Maximum iterations for convergence
}

# Perform GridSearchCV
def tune_model(X, y):
    # Scale the features to standardize them
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(class_weight='balanced', random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1)
    grid_search.fit(X_scaled, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best balanced accuracy: {grid_search.best_score_:.4f}")
    
    # Return the best model
    return grid_search.best_estimator_

# Example usage
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
            best_model = tune_model(X, y)
