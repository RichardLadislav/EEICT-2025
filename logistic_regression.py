import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score

def load_data(file_path):
    """Load dataset from CSV, handle NaN values, and return features and labels."""
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["Label"])  # Remove rows where Label is NaN
    df = df.dropna()  # Remove any other rows with NaN values
    X = df.drop(columns=["Label"])  # Features
    y = df["Label"]  # Labels (0 = Control, 1 = Parkinson's)
    return X, y

def train_and_evaluate_model(X, y, model=None, n_splits=5):
    """Perform stratified k-fold cross-validation and evaluate the model."""
    if model is None:
        model = LogisticRegression(max_iter=1000, penalty='l1',C=0.01, solver='saga', class_weight='balanced')  # Handle imbalanced classes
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    balanced_accuracies = []
    confusion_matrices = []
    feature_weights = []
    accuracies = []  # For regular accuracy
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        balanced_accuracies.append(bal_acc)
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(cm)
        
        # Regular Accuracy (accuracy_score)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        # Store feature weights
        feature_weights.append(model.coef_.flatten())  # Flatten to a 1D array
    
    avg_bal_acc = np.mean(balanced_accuracies)
    avg_cm = np.mean(confusion_matrices, axis=0)
    avg_acc = np.mean(accuracies)  # Mean accuracy
    
    # Sensitivity = Recall for Parkinson's class
    sensitivity = avg_cm[1, 1] / (avg_cm[1, 0] + avg_cm[1, 1])
    # Specificity = Recall for Control class
    specificity = avg_cm[0, 0] / (avg_cm[0, 0] + avg_cm[0, 1])
    
    # Average feature weights
    avg_feature_weights = np.mean(feature_weights, axis=0)
    
    return avg_acc, avg_bal_acc, avg_cm, sensitivity, specificity, avg_feature_weights

# Iterate through CSV files in folder
if __name__ == "__main__":
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//EEICT-2025//extracted_features_csv"  # Change to your folder containing CSVs 
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing {file_name}...")
            X, y = load_data(file_path)
            
            # Skip if dataset is empty after dropping NaN values
            if X.empty or y.empty:
                print(f"Skipping {file_name} due to empty dataset after NaN handling.")
                continue
            
            # Train and evaluate logistic regression
            avg_acc, avg_bal_acc, avg_cm, sensitivity, specificity, avg_feature_weights = train_and_evaluate_model(X, y)
            
            print(f"Results for {file_name}:")
            print(f"Accuracy: {avg_acc:.4f}")  # Regular accuracy
            print(f"Balanced Accuracy: {avg_bal_acc:.4f}")  # Reported for training evaluation
            print("Confusion Matrix:")
            print(avg_cm)
            print(f"Sensitivity: {sensitivity:.4f}")
            print(f"Specificity: {specificity:.4f}")
            print("Feature Weights:")
            for feature, weight in zip(X.columns, avg_feature_weights):
                print(f"{feature}: {weight:.4f}")
            print("------------------------------")
