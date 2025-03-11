import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
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
    """Perform stratified k-fold cross-validation using an SVM pipeline with feature scaling."""
    if model is None:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='linear', class_weight='balanced', probability=True))  # Linear SVM for interpretability
        ])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    balanced_accuracies = []
    confusion_matrices = []
    accuracies = []  # Regular accuracy
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        balanced_accuracies.append(bal_acc)
        cm = confusion_matrix(y_test, y_pred, normalize='true')  # Normalized Confusion Matrix
        confusion_matrices.append(cm)
        
        # Regular Accuracy
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    
    avg_bal_acc = np.mean(balanced_accuracies)
    avg_cm = np.mean(confusion_matrices, axis=0)  # Average confusion matrix
    avg_acc = np.mean(accuracies)  # Mean accuracy
    
    # Sensitivity = Recall for Parkinson's class
    sensitivity = avg_cm[1, 1]
    # Specificity = Recall for Control class
    specificity = avg_cm[0, 0]
    
    return avg_acc, avg_bal_acc, avg_cm, sensitivity, specificity, balanced_accuracies

def plot_results(conf_matrix, bal_acc_scores, file_name):
    """Generate visualizations for normalized confusion matrix and balanced accuracy scores."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion Matrix Heatmap (Normalized)
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Control", "Parkinson's"], 
                yticklabels=["Control", "Parkinson's"], ax=axes[0])
    axes[0].set_title("Normalized Confusion Matrix")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    
    # Balanced Accuracy Boxplot
    sns.boxplot(y=bal_acc_scores, ax=axes[1], color="lightcoral")
    axes[1].set_title("Balanced Accuracy Scores (Cross-Validation)")
    axes[1].set_ylabel("Balanced Accuracy")
    
    plt.suptitle(f"Results for {file_name}", fontsize=14)
    plt.tight_layout()
    plt.show()

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
            
            # Train and evaluate SVM model
            avg_acc, avg_bal_acc, avg_cm, sensitivity, specificity, bal_acc_scores = train_and_evaluate_model(X, y)
            
            print(f"Results for {file_name}:")
            print(f"Accuracy: {avg_acc:.4f}")  
            print(f"Balanced Accuracy: {avg_bal_acc:.4f}")  
            print("Normalized Confusion Matrix:")
            print(avg_cm)
            print(f"Sensitivity: {sensitivity:.4f}")
            print(f"Specificity: {specificity:.4f}")
            print("------------------------------")
            
            # Plot the results
            plot_results(avg_cm, bal_acc_scores, file_name)
