import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    """Perform stratified k-fold cross-validation using a pipeline with feature scaling."""
    if model is None:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=2000, solver='saga', class_weight='balanced'))
        ])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    balanced_accuracies = []
    confusion_matrices = []
    feature_weights = []
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
        
        # Store feature weights
        feature_weights.append(model.named_steps['classifier'].coef_.flatten())  # Extract weights
    
    avg_bal_acc = np.mean(balanced_accuracies)
    avg_cm = np.mean(confusion_matrices, axis=0)  # Average confusion matrix
    avg_acc = np.mean(accuracies)  # Mean accuracy
    
    # Sensitivity = Recall for Parkinson's class
    sensitivity = avg_cm[1, 1]
    # Specificity = Recall for Control class
    specificity = avg_cm[0, 0]
    
    # Average feature weights
    avg_feature_weights = np.mean(feature_weights, axis=0)
    
    return avg_acc, avg_bal_acc, avg_cm, sensitivity, specificity, avg_feature_weights, balanced_accuracies

def plot_results(conf_matrix, feature_weights, feature_names, bal_acc_scores, file_name):
    """Generate visualizations for normalized confusion matrix, feature weights, and balanced accuracy scores."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Confusion Matrix Heatmap (Normalized)
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Control", "Parkinson's"], 
                yticklabels=["Control", "Parkinson's"], ax=axes[0])
    axes[0].set_title("Normalized Confusion Matrix")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    
    # Feature Weights Bar Chart
    sorted_indices = np.argsort(np.abs(feature_weights))[::-1]  # Sort features by absolute weight
    sorted_features = np.array(feature_names)[sorted_indices]
    sorted_weights = feature_weights[sorted_indices]
    
    axes[1].barh(sorted_features[:10], sorted_weights[:10], color="royalblue")
    axes[1].set_title("Top 10 Feature Weights")
    axes[1].set_xlabel("Weight")
    axes[1].invert_yaxis()
    
    # Balanced Accuracy Boxplot
    sns.boxplot(y=bal_acc_scores, ax=axes[2], color="lightcoral")
    axes[2].set_title("Balanced Accuracy Scores (Cross-Validation)")
    axes[2].set_ylabel("Balanced Accuracy")
    
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
            
            # Train and evaluate logistic regression
            avg_acc, avg_bal_acc, avg_cm, sensitivity, specificity, avg_feature_weights, bal_acc_scores = train_and_evaluate_model(X, y)
            
            print(f"Results for {file_name}:")
            print(f"Accuracy: {avg_acc:.4f}")  
            print(f"Balanced Accuracy: {avg_bal_acc:.4f}")  
            print("Normalized Confusion Matrix:")
            print(avg_cm)
            print(f"Sensitivity: {sensitivity:.4f}")
            print(f"Specificity: {specificity:.4f}")
            print("Feature Weights:")
            for feature, weight in zip(X.columns, avg_feature_weights):
                print(f"{feature}: {weight:.4f}")
            print("------------------------------")
            
            # Plot the results
            plot_results(avg_cm, avg_feature_weights, X.columns, bal_acc_scores, file_name)
