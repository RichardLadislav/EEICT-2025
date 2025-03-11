import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score

def load_data(file_path):
    """Load dataset from CSV, handle NaN values, and return features and labels."""
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["Label"])  # Remove rows where Label is NaN
    df = df.dropna()  # Remove any other rows with NaN values
    X = df.drop(columns=["Label"])  # Features
    y = df["Label"]  # Labels (0 = Control, 1 = Parkinson's)
    return X, y

def train_and_evaluate_model(X, y, model, model_name, n_splits=5):
    """Perform stratified k-fold cross-validation and evaluate the model."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    balanced_accuracies = []
    confusion_matrices = []
    accuracies = []
    
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
    
    print(f"\n🔹 Results for {model_name}:")
    print(f"  Accuracy: {avg_acc:.4f}")  
    print(f"  Balanced Accuracy: {avg_bal_acc:.4f}")  
    print(f"  Sensitivity: {avg_cm[1, 1]:.4f}")
    print(f"  Specificity: {avg_cm[0, 0]:.4f}")
    print("------------------------------")
    
    return avg_acc, avg_bal_acc, avg_cm, balanced_accuracies

def plot_results(results):
    """Generate visualizations for each model's confusion matrix."""
    
    num_models = len(results)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    
    if num_models == 1:  # Handle single model case
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        sns.heatmap(res['confusion_matrix'], annot=True, fmt=".2f", cmap="Blues", 
                    xticklabels=["Control", "Parkinson's"],
                    yticklabels=["Control", "Parkinson's"], ax=ax)
        ax.set_title(f"Confusion Matrix - {name}")

    plt.tight_layout()
    plt.show()

def explain_model(model, X_train, X_test, model_name):
    """Use SHAP to interpret model predictions."""
    print(f"\n📊 SHAP Explanation for {model_name}...")
    
    if hasattr(model.named_steps['classifier'], "predict_proba"):
        explainer = shap.Explainer(model.predict_proba, X_train)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values[:, 1], X_test)  # Class 1 (Parkinson's)
    else:
        print(f"Skipping SHAP for {model_name}, as it does not support probability outputs.")

# ---------------- MAIN SCRIPT ---------------- #

if __name__ == "__main__":
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//EEICT-2025//extracted_features_csv"
    
    models = {
        "SVM": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(C=5.15, kernel='rbf', class_weight='balanced', gamma=0.27, probability=True))
        ]),
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=2000, solver='saga', class_weight='balanced'))
        ]),
        "Random Forest": Pipeline([
            ('scaler', StandardScaler()),  # Not required for RF, but keeps pipeline consistent
            ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
        ])
    }
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            print(f"\n🚀 Processing {file_name}...\n")
            X, y = load_data(file_path)
            
            # Skip if dataset is empty
            if X.empty or y.empty:
                print(f"⚠️ Skipping {file_name} due to empty dataset.")
                continue
            
            results = {}
            
            for model_name, model in models.items():
                avg_acc, avg_bal_acc, avg_cm, bal_acc_scores = train_and_evaluate_model(X, y, model, model_name)
                results[model_name] = {"confusion_matrix": avg_cm, "bal_acc_scores": bal_acc_scores}

            # Plot results
            plot_results(results)

            # Explain models with SHAP
            for model_name, model in models.items():
                explain_model(model, X, X, model_name)  # Using entire dataset for SHAP visualization
