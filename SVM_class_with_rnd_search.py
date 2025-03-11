import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, accuracy_score, 
                             roc_curve, auc, classification_report)

def load_data(file_path):
    """Load dataset from CSV, handle NaN values, and return features and labels."""
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["Label"])  # Remove rows where Label is NaN
    df = df.dropna()  # Remove any other rows with NaN values
    X = df.drop(columns=["Label"])  # Features
    y = df["Label"]  # Labels (0 = Control, 1 = Parkinson's)
    return X, y

def hyperparameter_tuning(X, y):
    """Perform hyperparameter tuning for SVM using GridSearchCV."""
    param_grid = {
        'classifier__C': [0.1, 1, 5, 10],
        'classifier__gamma': [0.01, 0.1, 0.5, 1],
    }
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', class_weight='balanced', probability=True))
    ])
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_

def train_and_evaluate_models(X, y, classifiers, n_splits=5):
    """Train and evaluate different classifiers using stratified k-fold cross-validation."""
    results = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for name, model in classifiers.items():
        print(f"Evaluating {name}...")
        balanced_accuracies, confusion_matrices, accuracies, auc_scores = [], [], [], []
        y_real, y_proba = [], []
        
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Store real & predicted probabilities for ROC curve
            y_real.extend(y_test)
            y_proba.extend(y_prob)
            
            # Metrics
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            balanced_accuracies.append(bal_acc)
            cm = confusion_matrix(y_test, y_pred, normalize='true')
            confusion_matrices.append(cm)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            
            # AUC Score
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_scores.append(auc(fpr, tpr))
        
        avg_bal_acc = np.mean(balanced_accuracies)
        avg_cm = np.mean(confusion_matrices, axis=0)
        avg_acc = np.mean(accuracies)
        avg_auc = np.mean(auc_scores)
        
        results[name] = {
            'accuracy': avg_acc,
            'balanced_accuracy': avg_bal_acc,
            'confusion_matrix': avg_cm,
            'auc': avg_auc,
            'roc_data': (y_real, y_proba),
        }
    
    return results

def plot_results(results):
    """Plot confusion matrices and ROC curves for the models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    for name, res in results.items():
        y_real, y_proba = res['roc_data']
        fpr, tpr, _ = roc_curve(y_real, y_proba)
        axes[1].plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.2f})")
    axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()
    
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))  # One subplot per model

    if len(results) == 1:  # Handle case where only one model is tested
         axes = [axes]  

    for ax, (name, res) in zip(axes, results.items()):
        sns.heatmap(res['confusion_matrix'], annot=True, fmt=".2f", cmap="Blues", 
                    xticklabels=["Control", "Parkinson's"],
                    yticklabels=["Control", "Parkinson's"], ax=ax)
        ax.set_title(f"Normalized Confusion Matrix - {name}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//EEICT-2025//extracted_features_csv"
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing {file_name}...")
            X, y = load_data(file_path)
            
            if X.empty or y.empty:
                print(f"Skipping {file_name} due to empty dataset.")
                continue
            
            best_svm, best_params = hyperparameter_tuning(X, y)
            print(f"Best SVM Params: {best_params}")
            
            classifiers = {
                'SVM (RBF)': best_svm,
                'Logistic Regression': Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(max_iter=2000, solver='saga', class_weight='balanced'))
                ])
            }
            
            results = train_and_evaluate_models(X, y, classifiers)
            plot_results(results)
            #explain_model(best_svm, X)
