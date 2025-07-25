# Phase 2: Model Development and Validation
# GSE20685 Breast Cancer Gene Expression Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD PREPROCESSED DATA
# =============================================================================

def load_preprocessed_data(file_path='preprocessed_data_phase1.pkl'):
    """Load preprocessed data from Phase 1"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print("Preprocessed data loaded successfully!")
        print(f"Train set: {data['X_train'].shape}")
        print(f"Test set: {data['X_test'].shape}")
        print(f"Target classes: {data['target_names']}")
        return data
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'")
        print("Please run Phase 1 first to generate the preprocessed data")
        return None

# =============================================================================
# 2. HYPERPARAMETER TUNING
# =============================================================================

def tune_random_forest(X_train, y_train, cv_folds=5):
    """Hyperparameter tuning for Random Forest"""
    print("\nTuning Random Forest hyperparameters...")
    
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Use smaller parameter grid for faster execution
    param_grid_small = {
        'n_estimators': [300, 500],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(
        rf, param_grid_small, 
        cv=cv_folds, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best RF parameters: {grid_search.best_params_}")
    print(f"Best RF CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def tune_svm(X_train, y_train, cv_folds=5):
    """Hyperparameter tuning for SVM"""
    print("\nTuning SVM hyperparameters...")
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    
    # Use smaller parameter grid for faster execution
    param_grid_small = {
        'C': [1, 10, 100],
        'gamma': ['scale', 0.01, 0.1],
        'kernel': ['rbf']
    }
    
    svm = SVC(random_state=42, probability=True)
    
    grid_search = GridSearchCV(
        svm, param_grid_small, 
        cv=cv_folds, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best SVM parameters: {grid_search.best_params_}")
    print(f"Best SVM CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# =============================================================================
# 3. MODEL TRAINING AND EVALUATION
# =============================================================================

def train_random_forest(X_train, y_train, X_test, y_test, best_rf=None):
    """Train and evaluate Random Forest classifier"""
    
    print("\n" + "="*40)
    print("RANDOM FOREST MODEL")
    print("="*40)
    
    if best_rf is None:
        # Use default parameters if no tuning was performed
        rf_params = {
            'n_estimators': 500,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        rf_model = RandomForestClassifier(**rf_params)
    else:
        rf_model = best_rf
    
    # Train model
    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)
    
    # Evaluation metrics
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    
    # AUC calculation
    if len(np.unique(y_test)) > 2:
        auc_rf = roc_auc_score(y_test, y_pred_proba_rf, multi_class='ovr', average='macro')
    else:
        auc_rf = roc_auc_score(y_test, y_pred_proba_rf[:, 1])
    
    print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
    print(f"Random Forest AUC: {auc_rf:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"Top 10 important features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"{i+1:2d}. {row['feature']:<15} ({row['importance']:.4f})")
    
    return rf_model, y_pred_rf, y_pred_proba_rf, feature_importance

def train_svm(X_train, y_train, X_test, y_test, best_svm=None):
    """Train and evaluate Support Vector Machine classifier"""
    
    print("\n" + "="*40)
    print("SUPPORT VECTOR MACHINE MODEL")
    print("="*40)
    
    if best_svm is None:
        # Use default parameters if no tuning was performed
        svm_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'random_state': 42,
            'probability': True
        }
        svm_model = SVC(**svm_params)
    else:
        svm_model = best_svm
    
    # Train model
    print("Training SVM...")
    svm_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_svm = svm_model.predict(X_test)
    y_pred_proba_svm = svm_model.predict_proba(X_test)
    
    # Evaluation metrics
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    
    # AUC calculation
    if len(np.unique(y_test)) > 2:
        auc_svm = roc_auc_score(y_test, y_pred_proba_svm, multi_class='ovr', average='macro')
    else:
        auc_svm = roc_auc_score(y_test, y_pred_proba_svm[:, 1])
    
    print(f"SVM Accuracy: {accuracy_svm:.4f}")
    print(f"SVM AUC: {auc_svm:.4f}")
    
    return svm_model, y_pred_svm, y_pred_proba_svm

# =============================================================================
# 4. CROSS-VALIDATION AND MODEL COMPARISON
# =============================================================================

def perform_cross_validation(models_dict, X, y, cv_folds=5):
    """Perform cross-validation for model comparison"""
    
    print("\n" + "="*40)
    print("CROSS-VALIDATION RESULTS")
    print("="*40)
    
    cv_results = {}
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for model_name, model in models_dict.items():
        print(f"\nEvaluating {model_name}...")
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        
        # AUC scores
        if len(np.unique(y)) > 2:
            cv_auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc_ovr', n_jobs=-1)
        else:
            cv_auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
        
        cv_results[model_name] = {
            'accuracy_mean': cv_scores.mean(),
            'accuracy_std': cv_scores.std(),
            'auc_mean': cv_auc_scores.mean(),
            'auc_std': cv_auc_scores.std(),
            'accuracy_scores': cv_scores,
            'auc_scores': cv_auc_scores
        }
        
        print(f"Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"AUC: {cv_auc_scores.mean():.4f} ± {cv_auc_scores.std():.4f}")
    
    return cv_results

# =============================================================================
# 5. COMPREHENSIVE EVALUATION PLOTS
# =============================================================================

def create_comprehensive_evaluation_plots(y_test, models_results, target_names, cv_results):
    """Create comprehensive evaluation plots"""
    
    plt.figure(figsize=(20, 16))
    
    # Extract results
    rf_results = models_results['Random Forest']
    svm_results = models_results['SVM']
    
    y_pred_rf = rf_results['predictions']
    y_pred_svm = svm_results['predictions']
    y_pred_proba_rf = rf_results['probabilities']
    y_pred_proba_svm = svm_results['probabilities']
    
    # 1. Confusion Matrices
    plt.subplot(3, 4, 1)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Random Forest - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.subplot(3, 4, 2)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('SVM - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 3. ROC Curves
    plt.subplot(3, 4, 3)
    if len(target_names) == 2:
        # Binary classification
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf[:, 1])
        fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_proba_svm[:, 1])
        
        auc_rf = roc_auc_score(y_test, y_pred_proba_rf[:, 1])
        auc_svm = roc_auc_score(y_test, y_pred_proba_svm[:, 1])
        
        plt.plot(fpr_rf, tpr_rf, label=f'RF (AUC = {auc_rf:.3f})', color='blue')
        plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.3f})', color='red')
    else:
        # Multiclass - show macro average
        auc_rf = roc_auc_score(y_test, y_pred_proba_rf, multi_class='ovr', average='macro')
        auc_svm = roc_auc_score(y_test, y_pred_proba_svm, multi_class='ovr', average='macro')
        
        plt.text(0.5, 0.5, f'RF Macro AUC: {auc_rf:.3f}\nSVM Macro AUC: {auc_svm:.3f}', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    
    # 4. Feature Importance (Random Forest)
    plt.subplot(3, 4, 4)
    if 'feature_importance' in rf_results:
        top_features = rf_results['feature_importance'].head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'], fontsize=8)
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Important Features (RF)')
        plt.gca().invert_yaxis()
    
    # 5. Model Performance Comparison
    plt.subplot(3, 4, 5)
    models = ['Random Forest', 'SVM']
    accuracies = [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_svm)]
    
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightcoral'])
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim([0, 1])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 6. Cross-validation results
    plt.subplot(3, 4, 6)
    cv_means = [cv_results['Random Forest']['accuracy_mean'], cv_results['SVM']['accuracy_mean']]
    cv_stds = [cv_results['Random Forest']['accuracy_std'], cv_results['SVM']['accuracy_std']]
    
    bars = plt.bar(models, cv_means, yerr=cv_stds, capsize=5, color=['lightgreen', 'lightsalmon'])
    plt.ylabel('Cross-validation Accuracy')
    plt.title('Cross-validation Performance')
    plt.ylim([0, 1])
    
    for bar, mean in zip(bars, cv_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{mean:.3f}', ha='center', va='bottom')
    
    # 7. Feature importance distribution
    plt.subplot(3, 4, 7)
    if 'feature_importance' in rf_results:
        plt.hist(rf_results['feature_importance']['importance'], bins=30, alpha=0.7, color='purple')
        plt.xlabel('Feature Importance')
        plt.ylabel('Number of Features')
        plt.title('Feature Importance Distribution')
    
    # 8. Prediction confidence distribution
    plt.subplot(3, 4, 8)
    rf_confidence = np.max(y_pred_proba_rf, axis=1)
    svm_confidence = np.max(y_pred_proba_svm, axis=1)
    
    plt.hist(rf_confidence, bins=20, alpha=0.7, label='Random Forest', color='blue')
    plt.hist(svm_confidence, bins=20, alpha=0.7, label='SVM', color='red')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Number of Predictions')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    
    # 9. Classification Reports Heatmap
    plt.subplot(3, 4, 9)
    rf_report = classification_report(y_test, y_pred_rf, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(rf_report).iloc[:-1, :-2].T  # Remove support and avg rows
    sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.3f')
    plt.title('RF Classification Report')
    
    plt.subplot(3, 4, 10)
    svm_report = classification_report(y_test, y_pred_svm, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(svm_report).iloc[:-1, :-2].T
    sns.heatmap(report_df, annot=True, cmap='Oranges', fmt='.3f')
    plt.title('SVM Classification Report')
    
    # 11. Cross-validation score distributions
    plt.subplot(3, 4, 11)
    rf_cv_scores = cv_results['Random Forest']['accuracy_scores']
    svm_cv_scores = cv_results['SVM']['accuracy_scores']
    
    x = np.arange(len(rf_cv_scores))
    width = 0.35
    
    plt.bar(x - width/2, rf_cv_scores, width, label='Random Forest', color='lightblue')
    plt.bar(x + width/2, svm_cv_scores, width, label='SVM', color='lightcoral')
    plt.xlabel('CV Fold')
    plt.ylabel('Accuracy')
    plt.title('Cross-validation Scores by Fold')
    plt.legend()
    plt.xticks(x, [f'Fold {i+1}' for i in range(len(rf_cv_scores))])
    
    # 12. Model comparison summary
    plt.subplot(3, 4, 12)
    metrics_comparison = pd.DataFrame({
        'Random Forest': [
            accuracies[0],
            cv_means[0],
            auc_rf if len(target_names) == 2 else auc_rf,
            rf_confidence.mean()
        ],
        'SVM': [
            accuracies[1],
            cv_means[1],
            auc_svm if len(target_names) == 2 else auc_svm,
            svm_confidence.mean()
        ]
    }, index=['Test Accuracy', 'CV Accuracy', 'AUC', 'Avg Confidence'])
    
    sns.heatmap(metrics_comparison, annot=True, cmap='RdYlBu_r', fmt='.3f', center=0.5)
    plt.title('Model Metrics Comparison')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 6. DETAILED PERFORMANCE ANALYSIS
# =============================================================================

def generate_detailed_performance_report(y_test, models_results, target_names, cv_results):
    """Generate detailed performance report"""
    
    print("\n" + "="*60)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*60)
    
    rf_results = models_results['Random Forest']
    svm_results = models_results['SVM']
    
    # Calculate all metrics
    rf_accuracy = accuracy_score(y_test, rf_results['predictions'])
    svm_accuracy = accuracy_score(y_test, svm_results['predictions'])
    
    # Classification reports
    rf_report = classification_report(y_test, rf_results['predictions'], 
                                    target_names=target_names, output_dict=True)
    svm_report = classification_report(y_test, svm_results['predictions'], 
                                     target_names=target_names, output_dict=True)
    
    # AUC scores
    if len(target_names) == 2:
        rf_auc = roc_auc_score(y_test, rf_results['probabilities'][:, 1])
        svm_auc = roc_auc_score(y_test, svm_results['probabilities'][:, 1])
    else:
        rf_auc = roc_auc_score(y_test, rf_results['probabilities'], 
                              multi_class='ovr', average='macro')
        svm_auc = roc_auc_score(y_test, svm_results['probabilities'], 
                               multi_class='ovr', average='macro')
    
    # Create comprehensive summary
    performance_summary = pd.DataFrame({
        'Model': ['Random Forest', 'SVM'],
        'Test_Accuracy': [rf_accuracy, svm_accuracy],
        'CV_Accuracy_Mean': [cv_results['Random Forest']['accuracy_mean'], 
                            cv_results['SVM']['accuracy_mean']],
        'CV_Accuracy_Std': [cv_results['Random Forest']['accuracy_std'], 
                           cv_results['SVM']['accuracy_std']],
        'AUC': [rf_auc, svm_auc],
        'Precision_Macro': [rf_report['macro avg']['precision'], 
                           svm_report['macro avg']['precision']],
        'Recall_Macro': [rf_report['macro avg']['recall'], 
                        svm_report['macro avg']['recall']],
        'F1_Score_Macro': [rf_report['macro avg']['f1-score'], 
                          svm_report['macro avg']['f1-score']]
    })
    
    print("\nCOMPREHENSIVE PERFORMANCE SUMMARY:")
    print(performance_summary.round(4).to_string(index=False))
    
    # Per-class performance
    print(f"\nPER-CLASS PERFORMANCE:")
    print(f"\nRandom Forest:")
    for class_name in target_names:
        if class_name in rf_report:
            metrics = rf_report[class_name]
            print(f"  {class_name:<15} - Precision: {metrics['precision']:.3f}, "
                  f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}")
    
    print(f"\nSVM:")
    for class_name in target_names:
        if class_name in svm_report:
            metrics = svm_report[class_name]
            print(f"  {class_name:<15} - Precision: {metrics['precision']:.3f}, "
                  f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}")
    
    return performance_summary

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Starting Phase 2: Model Development and Validation")
    
    # Load preprocessed data
    preprocessed_data = load_preprocessed_data()
    
    if preprocessed_data is None:
        exit(1)
    
    # Extract data
    X_train = preprocessed_data['X_train']
    X_test = preprocessed_data['X_test']
    y_train = preprocessed_data['y_train']
    y_test = preprocessed_data['y_test']
    target_names = preprocessed_data['target_names']
    
    # Option 1: Quick training with default parameters
    print("\nOption 1: Training with default parameters (faster)")
    
    # Train Random Forest
    rf_model, y_pred_rf, y_pred_proba_rf, feature_importance = train_random_forest(
        X_train, y_train, X_test, y_test
    )
    
    # Train SVM
    svm_model, y_pred_svm, y_pred_proba_svm = train_svm(
        X_train, y_train, X_test, y_test
    )
    
    # Prepare models for cross-validation
    models_dict = {
        'Random Forest': RandomForestClassifier(n_estimators=500, max_depth=10, 
                                              min_samples_split=5, random_state=42, n_jobs=-1),
        'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    }
    
    # Perform cross-validation
    cv_results = perform_cross_validation(models_dict, X_train, y_train)
    
    # Store results
    models_results = {
        'Random Forest': {
            'model': rf_model,
            'predictions': y_pred_rf,
            'probabilities': y_pred_proba_rf,
            'feature_importance': feature_importance
        },
        'SVM': {
            'model': svm_model,
            'predictions': y_pred_svm,
            'probabilities': y_pred_proba_svm
        }
    }
    
    # Create comprehensive evaluation plots
    create_comprehensive_evaluation_plots(y_test, models_results, target_names, cv_results)
    
    # Generate detailed performance report
    performance_summary = generate_detailed_performance_report(
        y_test, models_results, target_names, cv_results
    )
    
    # Save results for Phase 3
    phase2_results = {
        'models_results': models_results,
        'cv_results': cv_results,
        'performance_summary': performance_summary,
        'target_names': target_names,
        'y_test': y_test
    }
    
    with open('phase2_results.pkl', 'wb') as f:
        pickle.dump(phase2_results, f)
    
    print("\nPhase 2 completed successfully!")
    print("Results saved as 'phase2_results.pkl'")
    print("\nNext: Run Phase 3 for clinical interpretation and reporting")
    
    # Optional: Hyperparameter tuning (uncomment if needed)
    """
    print("\n" + "="*50)
    print("OPTIONAL: HYPERPARAMETER TUNING")
    print("="*50)
    print("Uncomment this section for hyperparameter tuning (takes longer)")
    
    # Tune hyperparameters
    best_rf = tune_random_forest(X_train, y_train)
    best_svm = tune_svm(X_train, y_train)
    
    # Retrain with best parameters
    rf_model_tuned, y_pred_rf_tuned, y_pred_proba_rf_tuned, feature_importance_tuned = train_random_forest(
        X_train, y_train, X_test, y_test, best_rf
    )
    
    svm_model_tuned, y_pred_svm_tuned, y_pred_proba_svm_tuned = train_svm(
        X_train, y_train, X_test, y_test, best_svm
    )
    """