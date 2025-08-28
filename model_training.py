# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting model training pipeline...")

    # 1. Load Processed Data
    logger.info("Loading processed features and target...")
    X = pd.read_csv('data/processed/tox21_SR-ARE_morgan_features.csv')
    y_df = pd.read_csv('data/processed/tox21_SR-ARE_target.csv')
    
    # Separate the SMILES column from the features for model training
    smiles_list = X['smiles']
    X_model = X.drop(columns=['smiles'])
    y = y_df['target']

    # 2. Train-Test Split (STRATIFY to preserve imbalance in both sets)
    logger.info("Performing stratified train-test split...")
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
        X_model, y, smiles_list, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Calculate the scale_pos_weight to handle class imbalance
    weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"Class imbalance ratio (non-toxic/toxic): {weight_ratio:.2f}")
    logger.info(f"Using scale_pos_weight = {weight_ratio:.2f} for XGBoost")

    # 4. Train XGBoost Model
    logger.info("Training XGBoost model...")
    model = xgb.XGBClassifier(
        scale_pos_weight=weight_ratio,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)

    # 5. Evaluate Model Performance
    logger.info("Evaluating model performance...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for the positive class

    # 5a. Classification Report (Precision, Recall, F1)
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=['Non-Toxic (0)', 'Toxic (1)']))

        # 5b. Confusion Matrix (SAVE WITHOUT DISPLAY)
    print("\nConfusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Toxic', 'Toxic'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    plt.title('Confusion Matrix for Toxicity Prediction')
    plt.savefig('models/confusion_matrix.png')
    plt.close(fig)  # <- Critical: Close the figure to avoid display issues
    print("Saved confusion_matrix.png")

    # 5c. Precision-Recall Curve (MOST IMPORTANT METRIC) (SAVE WITHOUT DISPLAY)
    average_precision = average_precision_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f'XGBoost (AP = {average_precision:.2f}, AUC-PR = {auc_pr:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="best")
    plt.savefig('models/precision_recall_curve.png')
    plt.close(fig)  # <- Critical: Close the figure
    print("Saved precision_recall_curve.png")

    logger.info(f"Average Precision (AP): {average_precision:.3f}")
    logger.info(f"AUC-PR: {auc_pr:.3f}")

    # 6. Interpret Model with SHAP (SAVE WITHOUT DISPLAY)
    logger.info("Generating SHAP explanations... (This may take a few minutes)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot: Which features are most important?
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False) # show=False is key
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig('models/shap_feature_importance.png')
    plt.close() # Close the plot to avoid display issues
    print("Saved shap_feature_importance.png")
    
    # 7. Save the trained model for later use in the dashboard
    logger.info("Saving model to disk...")
    joblib.dump(model, 'models/tox21_xgboost_model.pkl')
    logger.info("Model saved as 'models/tox21_xgboost_model.pkl'")

    logger.info("Model training and evaluation pipeline complete!")

if __name__ == '__main__':
    # Create directory for saving models and plots
    import os
    os.makedirs('models', exist_ok=True)
    main()