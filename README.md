---
title: Toxicity Predictor
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# 🧪 Toxicity Prediction App

This app predicts the likelihood of a compound being toxic based on the **SR-ARE assay** from the NIH Tox21 dataset. It uses an XGBoost machine learning model trained on molecular fingerprints ([Morgan Fingerprints](https://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints)).

## How to Use

1.  **Enter a SMILES string** in the input box (e.g., `CCO` for ethanol, `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` for caffeine).
2.  The app will display the 2D structure of the molecule.
3.  The model will output two predicted probabilities:
    -   **Probability of NON-TOXIC**
    -   **Probability of TOXIC**
4.  An interpretation guide provides context for the result.

## Model Details

-   **Task:** Binary classification (Toxic vs. Non-Toxic)
-   **Model:** XGBoost Classifier
-   **Feature:** 2048-bit Morgan Fingerprints (radius=2)
-   **Training Data:** Tox21 SR-ARE assay
-   **Performance:** AUC-PR = 0.34, Recall = 47%

## Limitations & Scope

This model is trained specifically to predict activity in the **SR-ARE** assay, which measures activation of the antioxidant response element (ARE) pathway, a specific type of oxidative stress. It does not predict general toxicity or other toxicity endpoints (e.g., genotoxicity, hepatotoxicity).

## 🛠️ Build & Run Locally

```bash
# Clone the repository
git clone <your-repo-url>
cd tox21_project

# Build the Docker image
docker build -t toxicity-predictor .

# Run the container
docker run -p 8501:8501 toxicity-predictor