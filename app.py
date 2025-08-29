# app.py
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import joblib
import matplotlib.pyplot as plt

# Set the page title and icon
st.set_page_config(
    page_title="Toxicity Predictor",
    page_icon="🧪",
    layout="wide"
)

# App title and description
st.title("🧪 Toxicity Prediction App")
st.markdown("""
Oxidative stress is a key mechanism behind many types of drug-induced toxicity, especially liver toxicity (hepatotoxicity). 
Predicting this activity early in drug discovery can save millions of dollars and prevent failures in later clinical stages.

This app predicts the likelihood of a compound being toxic based on the **SR-ARE (Stress Response - Antioxidant Response Element)**  assay from the Tox21 dataset.
 It uses a machine learning model trained on molecular fingerprints.
""")

# Load your trained model
@st.cache_resource # This decorator caches the model load, making the app faster
def load_model():
    model = joblib.load('models/tox21_xgboost_model.pkl')
    return model

model = load_model()

# Function to convert SMILES to Morgan fingerprint
def smiles_to_morgan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fp)

# Create a two-column layout
col1, col2 = st.columns(2)

with col1:
    st.header("Input Molecule")
    
    # Input method: SMILES string
    smiles_input = st.text_input(
        "Enter a SMILES string:",
        value="CCO",  # Default value: Ethanol
        help="e.g., CCO for ethanol, CN1C=NC2=C1C(=O)N(C(=O)N2C)C for caffeine"
    )
    
    # Input method: File upload
    uploaded_file = st.file_uploader("Or upload a SDF or SMILES file", type=['sdf', 'smiles', 'txt'])
    
    if uploaded_file is not None:
        # Simple file content display (you could add parsing here)
        content = uploaded_file.getvalue().decode("utf-8")
        smiles_input = content.split('\n')[0].strip()  # Take first line as SMILES
        st.write(f"Read SMILES from file: `{smiles_input}`")

with col2:
    st.header("Molecular Visualization")
    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            # Generate molecular image
            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img, caption="2D Molecular Structure")
            
            # Calculate molecular properties
            mol_weight = Chem.rdMolDescriptors.CalcExactMolWt(mol)
            st.write(f"**Molecular Weight:** {mol_weight:.2f} g/mol")
        else:
            st.error("Invalid SMILES string. Please enter a valid molecular structure.")

# Prediction section
st.header("Toxicity Prediction")
if smiles_input and mol:
    # Convert SMILES to features
    fingerprint = smiles_to_morgan(smiles_input)
    
    if fingerprint is not None:
        # Create DataFrame with correct feature names
        feature_names = [f'morgan_{i}' for i in range(2048)]
        input_df = pd.DataFrame([fingerprint], columns=feature_names)
        
        # Make prediction
        prediction_proba = model.predict_proba(input_df)[0]
        toxic_prob = prediction_proba[1]  # Probability of class 1 (toxic)
        non_toxic_prob = prediction_proba[0]  # Probability of class 0 (non-toxic)
        
        # Display results with color coding
        st.subheader("Prediction Results")
        
        # Create a metric display
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Probability of NON-TOXIC",
                value=f"{non_toxic_prob:.1%}",
                delta="Safe" if non_toxic_prob > 0.7 else None,
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                label="Probability of TOXIC",
                value=f"{toxic_prob:.1%}",
                delta="Warning" if toxic_prob > 0.5 else None,
                delta_color="inverse"
            )
        
        # Visualize probabilities
        fig, ax = plt.subplots(figsize=(8, 2))
        bars = ax.barh(['Non-Toxic', 'Toxic'], [non_toxic_prob, toxic_prob], 
                      color=['#4CAF50', '#F44336'])
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability')
        ax.set_title('Toxicity Probability Distribution')
        
        # Add value labels on bars
        for bar, value in zip(bars, [non_toxic_prob, toxic_prob]):
            ax.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height()/2,
                   f'{value:.1%}', ha='right', va='center', color='white', 
                   fontweight='bold')
        
        st.pyplot(fig)
        
        # Interpretation guidance
        st.subheader("Interpretation Guide")
        if toxic_prob > 0.7:
            st.error("🚨 High probability of toxicity. Further experimental validation is strongly recommended.")
        elif toxic_prob > 0.4:
            st.warning("⚠️ Moderate probability of toxicity. Consider additional screening.")
        else:
            st.success("✅ Low probability of toxicity. May proceed with further development steps.")
        
    else:
        st.error("Could not generate molecular features. Please check the SMILES string.")
elif smiles_input:
    st.error("Please enter a valid SMILES string to get a prediction.")

# Add footer with project information
st.markdown("---")
st.markdown("""
**Project Details:**
- **Model:** XGBoost Classifier
- **Training Data:** Tox21 SR-ARE assay
- **Features:** 2048-bit Morgan Fingerprints
- **Performance:** AUC-PR = 0.34, Recall = 47%
""")