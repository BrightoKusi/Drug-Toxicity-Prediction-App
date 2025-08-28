# feature_engineering.py
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def smiles_to_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """
    Converts a SMILES string to a Morgan fingerprint vector.
    Parameters:
        radius: The radius of the circular fingerprint.
        n_bits: The length of the fingerprint vector.
    Returns:
        A numpy array of length `n_bits` or None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

def compute_descriptors(smiles):
    """
    Computes a set of RDKit molecular descriptors for a SMILES string.
    Returns a list of descriptor values or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Get a list of all descriptor names
    descriptor_names = [x[0] for x in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    return calculator.CalcDescriptors(mol)

def create_feature_dataframe(df, target_task='SRARE'):
    """
    Main function to create a feature matrix (X) and target vector (y).
    Drops rows where the target task label is NaN.
    """
    logger.info("Starting feature engineering...")

    # 1. Handle the target - drop rows where the label for our chosen task is missing
    df_target = df[['smiles', target_task]].dropna(subset=[target_task])
    y = df_target[target_task].values
    logger.info(f"Target task '{target_task}' has {len(y)} valid samples.")

    # 2. Generate features for each valid SMILES string
    morgan_fps = []
    descriptor_list = []
    valid_smiles_list = []

    for smiles in df_target['smiles']:
        # Generate Morgan Fingerprint
        fp = smiles_to_morgan_fingerprint(smiles)
        # Generate Descriptors
        desc = compute_descriptors(smiles)

        if fp is not None and desc is not None:
            morgan_fps.append(fp)
            descriptor_list.append(desc)
            valid_smiles_list.append(smiles)
        else:
            logger.warning(f"Invalid SMILES string dropped: {smiles}")

    # Convert lists to arrays
    X_morgan = np.array(morgan_fps)
    X_descriptors = np.array(descriptor_list)
    y_final = y[:len(X_morgan)] # Align target with our valid features

    # 3. Create DataFrames for the different feature sets
    feature_data_morgan = pd.DataFrame(X_morgan, columns=[f'morgan_{i}' for i in range(X_morgan.shape[1])])
    feature_data_morgan['smiles'] = valid_smiles_list

    descriptor_names = [x[0] for x in Descriptors._descList]
    feature_data_descriptors = pd.DataFrame(X_descriptors, columns=descriptor_names)
    feature_data_descriptors['smiles'] = valid_smiles_list

    # 4. Save these processed feature sets
    feature_data_morgan.to_csv(f'data/processed/tox21_{target_task}_morgan_features.csv', index=False)
    feature_data_descriptors.to_csv(f'data/processed/tox21_{target_task}_descriptor_features.csv', index=False)
    pd.DataFrame({'smiles': valid_smiles_list, 'target': y_final}).to_csv(f'data/processed/tox21_{target_task}_target.csv', index=False)

    logger.info(f"Feature engineering complete. Morgan shape: {X_morgan.shape}, Descriptors shape: {X_descriptors.shape}")
    return feature_data_morgan, feature_data_descriptors, y_final

if __name__ == '__main__':
    # Load the raw data
    train_df = pd.read_csv('data/raw/tox21_raw_train.csv')
    # Choose a single task to start with, e.g., 'SRARE'
    create_feature_dataframe(train_df, target_task='SR-ARE')