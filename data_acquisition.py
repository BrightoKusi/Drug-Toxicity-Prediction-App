# data_acquisition.py
import deepchem as dc
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import logging

# Set up logging to track the process
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_tox21_data():
    """
    Loads the Tox21 dataset using DeepChem's loader.
    Returns a tuple of (train, valid, test) datasets.
    """
    logger.info("Loading Tox21 dataset...")
    tasks, datasets, transformers = dc.molnet.load_tox21()
    train_dataset, valid_dataset, test_dataset = datasets
    return tasks, train_dataset, valid_dataset, test_dataset

def deepchem_to_dataframe(dataset, tasks):
    """
    Converts a DeepChem dataset to a Pandas DataFrame.
    This is crucial for our standard ML workflow.
    """
    logger.info("Converting DeepChem dataset to Pandas DataFrame...")
    # Extract SMILES strings, features (ignored for now), and labels
    smiles = dataset.ids
    # labels is a 2D array of shape (n_samples, n_tasks)
    labels = dataset.y

    # Create a DataFrame
    df = pd.DataFrame(smiles, columns=['smiles'])
    # For each task, add a column to the DataFrame
    for i, task_name in enumerate(tasks):
        df[task_name] = labels[:, i]

    # DeepChem uses -1 to indicate missing labels. We'll convert to NaN.
    df.replace(-1, np.nan, inplace=True)
    return df

def save_raw_data(train_df, valid_df, test_df, filename_prefix='data/raw/tox21_raw'):
    """Saves the raw DataFrames to CSV files."""
    logger.info("Saving raw data to CSV...")
    train_df.to_csv(f'{filename_prefix}_train.csv', index=False)
    valid_df.to_csv(f'{filename_prefix}_valid.csv', index=False)
    test_df.to_csv(f'{filename_prefix}_test.csv', index=False)

if __name__ == '__main__':
    # Execute the ETL process
    tasks, train_dc, valid_dc, test_dc = load_tox21_data()
    train_df = deepchem_to_dataframe(train_dc, tasks)
    valid_df = deepchem_to_dataframe(valid_dc, tasks)
    test_df = deepchem_to_dataframe(test_dc, tasks)

    # Save the data
    save_raw_data(train_df, valid_df, test_df)
    logger.info("Data acquisition and saving complete!")