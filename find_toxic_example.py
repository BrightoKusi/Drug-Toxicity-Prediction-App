import pandas as pd

# Load the raw training data
df = pd.read_csv('data/raw/tox21_raw_train.csv')

# Find a compound that is definitively toxic (1) for the SR-ARE assay
# .dropna() ensures we get a valid label
toxic_example = df[df['SR-ARE'] == 1.0].dropna(subset=['SR-ARE']).iloc[0]

# Find a compound that is definitively non-toxic (0) for the SR-ARE assay
non_toxic_example = df[df['SR-ARE'] == 0.0].dropna(subset=['SR-ARE']).iloc[0]

print("=== KNOWN TOXIC (SR-ARE) ===")
print("SMILES:", toxic_example['smiles'])
print("Label:", toxic_example['SR-ARE'])
print("\n=== KNOWN NON-TOXIC (SR-ARE) ===")
print("SMILES:", non_toxic_example['smiles'])
print("Label:", non_toxic_example['SR-ARE'])