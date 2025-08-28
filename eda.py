# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the processed target
target_df = pd.read_csv('data/processed/tox21_SR-ARE_target.csv')
print("Target value counts:")
print(target_df['target'].value_counts())

# Plot the class distribution
plt.figure(figsize=(6,4))
sns.countplot(x=target_df['target'])
plt.title('Class Distribution (SRARE Assay)')
plt.xlabel('Toxic (1) vs. Non-Toxic (0)')
plt.savefig('data/processed/class_distribution.png')
plt.show()