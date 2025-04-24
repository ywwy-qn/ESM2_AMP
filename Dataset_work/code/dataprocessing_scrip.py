# Import libraries and initialize paths
import os
import re
import sys
import time
import torch
import argparse
import subprocess
import numpy as np
import pandas as pd

# Configure project paths
project_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(os.path.join(project_path, 'Dataset_work'))

# Import custom feature extractor
from protloc_mex_X.ESM2_fr import Esm2LastHiddenFeatureExtractor
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Configure command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seqf', type=str, required=True, help='Protein sequence file path')
parser.add_argument('--pairsf', type=str, required=True, help='Protein pairs file path')
args = parser.parse_args()

# Initialize file paths
sequence_file = args.seqf
pairs_file = args.pairsf

# Configure output directories
chunk_folder = os.path.join(project_path, r'Dataset_work\dataset_feature')
protein_ppifile = os.path.join(project_path, r'Dataset_work\dataset_feature\data_feature.h5')

# Load and filter protein sequences
PPI_protein = pd.read_excel(sequence_file)
PPI_protein = PPI_protein[PPI_protein['Sequence'].apply(len) <= 4000]  # Filter long sequences
os.makedirs(chunk_folder, exist_ok=True)

# Protein feature extraction setup
# ===========================================================================
# Initialize ESM-2 model components (update model path as needed)
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", 
                                           output_hidden_states=True)

# Configure feature extractor with multiple output features
feature_extractor = Esm2LastHiddenFeatureExtractor(
    tokenizer, model,
    compute_cls=True, 
    compute_eos=True,
    compute_mean=True,
    compute_segments=True
)

# Memory-efficient feature extraction with chunk processing
chunk_size = 700
for chunk_idx, i in enumerate(range(0, len(PPI_protein), chunk_size)):
    # Process data in chunks
    df_chunk = PPI_protein.iloc[i:i + chunk_size]
    
    # Progress tracking
    print(f"Processing rows {i}/{len(PPI_protein)} (Total: {len(PPI_protein)})")
    
    # Extract features with batch size 1 for memory conservation
    df_chunk_represent = feature_extractor.get_last_hidden_features_combine(
        df_chunk, 
        sequence_name='Sequence', 
        batch_size=1
    )
    
    # Save chunk results
    chunk_file = os.path.join(chunk_folder, f'protein_feature_chunk{chunk_idx}.h5')
    df_chunk_represent.to_hdf(chunk_file, key='df', mode='w')

print("Feature extraction for all chunks completed.")

# Feature aggregation and processing
# ===========================================================================
# Initialize data container
all_data = []

# Combine all chunk files
for file_name in os.listdir(chunk_folder):
    chunk_file = os.path.join(chunk_folder, file_name)
    if os.path.exists(chunk_file):
        df = pd.read_hdf(chunk_file, key='df')
        all_data.append(df)

# Create consolidated dataframe
final_df = pd.concat(all_data, ignore_index=True)

# Select relevant ESM-2 features using regex patterns
feature_columns = []
patterns_f = ['Entry', r'ESM2_cls\d+', r'ESM2_eos\d+', r'ESM2_segment\d+']
for pattern in patterns_f:
    for col in final_df.columns:
        if re.match(pattern, col):
            feature_columns.append(col)

feature_all_reserve1 = final_df[feature_columns]

# Column reordering to specific format
new_index = ['Entry']
feature_prefixes = [
    'ESM2_cls', 
    'ESM2_segment0_mean',
    'ESM2_segment1_mean',
    'ESM2_segment2_mean',
    'ESM2_segment3_mean',
    'ESM2_segment4_mean',
    'ESM2_segment5_mean',
    'ESM2_segment6_mean',
    'ESM2_segment7_mean',
    'ESM2_segment8_mean',
    'ESM2_segment9_mean', 
    'ESM2_eos'
]

# Generate column order dynamically
for prefix in feature_prefixes:
    for i in range(1280):
        col_name = f"{prefix}{i}"
        if col_name in feature_all_reserve1.columns:
            new_index.append(col_name)

feature_all_reserve = feature_all_reserve1[new_index]
del feature_all_reserve1, final_df  # Memory cleanup

# Protein pair processing
# ===========================================================================
# Load interaction pairs and merge features
test_data = pd.read_excel(pairs_file)

# Merge features for first protein in pair
test_data = pd.merge(test_data, feature_all_reserve, 
                    how='left', 
                    left_on='Protein1', 
                    right_on='Entry')

# Rename columns for first protein
new_column_names1 = []
for name in test_data.columns:
    if re.match('ESM.', name):
        new_name = '1_' + name
    elif name == 'Entry':
        new_name = 'Entry1'
    else:
        new_name = name
    new_column_names1.append(new_name)
test_data.columns = new_column_names1

# Merge features for second protein in pair
test_data = pd.merge(test_data, feature_all_reserve, 
                    how='left', 
                    left_on='Protein2', 
                    right_on='Entry')

# Rename columns for second protein
new_column_names2 = []
for name in test_data.columns:
    if re.match('ESM.', name):
        new_name2 = '2_' + name
    else:
        new_name2 = name
    new_column_names2.append(new_name2)
test_data.columns = new_column_names2

# Cleanup redundant columns
test_data = test_data.drop(columns=['Entry1', 'Entry'])

# Save final feature set
test_data.to_hdf(protein_ppifile, key='df', mode='w')
print(f"Feature data successfully saved to: {protein_ppifile}")