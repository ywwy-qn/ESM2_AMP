import pandas as pd
import os
import time
import sys
import logging
from pathlib import Path

project_root = Path(__file__)
sys.path.append(str(project_root))


from transformers import AutoTokenizer, AutoModelForMaskedLM
from protloc_mex_X.ESM2_fr import Esm2LastHiddenFeatureExtractor


# Initialize the tokenizer and model with the pretrained ESM2 model
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", output_hidden_states=True)



feature_extractor = Esm2LastHiddenFeatureExtractor(tokenizer, model,
                                                   compute_cls=True, compute_eos=True, compute_mean=True,
                                                   compute_segments=True)

    

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

file_path = os.path.join('esm2_infer_feature/data/protein_example.xlsx')
PPI_protein = pd.read_excel(file_path)

df = PPI_protein
start_time = time.time()
df_represent = pd.DataFrame()
total_rows = len(df)
chunk_size=700



for chunk_idx, i in enumerate(range(0, total_rows, chunk_size)):
    df_chunk = df.iloc[i:i + chunk_size]
    df_chunk_represent = feature_extractor.get_last_hidden_features_combine(df_chunk, sequence_name='Sequence',
                                                                            batch_size=1)
    save_path = 'esm2_infer_feature/output'
    chunk_folder = os.path.join(save_path, f'chunk{chunk_idx}')
    os.makedirs(chunk_folder, exist_ok=True)
    chunk_file_path = os.path.join(chunk_folder, 'protein_feature_dataframe.h5')
    df_chunk_represent.to_hdf(chunk_file_path, key='df', mode='w')
    print(f"Processed and saved rows {i} to {i + chunk_size} to {chunk_file_path}.")

    assert isinstance(df_chunk_represent, pd.DataFrame), "Feature extraction did not return a DataFrame."

    print(f"Processed rows {i} to {i + chunk_size}.")

print("Feature extraction for all chunks completed.")

logging.info(f"Total time: {(time.time()-start_time)} seconds")


logging.info('Save completed...')






