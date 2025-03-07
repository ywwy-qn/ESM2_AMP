

# ESM2_AMP: an Interpretable Framework for Protein-protein Interactions Prediction and Biological Mechanism Discovery

## Introduction

This project revolves around the paper titled "**ESM2_AMP: an Interpretable Framework for Protein-protein Interactions Prediction and Biological Mechanism Discovery**", aiming to provide relevant datasets and model resources. Through our framework, we hope to enhance the understanding of protein interactions and their underlying biological mechanisms, offering robust support for research in related fields.

### Author Contact Information:

- Author 1: Yawen Sun, Email: [2108437154@qq.com](mailto:2108437154@qq.com)
- Author 2: Rui Wang, Email: [2219312248@qq.com](mailto:2219312248@qq.com)
- Author 3: Zeyu Luo, Email: [1024226968@qq.com](mailto:1024226968@qq.com)

Your contributions, feedback, and suggestions are highly appreciated. If you encounter any issues or have questions, feel free to reach out to the authors via the provided email addresses. Thank you for your interest in our work!

## Work Environment Setup

To ensure that you can replicate our work from the paper accurately, we recommend using the following work environment:   
- Python version: 3.11.4
- protloc-mex-x version: 0.0.13

## Dataset Availability

This project provides training datasets and independent test sets for researchers to explore further. All data resources can be accessed in the [Datasets](https://github.com/ywwy-qn/ESM2_AMP/tree/main/Datasets#dataset) section. Detailed information on methods for extracting feature representations from protein sequences can be found in the published research paper (DOI: https://doi.org/10.1093/bib/bbad534), as well as in the corresponding GitHub repository: [Feature Representation for LLMs](https://github.com/yujuan-zhang/feature-representation-for-LLMs?tab=readme-ov-file#feature-representation-model). For detailed information on the extraction of feature embeddings for specific proteins, please refer to the Python library protloc-mex-x (https://pypi.org/project/protloc-mex-x/).

## Methods

### Feature representation
The proteins' feature representation used the pre-trained protein model ESM2 developed by Meta company and placed on Hugging Face. For more details, please search in https://huggingface.co/facebook/esm2_t33_650M_UR50D. Besides, we used [protloc-mex-x](https://pypi.org/project/protloc_mex_X/) which our team developed, containing detail for 'cls','mean', 'eos','segment 0-9' feature representation from ESM2.
 ```python
 tokenizer = AutoTokenizer.from_pretrained(modelPath + "/esm2_t33_650M_UR50D")
 model = AutoModelForMaskedLM.from_pretrained(modelPath + "/esm2_t33_650M_UR50D", output_hidden_states=True)
 protein_sequence_df = pd.read_excel(excel_file_path)
 feature_extractor = Esm2LastHiddenFeatureExtractor(tokenizer, model,
                                                    compute_cls=True, compute_eos=True, compute_mean=True,
                                                    compute_segments=True)
 print(f"{feature_extractor.device}")
```
Get all the required features:
 ```python
 feature_columns = []
 patterns_f = [r'ESM2_cls\d+', r'ESM2_eos\d+', r'ESM2_mean\d+']
 patterns_f += [f'ESM2_segment{i}_mean\d+' for i in range(10)]
 for pattern in patterns_f:
     for col in df_represent.columns:
         if re.match(pattern, col):
             feature_columns.append(col)
 ```



Then, we used function pd.merge() from the pandas Python package to concatenate data based on the different features required of protein pairs by each model. Taking the features required by the ESM2_AMP_CSE model as an example:

(1) The extracted features mentioned above represent the features of all proteins included in all samples. To obtain the features (ESM2_cls, ESM2_segment0-9, ESM2_eos) of the desired proteins.
 ```python
feature_columns = []
patterns_f = ['Entry', r'ESM2_cls\d+', r'ESM2_eos\d+',  r'ESM2_segment\d+']

for pattern in patterns_f:
    for col in final_df.columns:
        if re.match(pattern, col):
            feature_columns.append(col)

feature_all_reserve1 = final_df[feature_columns]
 ```
(2) The segment local features correspond to amino acid fragments containing actual sequences, numbered sequentially from 0 to 9. The ESM2_cls feature represents the feature embedding of the CLS token added at the beginning of the sequence, while the ESM2_eos feature represents the feature embedding of the EOS token appended at the end of the sequence. Therefore, the features should be ordered as ESM2_cls, ESM2_segment0-9, and ESM2_eos.
 ```python
 new_index = ['Entry']
 for prefix in ['ESM2_cls', 'ESM2_segment0_mean', 'ESM2_segment1_mean',
                'ESM2_segment2_mean','ESM2_segment3_mean','ESM2_segment4_mean',
                'ESM2_segment5_mean','ESM2_segment6_mean','ESM2_segment7_mean','ESM2_segment8_mean',
                'ESM2_segment9_mean', 'ESM2_eos']:
     for i in range(1280):
         col_name = f"{prefix}{i}"
         if col_name in feature_all_reserve1.columns:
             new_index.append(col_name)
 feature_all_reserve = feature_all_reserve1[new_index]
 ```
(3) The samples used for PPIs prediction are in the format of protein pairs, named Protein1 and Protein2 respectively. In the protein feature files extracted through the upon steps, the protein column is labeled as Entry. The pd.merge() function is used to concatenate the data.
 ```python
 protein_pairs = pd.merge(protein_pairs, feature_all_reserve, how='left',
                                       left_on='Protein1', right_on='Entry')
 new_column_names1 = []
 for name in protein_pairs.columns:
     if re.match('ESM.', name):
         new_name = '1_' + name
     elif name == 'Entry':
         new_name = 'Entry1'
     else:
         new_name = name
     new_column_names1.append(new_name)
 protein_pairs.columns = new_column_names1
 protein_pairs = pd.merge(protein_pairs, feature_all_reserve, how='left', left_on='Protein2', right_on='Entry')
 new_column_names2 = []
 for name in protein_pairs.columns:
     if re.match('ESM.', name):
         new_name2 = '2_' + name
     else:
         new_name2 = name
     new_column_names2.append(new_name2)
 protein_pairs.columns = new_column_names2
 ```

The initial protein pair features input to the Transformer encoder are constructed using the following method through a DataLoader. For each sample (i.e., a protein pair), the features are organized into a 2D matrix based on their different characteristics. If N features are selected, each feature has a dimensionality of 1280, resulting in a feature matrix of size N*1280 for each sample.

### Models

This project encompasses a series of models, including [ESM2_AMPS](https://github.com/ywwy-qn/ESM2_AMP/tree/main/Models/ESM2_AMPS), [ESM2_AMP_CSE](https://github.com/ywwy-qn/ESM2_AMP/tree/main/Models/ESM2_AMP_CSE), and [ESM2_DPM](https://github.com/ywwy-qn/ESM2_AMP/tree/main/Models/ESM2_DPM), aimed at providing comprehensive support for predicting protein interactions. Example inference code for each model is provided within their respective directories, while the required model weight files can be downloaded from the project's corresponding [figshare](https://figshare.com/articles/dataset/ESM2_AMP/28378157) page.

### Model training

During model training, the **AdamW** algorithm and **Optuna** are used for hyperparameter tuning to reduce sensitivity to the selection of parameters such as learning rate and weight decay, while GPU acceleration is employed to speed up model training. The **ReLU** activation function is adopted to accelerate model training and achieve better prediction results. Additionally, **He initialization** is applied to mitigate gradient vanishing and explosion issues, thereby improving the model's convergence speed.

(1) Install all required python packages
 ```python
 import numpy as np
 import pandas as pd
 import torch
 import torch.nn as nn
 import torch.optim as optim
 from torch.nn import TransformerEncoder, TransformerEncoderLayer
 from torch.optim.lr_scheduler import ReduceLROnPlateau
 from torch.nn.init import kaiming_normal_
 from torch.optim import AdamW
 import optuna
 import time
 import logging
 from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
 import matplotlib.pyplot as plt
 from sklearn.metrics import matthews_corrcoef
 from sklearn.model_selection import StratifiedKFold
 import torch.nn.init as init
 ```

(If these packages are not installed, follow the code below to install them):
 ```python
 pip install numpy pandas torch optuna scikit-learn matplotlib
 ```
 Install it on the CLI
 ```python
 !pip install numpy pandas torch optuna scikit-learn matplotlib
 ```

If you are using **Anaconda**, you can also install these packages using **conda**
 ```python
 conda install numpy pandas pytorch optuna scikit-learn matplotlib -c pytorch
 ```

(2) Input data and create CustomDataset class.
 ```python
 class CustomDataset(torch.utils.data.Dataset):
     def __init__(self, data_features, data_label, reshape_shape=(20, 1280)):
         self.features = torch.tensor(data_features.values.reshape(-1, *reshape_shape)).float()
         assert len(data_features) == len(data_label), "The sample number of features and label data does not match."
         self.labels = torch.tensor(data_label).float()
     def __len__(self):
         return len(self.features)

     def __getitem__(self, idx):
         return self.features[idx], self.labels[idx]
 ```
(3) Model Design (Specific information about the [ESM2_AMPS](https://github.com/ywwy-qn/ESM2_AMP/tree/main/Models/ESM2_AMPS), [ESM2_AMP_CSE](https://github.com/ywwy-qn/ESM2_AMP/tree/main/Models/ESM2_AMP_CSE), and [ESM2_DPM](https://github.com/ywwy-qn/ESM2_AMP/tree/main/Models/ESM2_DPM) models can be found in the Model section).

(4) Model training
 ```python
 # Gets the hyperparameter section
 lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)  # 使用loguniform分布更合理
 weight_decay = trial.suggest_loguniform('weight_decay', 1e-4, 1e-2)

 # Part of hyperparameter optimization
 criterion = nn.BCEWithLogitsLoss()
 optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
 scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-8)
 ```

### Model evaluation

During the model evaluation phase, multiple metrics such as **Accuracy**, **MCC**, **Recall**, **F1 score**, and **Precision** are used to assess the model's performance.The evaluation metrics and calculation methods are shown in code: 

 ```python
val_loss = 0.0
model.eval()
with torch.no_grad():
   y_pred_list = []
   y_true_list = []
   all_outputs = []
   all_labels = []

for features, labels in val_dataloader:
   features, labels = features.to(device), labels.float().to(device)
   outputs = model(features)
   all_outputs.append(outputs)
   all_labels.append(labels)
   loss = criterion(outputs, labels)
   val_loss += loss.item()
   predictions = outputs.sigmoid().cpu().numpy() > 0.5
   y_pred_list.extend(predictions)
   y_true_list.extend(labels.cpu().numpy())
                
 # Calculating AUC
 all_outputs2 = torch.cat(all_outputs, dim=0)
 all_labels2 = torch.cat(all_labels, dim=0)
 auc = roc_auc_score(all_labels2.cpu().numpy(), all_outputs2.cpu().numpy())

 # Calculating MCC, Accuracy, Recall, F1 score
 mcc = matthews_corrcoef(y_true_list, y_pred_list)
 accuracy = accuracy_score(y_true_list, y_pred_list)
 recall = recall_score(y_true_list, y_pred_list, pos_label=1)
 f1 = f1_score(y_true_list, y_pred_list, pos_label=1)
 ```


### Attention-based Explainable Analysis

Both models within the ESM2_AMP framework utilize the multi-head attention mechanism of the Transformer encoder. By leveraging the weight matrix allocation in the multi-head attention mechanism, the attention weights corresponding to the sample features are extracted, and their feature importance is calculated and quantified. The detailed process will be updated in subsequent article publications.

### Feature attribution interpretable method

#### model constructing
The features obtained from ESM2 are fed into an autoencoder for dimensionality reduction to derive a new feature representation, which is then input into a random forest model. This process is named AE_RF and serves as the underlying model for Tree SHAP.

**AE pretraining**
We performed dimensionality reduction on the extracted ESM2 protein feature representations, which originally had 1280 dimensions per feature, with the Autoencoder's hidden layer dimension set to 150. It is necessary to restructure the original feature dimensions to fit the input requirements of the Autoencoder.
(1) Taking the original 2D matrix of size N15360 as an example, where N is the number of proteins and 15360 is the sum of the 1280 dimensions of ESM2_cls, ESM2_eos, and ESM2_segment0-9 features, use the reshape() function to restructure it into N*12*1280.
 ```python
 flattened_data = feature_all.reshape(-1, 1280)
 ```
(2) AE model design and pretraining (For detailed model information, refer to the [Autoencoder_pretraining](https://github.com/ywwy-qn/ESM2_AMP/blob/main/Feature%20attribution/Autoencoder_pretraining.py) code in the **Feature attribution** module). After obtaining the model weight file, input the data and set the hidden layer output to obtain the dimensionality-reduced features corresponding to the proteins.
 ```python
 with torch.no_grad():
     z, _ = ae_model(data_feature_tensor)
 z_npy = z.cpu().numpy()
 ```
(3) The obtained z_npy is a 12N * 150 dimensional vector, where the first dimension represents all feature types of all proteins, and the second dimension represents the 150-dimensional vector of each feature. This step restructures it so that the first dimension corresponds to each protein, resulting in an N * 1800 matrix, where 1800 is derived from 12*150.
 ```python
 combined_features_list = []
 for protein in tqdm(protein_names_li, desc="Processing Proteins", unit="protein"):
     related_rows = data_all[data_all['Name'].str.contains(protein)]
     if related_rows.empty:
         print(f"Warning: No rows found for protein {protein}.")
         continue

     suffixes = related_rows['Name'].str.split('_').str[-1]
     combined_features = {}

     for suffix in suffixes:
         matching_rows = related_rows[related_rows['Name'].str.endswith(suffix)]
         features = matching_rows.iloc[:, 1:].copy()
         num_features = features.shape[1]
         new_columns = [f"{suffix}_{i}" for i in range(num_features)]
         features.columns = new_columns
         for col in features.columns:
             if col not in combined_features:
                 combined_features[col] = features.iloc[0][col]
     combined_features_list.append(combined_features)
 ```
(4) Previously, we obtained dimensionality-reduced data at the protein level. Since the input for the random forest model requires features at the protein pair level, this step involves concatenating the dimensionality-reduced protein features based on the protein pair samples.
 ```python
 protein_pairs = pd.merge(protein_pairs, protein_features, how='left',
                                  left_on='Protein1', right_on='new_name')
 # Rename the column name
 new_column_names1 = []
 for name in protein_pairs.columns:
     if name == 'new_name':
         new_name = 'new_name1'
     elif name in ['Protein1', 'Protein2', 'Label', 'Pairs']:
         new_name = name
     else:
         new_name = '1_' + name
     new_column_names1.append(new_name)
 protein_pairs.columns = new_column_names1

 protein_pairs = pd.merge(protein_pairs, protein_features, how='left',
                                  left_on='Protein2', right_on='new_name')
 # Rename the column name
 new_column_names2 = []
 for name in protein_pairs.columns:
     if re.match('1_', name):
         new_name2 = name
     elif name in ['Protein1', 'Protein2', 'Label', 'Pairs', 'new_name1']:
         new_name2 = name
     elif name == 'new_name':
         new_name2 = 'new_name2'
     else:
         new_name2 = '2_' + name
     new_column_names2.append(new_name2)
 protein_pairs.columns = new_column_names2
 # delete redundant information
 protein_pairs = protein_pairs.drop(columns=['new_name1', 'new_name2'])
 ```
(5) Random Forest model design and train. Tree SHAP is an interpretability method that relies on decision tree models, and we adopt the Random Forest model. The model design code can be found in the RF file within the **Feature Attribution** module. The process of training the RF model and saving the model weights is as follows:
 ```python
 study = optuna.create_study(direction='maximize')
 # operation optimization
 study.optimize(objective, n_trials=30)
 # Get the best parameters
 best_params = study.best_params
 # The final model is trained using the optimal parameters
 best_model = RandomForestClassifier(**best_params, random_state=42)
 best_model.fit(X_train, y_train)
 # save model
 joblib.dump(best_model, outputPath + '/best_rf_model.pkl')

 # Save the best parameters to Excel file
 df_best_params = pd.DataFrame([best_params])
 df_best_params.to_excel(outputPath + '/best_rf_params.xlsx', index=False)

 # Save results of all trials to Excel file
 results_df.to_excel(outputPath + '/trials_results.xlsx', index=False)
 ```
(6) Tree SHAP feature importance calculation. The dimensionality-reduced protein pair data obtained above will be input into the trained RF model for inference, serving as the underlying model to calculate the SHAP values.
 ```python
 import shap
 import matplotlib.pyplot as plt
 from joblib import load

 # Initialize SHAP explainer and calculate shap_values
 def init_shap_analysis(model, X_train):
     """Initialize the SHAP TreeExplainer and return shap values."""
     # Select a random subset of the data for SHAP background (speed optimization)
     np.random.seed(0)
     background_index = np.random.choice(X_train.shape[0], size=200, replace=False)
     background_data = X_train.iloc[background_index].values
     # Initialize SHAP TreeExplainer
     explainer = shap.TreeExplainer(model, background_data)
     shap_values = explainer.shap_values(X_train)
     return shap_values
 # Model import
 model = load(r'D:\Aywwy\wy\other_dataset\TreeSHAP\AE_RF\rf_train_result\split_ae_rf_trat253570414\best_rf_model.pkl')
 # Initialize SHAP and calculate shap_values
 shap_values = init_shap_analysis(model, X_test)
 ```






### Related Works
If you are interested in feature extraction and model interpretation for large language models, you may find our previous work helpful:

- Interpretable feature extraction and dimensionality reduction in ESM2 for protein localization prediction: [Link](https://doi.org/10.1093/bib/bbad534); [GitHub Repositor](https://github.com/yujuan-zhang/feature-representation-for-LLMs)



**Important Note**: As the associated research papers are officially published, this project will continuously update and improve to better serve the scientific research community.
