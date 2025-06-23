

# ESM2_AMP: an Interpretable Framework for Protein-protein Interactions Prediction and Biological Mechanism Discovery

## Introduction

This project revolves around the paper titled "**ESM2_AMP: an Interpretable Framework for Protein-protein Interactions Prediction and Biological Mechanism Discovery**", aiming to provide relevant datasets and model resources. Through our framework, we hope to enhance the understanding of protein interactions and their underlying biological mechanisms, offering robust support for research in related fields.

### Author Contact Information:

- Author 1: Yawen Sun, Email: [2108437154@qq.com](mailto:2108437154@qq.com)
- Author 2: Rui Wang, Email: [2219312248@qq.com](mailto:2219312248@qq.com)
- Author 3: Zeyu Luo, Email: [1024226968@qq.com](mailto:1024226968@qq.com)

Your contributions, feedback, and suggestions are highly appreciated. If you encounter any issues or have questions, feel free to reach out to the authors via the provided email addresses. Thank you for your interest in our work!

## Work Environment Setup

For the environment configuration of other ESM2_AMP modules
 1. Clone this repository or download the project files.
```bash  
# Clone the project  
git clone https://github.com/ywwy-qn/ESM2_AMP.git  

```
 2. Navigate to the project directory.
```bash 
cd your_path_to_project/ESM2_AMP # Navigate to the project directory  
```
 
 4. Create a new Conda environment with Python version >= 3.11, then activate the environment:
```bash
conda create -n esm2_amp-env python=3.11
conda activate esm2_amp-env
```

 4. For CPU-only setup (if you don't need GPU acceleration):
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

 5. (Optional) To enable GPU acceleration with CUDA (e.g., CUDA 11.8), please first install the necessary dependencies via Conda:
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```
 After successfully installing the dependencies, install the current package with:
```bash
pip install .
```
**Note:** In step5, a matching torch version needs to be installed based on the user's own cuda version. The PyTorch link is [PyTorch](https://pytorch.org/get-started/previous-versions/)

## Dataset Availability

This project provides training datasets and independent test sets for researchers to explore further. All data resources can be accessed in the **data** folder within each subdirectory, with detailed features available on [figshare](https://figshare.com/articles/dataset/ESM2_AMP/28378157) Detailed information on methods for extracting feature representations from protein sequences can be found in the published research paper (DOI: https://doi.org/10.1093/bib/bbad534), as well as in the corresponding GitHub repository: [Feature Representation for LLMs](https://github.com/yujuan-zhang/feature-representation-for-LLMs?tab=readme-ov-file#feature-representation-model). For detailed information on the extraction of feature embeddings for specific proteins, please refer to the Python library protloc-mex-x (https://pypi.org/project/protloc-mex-x/).

Model founding: 
This project provides ready-to-use implementations of the **ESM2_AMPS**, **ESM2_AMP_CSE**, **ESM2_DPM**, and **ESM2_GRU** models (ensure your environment meets the upon requirements). 

In the **model** script in the AMPmodel directory, the encoder_type is set with three options. 

When encoder_type == 'transformer', the ESM2_AMPS and ESM2_AMP_CSE models with transformer are built.

The design differences between the ESM2_AMPS and ESM2_AMP_CSE models mainly lie in the data selection and the dimensions of the dataset, which are primarily reflected in the **AMPmodel/dataset** script.

ESM2_DPM: The global pooling features of the sequence are selected, referred to as ESM2_mean, with a **DNN** used as the downstream classifier.

ESM2_GRU: The transformer module is replaced with **GRU**. When encoder_type == 'gru', it becomes the ESM2_GRU model.


The function encapsulation in the **AMPmodel/check** script for the evalution of the model and the saving of parameters during training, etc.

## model prediction
During the model prediction phase, multiple metrics such as **Accuracy**, **MCC**, **Recall**, **F1 score**, and **Precision** are used to assess the model's performance.The evaluation metrics and calculation methods are shown in [code](https://github.com/ywwy-qn/ESM2_AMP/AMPmodel/check.py). The implementation code of the model prediction is in the **model_pred** directory, and the implementation process is as follows:

### ESM2_AMPS
```bash
python model_pred/ESM2_AMPS_pred.py
```
### ESM2_AMP_CSE
```bash
python model_pred/ESM2_AMP_CSE_pred.py
```
### ESM2_DPM
```bash
python model_pred/ESM2_DPM_pred.py
```

## Comparison_model
During this process, we trained, validated, and tested the ESM2_AMPS and ESM2_GRU models on the **Bernett dataset**.

We have provided the trained model weight files on [figshare](https://figshare.com/articles/dataset/ESM2_AMP/28378157) for prediction purposes. These two weight files are saved as **5.ESM2_AMPS_Bernett.zip** and **6.ESM2_GRU_Bernett.zip**, respectively.

The testing script is as follows:
```bash
python Comparison_model/ESM2_AMPS_Bernett/Train_test.py
python Comparison_model/ESM2_GRU_Bernett/Train_test_GRU.py
```

We have also provided the **training** code for implementation. After running the script, the model weight files will be saved in the following directories:

-Comparison_model/ESM2_AMPS_Bernett/weights_file/bestcheckpoint

-Comparison_model/ESM2_GRU_Bernett/weights_file/bestcheckpoint

The training and validation scripts are as follows:
```bash
python Comparison_model/ESM2_AMPS_Bernett/Train_val.py
python Comparison_model/ESM2_GRU_Bernett/Train_val_GRU.py
```


## AMPmodel_explainable

1.Attention-based Explainable Analysis ([Attention_explainable](https://github.com/ywwy-qn/ESM2_AMP/tree/main/AMPmodel_explainable/Attention_explainable))
Both models within the **ESM2_AMP** framework utilize the multi-head attention mechanism of the Transformer encoder. By leveraging the weight matrix allocation in the multi-head attention mechanism, the attention weights corresponding to the sample features are extracted, and their feature importance is calculated and quantified.

### ESM2_AMPS model Attention weights and visualization
```bash
python AMPmodel_explainable/Attention_explainable/ESM2_AMPS_attention_weights_visualization.py
```

### ESM2_AMP_CSE model Attention weights and visualization
```bash
python AMPmodel_explainable/Attention_explainable/ESM2_AMP_CSE_attention_weights_visualization.py
```

2.[Integrated_Gradients](https://github.com/ywwy-qn/ESM2_AMP/tree/main/AMPmodel_explainable/Integrated_Gradients) for ESM2_AMPS and ESM2_AMP_CSE
The Integrated Gradients (IG) method was used to compute feature importance values for both models, followed by analysis.

### ESM2_AMPS model Integrated Gradients and visualization
```bash
python AMPmodel_explainable/Integrated_Gradients/ESM2_AMPS_IG_attribution.py
```

### ESM2_AMP_CSE modelIntegrated Gradients and visualization
```bash
python AMPmodel_explainable/Integrated_Gradients/ESM2_AMP_CSE_IG_attribution.py
```
**Note:** Since the baseline is calculated by randomly selected samples when computing the IG values, the final IG values may vary. However, the fundamental point we aim to convey remains valid.

## Feature attribution
Including [segment0-9](https://github.com/ywwy-qn/ESM2_AMP/tree/main/Feature_attribution/segment_0_9) features and [cls_segment_eos](https://github.com/ywwy-qn/ESM2_AMP/tree/main/Feature_attribution/cls_segment_eos) features.

#### model constructing

The features obtained from ESM2 are fed into an autoencoder for dimensionality reduction to derive a new feature representation, which is then input into a random forest model. This process is named AE_RF and serves as the underlying model for Tree SHAP.

**AE pretraining**
We performed dimensionality reduction on the extracted ESM2 protein feature representations, which originally had 1280 dimensions per feature, with the Autoencoder's hidden layer dimension set to 150. It is necessary to restructure the original feature dimensions to fit the input requirements of the Autoencoder.

1.Taking the original 2D matrix of size N15360 as an example, where N is the number of proteins and 15360 is the sum of the 1280 dimensions of `ESM2_cls`, `ESM2_eos`, and `ESM2_segment0-9` features, use the `reshape()` function to restructure it into N * 12 * 1280.

 ```python
 flattened_data = feature_all.reshape(-1, 1280)
 ```
2.AE model design (For detailed model information, refer to the [Autoencoder_model](https://github.com/ywwy-qn/ESM2_AMP/blob/main/Feature_attribution/Autoencoder_model.py) code in the **Feature_attribution** module). After obtaining the model weight file, input the data and set the hidden layer output to obtain the dimensionality-reduced features corresponding to the proteins.

 ```python
 with torch.no_grad():
     z, _ = ae_model(data_feature_tensor)
 z_npy = z.cpu().numpy()
 ```
3.The obtained z_npy is a 12N * 150 dimensional vector, where the first dimension represents all feature types of all proteins, and the second dimension represents the 150-dimensional vector of each feature. This step restructures it so that the first dimension corresponds to each protein, resulting in an N * 1800 matrix, where 1800 is derived from 12 * 150.

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
4.Previously, we obtained dimensionality-reduced data at the protein level. Since the input for the random forest model requires features at the protein pair level, this step involves concatenating the dimensionality-reduced protein features based on the protein pair samples.

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
5.Random Forest model design and train. Tree SHAP is an interpretability method that relies on decision tree models, and we adopt the Random Forest model. The process of training the RF model and saving the model weights is as follows:

 ```python
 study = optuna.create_study(direction='maximize')
 # operation optimization
 study.optimize(objective, n_trials=30)
 # Get the best parameters
 best_params = study.best_params
 # The final model is trained using the optimal parameters
 best_model = RandomForestClassifier(**best_params, random_state=42)
 best_model.fit(X_train, y_train)
 joblib.dump(best_model, outputPath + '/best_rf_model.pkl')

 # Save the best parameters to Excel file
 df_best_params = pd.DataFrame([best_params])
 df_best_params.to_excel(outputPath + '/best_rf_params.xlsx', index=False)

 # Save results of all trials to Excel file
 results_df.to_excel(outputPath + '/trials_results.xlsx', index=False)
 ```
6.Tree SHAP feature importance calculation. The dimensionality-reduced protein pair data obtained above will be input into the trained RF model for inference, serving as the underlying model to calculate the SHAP values.

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

7. DNN model design. Integrated Gradient is an interpretability method that relies on decision tree models, and we adopt the DNN model. The model design code can be found in the DNN file within the **Feature Attribution** module.
 ```python
 class DNN(nn.Module):
    def __init__(self, input_dim=3000, hidden1_dim=1500, hidden2_dim=500, output_dim=1):
        super(DNN, self).__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, output_dim))
        
    def forward(self, X):
        return self.mlp_layers(X)

class CustomDataset(Dataset):
    def __init__(self, data_features, data_label=None):
        self.features = torch.tensor(data_features.values).float()
        self.has_labels = data_label is not None
        if self.has_labels:
            self.labels = torch.tensor(data_label).float()
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.has_labels:
            return self.features[idx], self.labels[idx]
        return self.features[idx]
 ```
8. Integrated Gradient Calculate.
```python
class IGFeatureImportance:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.ig = IntegratedGradients(model)
    
    def compute_attributions(self, X_data, target_label=None):
        """
        Compute IG attributions using random 200 samples as baseline.
        
        Parameters:
        - X_data: Input data (pd.DataFrame)
        - target_label: Target label for classification models (int)
        
        Returns:
        - attributions: Computed attributions (np.ndarray)
        """
        X_tensor = torch.tensor(X_data.values, dtype=torch.float32).to(self.device)
        
        # Randomly select 200 samples as baseline
        np.random.seed(0)
        background_index = np.random.choice(X_tensor.shape[0], size=200, replace=False)
        baseline_data = X_tensor[background_index].mean(dim=0, keepdim=True).to(self.device)
        
        # Compute attributions
        attributions = self.ig.attribute(
            inputs=X_tensor,
            baselines=baseline_data,
            target=target_label,
            n_steps=50,
            internal_batch_size=64
        )
        
        return attributions.detach().cpu().numpy()

 def group_by_prefix(attributions, feature_names):

        prefixes = []
        for f in feature_names:
            parts = f.split('_')
            if parts[0] == '1':
                parts[0] = 'A'
            elif parts[0] == '2':
                parts[0] = 'B'
            prefixes.append("_".join(parts[:2]))

        unique_prefixes = sorted(set(prefixes))

        grouped_attributions = np.zeros((attributions.shape[0], len(unique_prefixes)))
        for j, prefix in enumerate(unique_prefixes):
            cols = [i for i, f in enumerate(feature_names)
                    if f.startswith(prefix.replace("A_", "1_").replace("B_", "2_"))]
            grouped_attributions[:, j] = np.mean(np.abs(attributions[:, cols]), axis=1)

        return grouped_attributions, unique_prefixes
```
**The following are the implementations of several specific feature attribution methods.**
### segment0-9 (It can be compared with the interpretable results of the ESM2_AMPS model)
Including AE_RF_Gini, AE_RF_SHAP, AE_DNN_SHAP, AE_DNN_IG.

1.AE_infer
The original features are dimensionally reduced by AE to obtain new features.
```bash
python Feature_attribution/segment_0_9/segment_ae_infer.py
```
After running this line of code, the new features will be saved under the path 'Feature_attribution/segment_0_9/AE_output' for use in subsequent steps.

2.AE_DNN_SHAP
Using DNN as the base classifier, the SHAP feature attribution method is employed to compute feature importance.
```bash
python Feature_attribution/segment_0_9/AE_DNN_SHAP/segment_AE_DNN_SHAP_code.py
```
After running this line of code, the plot will be saved under the path 'Feature_attribution/segment_0_9/AE_DNN_SHAP/output'.

3.AE_DNN_IG
Using DNN as the base classifier, the IG feature attribution method is employed to compute feature importance.
```bash
python Feature_attribution/segment_0_9/AE_DNN_IG/segment_AE_DNN_IG_code.py
```
After running this line of code, the plot will be saved under the path 'Feature_attribution/segment_0_9/AE_DNN_IG/output'.

4.AE_RF_SHAP
Using RF as the base classifier, the SHAP feature attribution method is employed to compute feature importance.
```bash
python Feature_attribution/segment_0_9/AE_RF_SHAP/segment_AE_RF_SHAP_bar_plot_code.py
```
After running this line of code, the plot will be saved under the path 'Feature_attribution/segment_0_9/AE_RF_SHAP/output'.

5.AE_RF_Gini
Using RF as the base classifier, calculate gini importance.
```bash
python Feature_attribution/segment_0_9/AE_RF_Gini/segment_AE_RF_Gini_bar_plot_code.py
```
After running this line of code, the plot will be saved under the path 'Feature_attribution/segment_0_9/AE_RF_Gini/output'.


### cls_segment_eos (It can be compared with the interpretable results of the ESM2_AMP_CSE model)
Also including AE_RF_Gini, AE_RF_SHAP, AE_DNN_SHAP, AE_DNN_IG.

1.AE_infer
```bash
python Feature_attribution/cls_segment_eos/cls_segment_eos_ae_infer.py
```

2.AE_DNN_SHAP
```bash
python Feature_attribution/cls_segment_eos/AE_DNN_SHAP/cls_segment_eos_AE_DNN_SHAP_code.py
```

3.AE_DNN_IG
```bash
python Feature_attribution/cls_segment_eos/AE_DNN_IG/cls_segment_eos_AE_DNN_IG_code.py
```

4.AE_RF_SHAP
```bash
python Feature_attribution/cls_segment_eos/AE_RF_SHAP/cls_segment_eos_AE_RF_SHAP_bar_plot_code.py
```

5.AE_RF_Gini
```bash
python Feature_attribution/cls_segment_eos/AE_RF_Gini/cls_segment_eos_AE_RF_Gini_bar_plot_code.py
```

## Identification_computational_methods_of_functional_AA_regions
To explore the potential association between feature attention weights and specific residues or residue regions, this study conducted a detailed analysis of the **ESM2_AMPS** model, which relies solely on local features of fragments. Based on samples from the **real_test** dataset, the top three features with the highest weight values in each sample were first identified, and the proportion of their coverage of functional amino acid sequences was calculated. Meanwhile, the three features with the lowest weights were selected as a negative control group for comparison. Information on functional amino acid regions for all proteins in the dataset was obtained from the **UniProt** and **InterPro** databases.

- When analyzing based on the **UniProt** database, the selected functional amino acid region types included: **"Domain"**, **"Region"**, **"Compositional bias"**, **"Repeat"**, and **"Motif"**.
- When analyzing based on the **InterPro** database, the selected functional amino acid region types included: **"Domain"**, **"Repeat"**, **"Active_site"**, **"Binding_site"**, **"Conserved_site"**, and **"Ptm"** (Note: **"Homologous_superfamily"** and **"Family"** were not included in the calculations, as they belong to classification-level special sequence fragments).

Here is the code [detail](https://github.com/ywwy-qn/ESM2_AMP/tree/main/Identification_computational_methods_of_functional_AA_regions/code).

This section focuses on quantifying the attention weights of the ESM2_AMPS model that only contains segment features, followed by detailed calculations of the coverage and hit rates of functional amino acid regions in each segment of the samples in the independent test set.

We have provided samples in the **data** directory categorized into four types based on prediction results (true positives, false positives, false negatives, and true negatives). For each of these categories, we calculate the coverage and hit rates. The **Total_information_protein** directory contains functional region information for protein amino acid sequences recorded in public databases. 
The **code** directory contains the relevant code for this section:

### step1: Calculate the coverage rate.
This step consists of two scripts: one for calculating the coverage of the Top three segments and another for the Low three segments. These scripts compute both the overall functional amino acid region coverage and the coverage for five specific types: "Domain", "Region", "Compositional bias", "Repeat", and "Motif".
```bash
python Identification_computational_methods_of_functional_AA_regions/code/coverage_Top_computational_process.py
python Identification_computational_methods_of_functional_AA_regions/code/coverage_Low_computational_process.py
```
After running the scripts, the results will be saved in Excel format in the **database_uniprot_out** directory.

### step2: Calculate the hit rate.
Based on the coverage calculated in the previous step, different thresholds are set to determine whether samples in each category (TP, FP, TN, FN) hit functional regions, and the hit rate of samples in each category is computed. Similarly, there are two separate scripts for Top and Low segments.
```bash
python Identification_computational_methods_of_functional_AA_regions/code/calculation_of_Top_hit_rate.py
python Identification_computational_methods_of_functional_AA_regions/code/calculation_of_Low_hit_rate.py
```
After running the scripts, the results will be saved in Excel format in the **database_uniprot_out** directory.

### step3: Visualization.
Based on the obtained coverage results, the data is divided into ten coverage intervals for visualization. The visualization is implemented through two separate scripts:

1. visualization.py - Generates plots specifically for either Top three or Low three segments
2. visualization2.py - Creates comparative visualizations between Top and Low segments
```bash
python Identification_computational_methods_of_functional_AA_regions/code/visualization.py
python Identification_computational_methods_of_functional_AA_regions/code/visualization2.py
```
After running the scripts, the plots will be saved in Pdf format in the **database_uniprot_out** directory.

## Feature representation
Additionally, we provide code in the **esm2_infer_feature** directory that enables feature extraction from protein amino acid sequences using the **ESM2** model. 

The proteins' feature representation used the pre-trained protein model ESM2 developed by Meta company and placed on Hugging Face. For more details, please search in https://huggingface.co/facebook/esm2_t33_650M_UR50D. Besides, we used [protloc-mex-x](https://pypi.org/project/protloc_mex_X/) which our team developed, containing detail for `'cls'`,`'mean'`, `'eos'`,`'segment 0-9'` feature representation from ESM2. For ease of use, we have included sample data in the **data** directory as examples, along with the corresponding scripts to perform the feature extraction.
```bash
python esm2_infer_feature/esm2_infer_features_example.py
```
After running the scripts, the features' file will be saved in the **output** directory.

**Note:** If you want to extract the features of your proteins, you can replace the sample data in **data** with your own data and then run the script to achieve it.


**Important Notes**:  
  
  • For model inference, the weight file could be foud on [figshare](https://figshare.com/articles/dataset/ESM2_AMP/28378157).



### Model training

During model training, Optuna is primarily employed for **Bayesian optimization-based hyperparameter selection using the Tree-structured Parzen Estimator (TPE) algorithm**, with key details as follows:

The training process code is in the [Model_work](https://github.com/ywwy-qn/ESM2_AMP/tree/main/Model_work) directory.


- In **ESM2_AMPS** model training process, the learning rate was tuned within the range of 1e-5 to 1e-3, while the weight decay was adjusted between 1e-4 and 1e-2. For the MLP module, the first hidden layer size was varied from 480 to 640 with a step size of 160, and the second hidden layer size was explored from 80 to 320 with a step size of 80. Here are the code [details](https://github.com/ywwy-qn/ESM2_AMP/blob/main/Model_work/ESM2_AMPS/optuna_train_5fold.py) and [result](https://github.com/ywwy-qn/ESM2_AMP/blob/main/model_pred/ESM2_AMPS_config.yaml).
- **ESM2_AMP_CSE** model maintained these parameters but extended the weight decay range to 1e-4-1e-1. Here are the code details and [result](https://github.com/ywwy-qn/ESM2_AMP/blob/main/model_pred/ESM2_AMP_CSE_config.yaml).
- In **ESM2_DPM** model training process, the learning rate was tuned within the range of 1e-6 to 1e-5, while the weight decay was adjusted between 1e-3 and 1e-1. For the DNN module, the first hidden layer size varied from 960 to 1280 with a step size of 320, the second hidden layer size was explored from 320 to 640 with a step size of 160, and the last layer was 40 to 160 with 60 steps. Here are the code details and [result](https://github.com/ywwy-qn/ESM2_AMP/blob/main/model_pred/ESM2_DPM_config.yaml).

Taking the training of the ESM2_AMPS model as an example, the following is the detailed code for parameter tuning using **optuna**.

``` python
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-4, 1e-2)
    hidden1_dim = trial.suggest_int('hidden1_dim', low=480, high=640, step=160)
    hidden2_dim = trial.suggest_int('hidden2_dim', low=80, high=320, step=80)
```


The specific training process is as follows:

1.ESM2_AMPS model training
``` bash
    python Model_work/ESM2_AMPS/optuna_train_5fold.py
```
The relevant results of the training will be saved in the **Model_work/ESM2_AMPS/output** directory.

2.ESM2_AMP_CSE model training
``` bash
    python Model_work/ESM2_AMP_CSE/optuna_train_5fold.py
```
The relevant results of the training will be saved in the **Model_work/ESM2_AMP_CSE/output** directory.

1.ESM2_DPM model training
``` bash
    python Model_work/ESM2_DPM/optuna_train_5fold.py
```
The relevant results of the training will be saved in the **Model_work/ESM2_DPM/output** directory.



**Note:** Due to the large size of the training dataset, considering the complexity of reading the data and the preprocessing steps, the processed features of protein pairs will be directly received here instead of performing preprocessing operations on the protein features. This will save a certain amount of time.

### Related Works

If you are interested in feature extraction and model interpretation for large language models, you may find our previous work helpful:

- Interpretable feature extraction and dimensionality reduction in ESM2 for protein localization prediction: [Link](https://doi.org/10.1093/bib/bbad534); [GitHub Repositor](https://github.com/yujuan-zhang/feature-representation-for-LLMs)

**Important Note**: As the associated research papers are officially published, this project will continuously update and improve to better serve the scientific research community.

