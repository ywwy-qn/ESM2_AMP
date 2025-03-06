

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
'''python
tokenizer = AutoTokenizer.from_pretrained(modelPath + "/esm2_t33_650M_UR50D")
model = AutoModelForMaskedLM.from_pretrained(modelPath + "/esm2_t33_650M_UR50D", output_hidden_states=True)
protein_sequence_df = pd.read_excel(excel_file_path)  # /'cpu', simply replace 'auto' with 'cuda'/'cpu'
feature_extractor = Esm2LastHiddenFeatureExtractor(tokenizer, model,
                                                   compute_cls=True, compute_eos=True, compute_mean=True,
                                                   compute_segments=True)
print(f"{feature_extractor.device}")

Then, we used function pd.merge() from the pandas Python package to concatenate data based on the different features required by each model. Taking the segment local features required by the ESM2_AMPS model as an example:



The initial protein pair features input to the Transformer encoder are constructed using the following method through a DataLoader. For each sample (i.e., a protein pair), the features are organized into a 2D matrix based on their different characteristics. If N features are selected, each feature has a dimensionality of 1280, resulting in a feature matrix of size N*1280 for each sample.

### Models

This project encompasses a series of models, including [ESM2_AMPS](https://github.com/ywwy-qn/ESM2_AMP/tree/main/Models/ESM2_AMPS), [ESM2_AMP_CSE](https://github.com/ywwy-qn/ESM2_AMP/tree/main/Models/ESM2_AMP_CSE), and [ESM2_DPM](https://github.com/ywwy-qn/ESM2_AMP/tree/main/Models/ESM2_DPM), aimed at providing comprehensive support for predicting protein interactions. Example inference code for each model is provided within their respective directories, while the required model weight files can be downloaded from the project's corresponding [figshare](https://figshare.com/articles/dataset/ESM2_AMP/28378157) page.

### Model training

During model training, the **AdamW** algorithm and **Optuna** are used for hyperparameter tuning to reduce sensitivity to the selection of parameters such as learning rate and weight decay, while GPU acceleration is employed to speed up model training. The **ReLU** activation function is adopted to accelerate model training and achieve better prediction results. Additionally, **He initialization** is applied to mitigate gradient vanishing and explosion issues, thereby improving the model's convergence speed.

### Model evaluation

During the model evaluation phase, multiple metrics such as **Accuracy**, **MCC**, **Recall**, **F1 score**, and **Precision** are used to assess the model's performance.The evaluation metrics and calculation methods are shown in Equation: 

![529764fd-a8fb-4fa6-a409-2c1275bc97bf](https://github.com/user-attachments/assets/295bc8cb-6ae6-406c-8faf-c84f404d42c7)


### Attention-based Explainable Analysis

Both models within the ESM2_AMP framework utilize the multi-head attention mechanism of the Transformer encoder. By leveraging the weight matrix allocation in the multi-head attention mechanism, the attention weights corresponding to the sample features are extracted, and their feature importance is calculated and quantified. The detailed process will be updated in subsequent article publications.

### Feature attribution interpretable method

#### model constructing
The features obtained from ESM2 are fed into an autoencoder for dimensionality reduction to derive a new feature representation, which is then input into a random forest model. This process is named AE_RF and serves as the underlying model for Tree SHAP.

**AE pretraining**

### Related Works
If you are interested in feature extraction and model interpretation for large language models, you may find our previous work helpful:

- Interpretable feature extraction and dimensionality reduction in ESM2 for protein localization prediction: [Link](https://doi.org/10.1093/bib/bbad534); [GitHub Repositor](https://github.com/yujuan-zhang/feature-representation-for-LLMs)



**Important Note**: As the associated research papers are officially published, this project will continuously update and improve to better serve the scientific research community.
