

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
The proteins' feature representation used the pre-trained protein model ESM2 developed by Meta company and placed on Hugging Face. For more details, please search in https://huggingface.co/facebook/esm2_t33_650M_UR50D. Besides, we used [protloc-mex-x](https://pypi.org/project/protloc_mex_X/) which our team developed, containing detail for 'cls','mean', 'eos','segment 0-9','pho' feature representation from ESM2.
Then, we used function(pd.merge) from pandas python library.

The initial protein pair features input to the Transformer encoder are constructed using the following method through a DataLoader. For each sample (i.e., a protein pair), the features are organized into a 2D matrix based on their different characteristics. If N features are selected, each feature has a dimensionality of 1280, resulting in a feature matrix of size N*1280 for each sample.

### Models

This project encompasses a series of models, including [ESM2_AMPS](https://github.com/ywwy-qn/ESM2_AMP/tree/main/Models/ESM2_AMPS), [ESM2_AMP_CSE](https://github.com/ywwy-qn/ESM2_AMP/tree/main/Models/ESM2_AMP_CSE), and [ESM2_DPM](https://github.com/ywwy-qn/ESM2_AMP/tree/main/Models/ESM2_DPM), aimed at providing comprehensive support for predicting protein interactions. Example inference code for each model is provided within their respective directories, while the required model weight files can be downloaded from the project's corresponding [figshare](https://figshare.com/articles/dataset/ESM2_AMP/28378157) page.

### Model training

During model training, the **AdamW** algorithm and **Optuna** are used for hyperparameter tuning to reduce sensitivity to the selection of parameters such as learning rate and weight decay, while GPU acceleration is employed to speed up model training. The **ReLU** activation function is adopted to accelerate model training and achieve better prediction results. Additionally, **He initialization** is applied to mitigate gradient vanishing and explosion issues, thereby improving the model's convergence speed.

### Model evaluation

During the model evaluation phase, multiple metrics such as **Accuracy**, **MCC**, **Recall**, **F1 score**, and **Precision** are used to assess the model's performance.The evaluation metrics and calculation methods are shown in Equation: 
     ■(Accuracy =(TP+TN)/(TP+TN+FP+FN)@Recall=TP/(TP+FN)@Precision= TP/(TP+FP)@F1=2×(Precision×Recall)/(Precision+Recall)@MCC=(TP×TN-FP×FN)/√((TP+FP)(TP+FN)(TN+FP)(TN+FN) ))




**Important Note**: As the associated research papers are officially published, this project will continuously update and improve to better serve the scientific research community.
