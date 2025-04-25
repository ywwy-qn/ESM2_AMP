# Model

## 1. Models Based on Different Feature Selection

**ESM2_AMP** framework initiates by employing **ESM2** to extract features for each protein. These features are then fed into a **Transformer encoder**, which utilizes a self-attention mechanism to comprehensively integrate features from both proteins. This process effectively captures both intra-protein interaction relationships and inter-protein interaction patterns. Ultimately, the integrated features are passed through a **multilayer perceptron (MLP)** classifier to predict the likelihood of interaction between the protein pair.
Based on different feature integration methods and the ESM2_AMP framework, the ESM2_AMPS and ESM2_AMP_CSE models were developed. The ESM2_AMPS model relies solely on segment features, while the ESM2_AMP_CSE model incorporates both special token and segment features. Additionally, a contrast model, ESM2_DPM, was constructed by extracting only the global sequence pooling features of two proteins. These models aim to explore the roles of features in predicting protein-protein interactions (PPIs), with ESM2_AMPS focusing on local interactions and ESM2_AMP_CSE integrating information. ESM2_DPM serves as a baseline to evaluate the performance of global features independently.

### 1.1 ESM2_AMPS Model: Dependent on Segment Features

In the ESM2_AMPS model, the segment features of protein pairs obtained by fine-tuning the ESM2 protein large language model are selected and labeled as `"A_segment0-9"` and `"B_segment0-9"`. These features are concatenated into a matrix $E \in \mathbb{R}^{20 \times 1280}$ and fed into a Transformer encoder. The self-attention mechanism within the encoder fuses these features, capturing intra- and inter-protein interaction patterns. For more details, see [work details](https://github.com/ywwy-qn/ESM2_AMP/blob/main/Models/ESM2_AMPS/ESM2_AMPS%20model%20Code.py).

### 1.2 ESM2_AMP_CSE Model: Dependent on Special Token and Segment Features

The ESM2_AMP_CSE model combines segment features (`ESM2_segment0-9`) with special token features (`ESM2_cls` and `ESM2_eos`). The features are concatenated in the following order: `A_cls`, `A_segment0-9`, `A_eos`, followed by `B_cls`, `B_segment0-9`, `B_eos`. The feature matrix for each sample input to the Transformer encoder is $E \in \mathbb{R}^{24 \times 1280}$. Subsequently, the features are fused through the self-attention mechanism. For more details, see [work details](https://github.com/ywwy-qn/ESM2_AMP/blob/main/Models/ESM2_AMP_CSE/ESM2_AMP_CSE%20model%20Code.py).

### 1.3 Contrast Experiment: ESM2_DPM Model Dependent Only on Global Pooling Features

Additionally, the contrast model (ESM2_DPM) uses only the global features of proteins A and B (`ESM2_mean`), in the order of `A_mean` and `B_mean`. These features are input into a deep neural network (DNN) to predict protein-protein interactions, with each sample's input feature dimension being $2 \times 1280$. This serves as a baseline to evaluate the impact of local versus global features. For more details, see [work details](https://github.com/ywwy-qn/ESM2_AMP/blob/main/Models/ESM2_DPM/ESM2_DPM%20model%20Code.py).

---

