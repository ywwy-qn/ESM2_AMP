Model founding: including ESM2_AMPS, ESM2_AMP_CSE, ESM2_DPM, ESM2_GRU and the ablation experiment.

In the **model** script, the encoder_type is set with three options. 

When encoder_type == 'transformer', the ESM2_AMPS and ESM2_AMP_CSE models with transformer are built.

The design differences between the ESM2_AMPS and ESM2_AMP_CSE models mainly lie in the data selection and the dimensions of the dataset, which are primarily reflected in the **dataset** script.

ESM2_DPM: The global pooling features of the sequence are selected, referred to as ESM2_mean, with a **DNN** used as the downstream classifier.

ESM2_GRU: The transformer module is replaced with **GRU**. When encoder_type == 'gru', it becomes the ESM2_GRU model.

Ablation experiment: In the **ablation experiment**, we removed the transformer encoder part and only included the global pooling operation. Therefore, in this model, encoder_type == 'mean'.

The function encapsulation in the **check** script for the validation of the model and the saving of parameters during training, etc.
