# from .model import AMP_model
# from .dataset import AMPDataset
# from .train import train_model
# from .check import model_check




from AMPmodel.model import AMP_model
from AMPmodel.dataset import load_dataset
from AMPmodel.check import fix_state_dict, evaluate_model


__all__ = ['AMP_model', 'load_dataset', 'fix_state_dict', 'evaluate_model']
