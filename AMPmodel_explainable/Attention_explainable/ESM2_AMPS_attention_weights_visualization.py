

import os
import numpy as np
import torch
import torch.nn as nn

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import Module
from Attention_visual_map import Attention_visualization_diagram



open_path = os.path.join('AMPmodel_explainable/Attention_explainable/attention_weights/ESM2_AMPS')

# attention weights
layer0_test_positive = np.load(open_path + '/positive_infer_weights/positive_segment_merged_test_atten_average_layers0.npy')
layer0_test_negative = np.load(open_path + '/negative_infer_weights/negative_segment_merged_test_atten_average_layers0.npy')
merged_test0 = np.concatenate((layer0_test_positive, layer0_test_negative), axis=0)

layer1_test_positive = np.load(open_path + '/positive_infer_weights/positive_segment_merged_test_atten_average_layers1.npy')
layer1_test_negative = np.load(open_path + '/negative_infer_weights/negative_segment_merged_test_atten_average_layers1.npy')
merged_test1 = np.concatenate((layer1_test_positive, layer1_test_negative), axis=0)

layer2_test_positive = np.load(open_path + '/positive_infer_weights/positive_segment_merged_test_atten_average_layers2.npy')
layer2_test_negative = np.load(open_path + '/negative_infer_weights/negative_segment_merged_test_atten_average_layers2.npy')
merged_test2 = np.concatenate((layer2_test_positive, layer2_test_negative), axis=0)

layer3_test_positive = np.load(open_path + '/positive_infer_weights/positive_segment_merged_test_atten_average_layers3.npy')
layer3_test_negative = np.load(open_path + '/negative_infer_weights/negative_segment_merged_test_atten_average_layers3.npy')
merged_test3 = np.concatenate((layer3_test_positive, layer3_test_negative), axis=0)

layer4_test_positive = np.load(open_path + '/positive_infer_weights/positive_segment_merged_test_atten_average_layers4.npy')
layer4_test_negative = np.load(open_path + '/negative_infer_weights/negative_segment_merged_test_atten_average_layers4.npy')
merged_test4 = np.concatenate((layer4_test_positive, layer4_test_negative), axis=0)

layer5_test_positive = np.load(open_path + '/positive_infer_weights/positive_segment_merged_test_atten_average_layers5.npy')
layer5_test_negative = np.load(open_path + '/negative_infer_weights/negative_segment_merged_test_atten_average_layers5.npy')
merged_test5 = np.concatenate((layer5_test_positive, layer5_test_negative), axis=0)



## attention weights plot
# Total samples


label_name = ['A_segment0', 'A_segment1', 'A_segment2', 'A_segment3', 'A_segment4', 'A_segment5', 
              'A_segment6', 'A_segment7', 'A_segment8', 'A_segment9', 'B_segment0', 'B_segment1', 
              'B_segment2', 'B_segment3', 'B_segment4', 'B_segment5', 'B_segment6', 'B_segment7', 
              'B_segment8', 'B_segment9']






# Initialize with your data
visualizer = Attention_visualization_diagram(
    merged_test=merged_test0,
    i=0,
    label_name=label_name,
    model_name='ESM2_AMPS'
)
# Plot average attention
visualizer.plot_total()
# Plot attention for single sample (e.g., sample 30) The samples' number from 0, then m=29
visualizer.plot_single(m=29)

# Initialize with your data
visualizer = Attention_visualization_diagram(
    merged_test=merged_test1,
    i=1,
    label_name=label_name,
    model_name='ESM2_AMPS'
)
# Plot average attention
visualizer.plot_total()
# Plot attention for single sample (e.g., sample 30) The samples' number from 0, then m=29
visualizer.plot_single(m=29)


# Initialize with your data
visualizer = Attention_visualization_diagram(
    merged_test=merged_test2,
    i=2,
    label_name=label_name,
    model_name='ESM2_AMPS'
)
# Plot average attention
visualizer.plot_total()
# Plot attention for single sample (e.g., sample 30) The samples' number from 0, then m=29
visualizer.plot_single(m=29)



# Initialize with your data
visualizer = Attention_visualization_diagram(
    merged_test=merged_test3,
    i=3,
    label_name=label_name,
    model_name='ESM2_AMPS'
)
# Plot average attention
visualizer.plot_total()
# Plot attention for single sample (e.g., sample 30) The samples' number from 0, then m=29
visualizer.plot_single(m=29)



# Initialize with your data
visualizer = Attention_visualization_diagram(
    merged_test=merged_test4,
    i=4,
    label_name=label_name,
    model_name='ESM2_AMPS'
)
# Plot average attention
visualizer.plot_total()
# Plot attention for single sample (e.g., sample 30) The samples' number from 0, then m=29
visualizer.plot_single(m=29)



# Initialize with your data
visualizer = Attention_visualization_diagram(
    merged_test=merged_test5,
    i=5,
    label_name=label_name,
    model_name='ESM2_AMPS'
)
# Plot average attention
visualizer.plot_total()
# Plot attention for single sample (e.g., sample 30) The samples' number from 0, then m=29
visualizer.plot_single(m=29)

print('Save Map Success!')



