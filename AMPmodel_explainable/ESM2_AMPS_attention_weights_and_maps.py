

import os
import numpy as np
import torch
import torch.nn as nn

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入模块
from AMPmodel.dataset import load_dataset
from AMPmodel.model import AMP_model
from AMPmodel.check import fix_state_dict
from attention_extract_function import extract_attention_weights
import numpy as np
from Attention_visual_map import Attention_visualization_diagram


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# 构建正确的文件路径
feature_file = os.path.join(project_root, "model_pred/data", "real_test_dataset_features.h5")
sample_file = os.path.join(project_root, "model_pred/data", "real_test_dataset_samples.xlsx")

test_dataset = load_dataset(
    mode='segment',
    feature_file=feature_file,  # 使用绝对路径
    sample_file=sample_file
)


test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载模型
model = AMP_model(input_dim=1280, hidden1_dim=640, hidden2_dim=240, output_dim=1, encoder_type="transformer").float().to(device)
checkpoint = torch.load("model_pred/weights_file/ESM2_AMPS.pth")

if 'module' in list(checkpoint['model_state_dict'].keys())[0]:
    checkpoint['model_state_dict'] = fix_state_dict(checkpoint['model_state_dict'])

model.load_state_dict(checkpoint['model_state_dict'])



merged_test0, merged_test1, merged_test2, merged_test3, merged_test4, merged_test5 = extract_attention_weights(model, test_dataloader)


save_path = os.path.join('attention_weights/ESM2_AMPS')
os.makedirs(save_path, exist_ok=True)


np.save(save_path + '/ESM2_AMPS_merged_test_atten_average_layers0.npy', merged_test0)
np.save(save_path + '/ESM2_AMPS_merged_test_atten_average_layers1.npy', merged_test1)
np.save(save_path + '/ESM2_AMPS_merged_test_atten_average_layers2.npy', merged_test2)
np.save(save_path + '/ESM2_AMPS_merged_test_atten_average_layers3.npy', merged_test3)
np.save(save_path + '/ESM2_AMPS_merged_test_atten_average_layers4.npy', merged_test4)
np.save(save_path + '/ESM2_AMPS_merged_test_atten_average_layers5.npy', merged_test5)

print('Save Success!')




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



# #第0层


# # 计算平均值
# average0 = np.mean(merged_test0, axis=0)

# # 定义颜色
# custom_blue = (54/255., 125/255., 176/255.)

# # 创建颜色映射
# cmap_name = 'custom_blue'
# cm = LinearSegmentedColormap.from_list(cmap_name, [(1, 1, 1), custom_blue], N=256)
# # 使用Seaborn绘制热图
# plt.figure(figsize=(12, 10))
# sns.heatmap(average0, cmap=cm, annot=False, fmt=".1f", linewidths=.5, square=True, cbar=True,
#             xticklabels=label_name, yticklabels=label_name)  # annot=False表示不显示每个单元格的具体数值

# ax = plt.gca()  # 获取当前的axes实例
# ax.xaxis.tick_top()  # 将x轴的刻度线和标签移到顶部
# ax.xaxis.set_label_position('top')  # 设置x轴标签的位置到顶部
# plt.xticks(rotation=45, ha='left')
# plt.title('ESM2_AMP_CSE attention significant map in Layer1')

# total_plot_path = os.path.join('attention_weights/ESM2_AMP_CSE/total_samples_map')
# os.makedirs(total_plot_path, exist_ok=True)

# plt.savefig(total_plot_path + '/ESM2_AMP_CSE attention significant map in Layer1.pdf')

# plt.tight_layout()  # 调整布局以尽量减少重叠
# plt.show()








# ## Single sample
# Attention_visualization_diagram_single(merged_test0, 0, 29, label_name, 'ESM2_AMP_CSE')







