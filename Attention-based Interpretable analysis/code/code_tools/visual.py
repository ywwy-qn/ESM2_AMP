import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle



def read_excel_file(file_path, required_columns):
    """
    读取Excel文件并验证所需列是否存在且为数值类型。
    """
    if not os.path.exists(file_path):
        print(f"文件路径不存在: {file_path}")
        sys.exit(1)
    
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        sys.exit(1)
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"缺少以下指定列: {missing_columns}")
        sys.exit(1)
    
    # 确保所需列为数值类型，非数值的转换为NaN
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def bin_data(df, columns, bins, bin_labels):
    """
    对指定列的数据进行分箱处理，并计算每个箱的计数。
    """
    counts = {}
    for col in columns:
        # 去除NaN值，避免错误计数
        series = df[col].dropna()
        # 使用pd.cut进行分箱，include_lowest确保第一个区间包含最小值
        binned = pd.cut(series, bins=bins, labels=bin_labels, right=True, include_lowest=True)
        counts[col] = binned.value_counts().sort_index()
    return counts

def plot_donut_charts(counts, bin_labels, title):
    """
    绘制环形饼图，展示各列数据的分布情况。
    """
    required_columns = list(counts.keys())
    num_layers = len(required_columns)
    fig, ax = plt.subplots(figsize=(10, 8))
    radius = 0.8  # 外层半径
    width = 0.2  # 每层饼图的宽度

    colors = plt.cm.tab20(np.linspace(0, 1, len(bin_labels)))  # 为不同区间分配颜色

    for i, col in enumerate(required_columns):
        data_to_plot = counts[col]
        # 计算当前层的半径
        current_radius = radius - i * width
        wedges, texts, autotexts = ax.pie(
            data_to_plot,
            labels=None,
            autopct='%1.1f%%',
            startangle=90,
            radius=current_radius,
            pctdistance= (current_radius-width/2)/current_radius,
            wedgeprops=dict(width=width, edgecolor='white'),
            colors=colors
        )
    
        # 添加中央注释
        ax.text(0, current_radius-width/2, col.replace('_colate', ''), ha='center', va='center', fontsize=13, fontweight='bold')
    
    # 添加中心圆环
    center_circle = Circle((0, 0), radius - num_layers * width, edgecolor='black', facecolor='white', linewidth=1)
    ax.add_artist(center_circle)

    # 设置标题
    ax.set_title(title, fontsize=16)

    # 设置图例
    # 创建自定义的图例句柄
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=colors[i], edgecolor='white', label=label) for i, label in enumerate(bin_labels)]
    ax.legend(handles=legend_patches, title="Bins", loc="upper right", bbox_to_anchor=(1.3, 1))

    ax.axis('equal')  # 保持圆形
    plt.tight_layout()
    plt.show()
    
    return fig