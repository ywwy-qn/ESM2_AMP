import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle



def read_excel_file(file_path, required_columns):
    """
    Read the Excel file and verify whether the required columns exist and are of numeric type.
    """
    if not os.path.exists(file_path):
        print(f"The file path does not exist: {file_path}")
        sys.exit(1)
    
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"An error occurred when reading an Excel file: {e}")
        sys.exit(1)
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"The following specified columns are missing: {missing_columns}")
        sys.exit(1)
    

    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def bin_data(df, columns, bins, bin_labels):
    """
    Perform binning on the data of specified columns and calculate the count for each bin.
    """
    counts = {}
    for col in columns:
        series = df[col].dropna()
        binned = pd.cut(series, bins=bins, labels=bin_labels, right=True, include_lowest=True)
        counts[col] = binned.value_counts().sort_index()
    return counts

def plot_donut_charts(counts, bin_labels, title):
    """
    Draw a circular pie chart to show the distribution of data in each column.
    """
    required_columns = list(counts.keys())
    num_layers = len(required_columns)
    fig, ax = plt.subplots(figsize=(10, 8))
    radius = 0.8
    width = 0.2

    colors = plt.cm.tab20(np.linspace(0, 1, len(bin_labels)))

    for i, col in enumerate(required_columns):
        data_to_plot = counts[col]

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
    

        ax.text(0, current_radius-width/2, col.replace('_colate', ''), ha='center', va='center', fontsize=13, fontweight='bold')
    

    center_circle = Circle((0, 0), radius - num_layers * width, edgecolor='black', facecolor='white', linewidth=1)
    ax.add_artist(center_circle)


    ax.set_title(title, fontsize=16)


    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=colors[i], edgecolor='white', label=label) for i, label in enumerate(bin_labels)]
    ax.legend(handles=legend_patches, title="Bins", loc="upper right", bbox_to_anchor=(1.3, 1))

    ax.axis('equal')
    plt.tight_layout()
    plt.show()
    
    return fig