import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap

class Attention_visualization_diagram:
    def __init__(self, merged_test, i, label_name, model_name):
        """
        Initialize the attention visualization diagram class.
        
        Parameters:
        - merged_test: The attention weights data
        - i: Layer index
        - label_name: Labels for the axes
        - model_name: Name of the model for title and saving
        """
        self.merged_test = merged_test
        self.i = i
        self.label_name = label_name
        self.model_name = model_name
        self.custom_blue = (54/255., 125/255., 176/255.)
        self.cmap = LinearSegmentedColormap.from_list(
            'custom_blue', 
            [(1, 1, 1), self.custom_blue], 
            N=256
        )
    
    def _configure_plot(self, ax):
        """Common configuration for both types of plots."""
        ax.xaxis.tick_top()  # Move x-axis ticks and labels to top
        ax.xaxis.set_label_position('top')  # Set x-axis label position to top
        plt.xticks(rotation=45, ha='left')
        plt.tight_layout()
    
    def plot_total(self):
        """Plot the average attention weights across all samples."""
        # Calculate average
        average = np.mean(self.merged_test, axis=0)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            average, 
            cmap=self.cmap, 
            annot=False, 
            fmt=".1f", 
            linewidths=.5, 
            square=True, 
            cbar=True,
            xticklabels=self.label_name, 
            yticklabels=self.label_name
        )
        
        ax = plt.gca()
        self._configure_plot(ax)
        plt.title(f'{self.model_name} attention significant map in Layer{self.i+1}')
        
        # Save plot
        total_plot_path = os.path.join(f'AMPmodel_explainable/Attention_explainable/attention_weights/{self.model_name}/total_samples_map')
        os.makedirs(total_plot_path, exist_ok=True)
        plt.savefig(os.path.join(total_plot_path, f'{self.model_name} attention significant map in Layer{self.i+1}.pdf'))
        
        # plt.show()
    
    def plot_single(self, m):
        """Plot attention weights for a single sample."""
        # Get data for single sample
        attention_map_data = np.array(self.merged_test)[m]
        
        # Create plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            attention_map_data, 
            cmap=self.cmap, 
            annot=False, 
            fmt=".1f", 
            linewidths=.5, 
            square=True, 
            cbar=True,
            xticklabels=self.label_name, 
            yticklabels=self.label_name
        )
        
        ax = plt.gca()
        self._configure_plot(ax)
        plt.title(f'Pair{m+1} {self.model_name} attention significant map in Layer{self.i+1}')
        
        # Save plot
        single_plot_path = os.path.join(f'AMPmodel_explainable/Attention_explainable/attention_weights/{self.model_name}/single_samples_map')
        os.makedirs(single_plot_path, exist_ok=True)
        plt.savefig(os.path.join(single_plot_path, f'Pair{m+1} {self.model_name} attention significant map in Layer{self.i+1}.pdf'))
        
        # plt.show()