import pandas as pd
import numpy as np
from code_tools.visual import read_excel_file, bin_data, plot_donut_charts
import os

kuuu = "uniprot"  # uniprot or interpro
pre_list = ["0-0", "0-1", "1-0", "1-1"]

for pre in pre_list:
    types = ['Top', 'Low']
    file_dict = {
        'Top': {
            'file_path': f"Identification_computational_methods_of_functional_AA_regions/database_{kuuu}_out/{pre}/{pre}_Top3_out.xlsx",
            'required_columns': [
                ['Top1_colate', 'Top2_colate', 'Top3_colate'],
                ['Domain_Top1_colate', 'Domain_Top2_colate', 'Domain_Top3_colate'],
                ['Region_Top1_colate', 'Region_Top2_colate', 'Region_Top3_colate'],
                ['Repeat_Top1_colate', 'Repeat_Top2_colate', 'Repeat_Top3_colate'],
            ]
        },
        'Low': {
            'file_path': f"Identification_computational_methods_of_functional_AA_regions/database_{kuuu}_out/{pre}/{pre}_Low3_out.xlsx",
            'required_columns': [
                ['Low1_colate', 'Low2_colate', 'Low3_colate'],
                ['Domain_Low1_colate', 'Domain_Low2_colate', 'Domain_Low3_colate'],
                ['Region_Low1_colate', 'Region_Low2_colate', 'Region_Low3_colate'],
                ['Repeat_Low1_colate', 'Repeat_Low2_colate', 'Repeat_Low3_colate'],
            ]
        }
    }
    
    out_dir = f"Identification_computational_methods_of_functional_AA_regions/database_{kuuu}_out/{pre}"
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    for i in types:
        file_path = file_dict[i]['file_path']
        
        # Check if file exists before processing
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        for ii in range(3):
            required_columns = file_dict[i]['required_columns'][ii]
            
            try:
                df = read_excel_file(file_path, required_columns)
                
                bins = np.arange(0, 101, 10)
                bin_labels = [f'{i}-{i+10}' for i in bins[:-1]]
                
                counts = bin_data(df, required_columns, bins, bin_labels)
                
                total_counts = {col: counts[col].sum() for col in required_columns}
                print(f"Prefix: {pre}, Type: {i}, Columns: {required_columns[0].replace('1_colate', '')}")
                print("The total number of each column:", total_counts)
                
                plot_title = f"Overall Special Segmentation Coverage of the {required_columns[0].replace('1_colate', '')} Three Attention Segments"
                fig = plot_donut_charts(counts, bin_labels, plot_title)
                
                output_pdf_path = os.path.join(out_dir, f"{required_columns[0].replace('1_colate', '')} Nested Donut Chart.pdf")
                fig.savefig(output_pdf_path, format='pdf', bbox_inches='tight')
                
                print(f"The chart has been saved as {output_pdf_path}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Error processing {file_path} with columns {required_columns}: {str(e)}")
                continue