import numpy as np
import pandas as pd
from code_tools.visual import read_excel_file, bin_data, plot_donut_charts



kuuu = "uniprot" #uniprot  interpro
pre_list = ["0-0","0-1","1-0","1-1"]
for pre in pre_list:

    file_dict={
        "Top":{'file_path':f"Identification_computational_methods_of_functional_AA_regions/database_{kuuu}_out/{pre}/{pre}_Top3_out.xlsx",
        'required_columns':['Top1_colate', 'Top2_colate', 'Top3_colate']},
        "Low":{'file_path':f"Identification_computational_methods_of_functional_AA_regions/database_{kuuu}_out/{pre}/{pre}_Low3_out.xlsx",
        'required_columns':['Low1_colate', 'Low2_colate', 'Low3_colate']}
        }
    out_dir = f"Identification_computational_methods_of_functional_AA_regions/database_{kuuu}_out/{pre}"


    types = ["Top", "Low"]
    df = pd.DataFrame([], columns=types)

    for typee in types:

        file_path = file_dict[typee]['file_path']
        required_columns = file_dict[typee]['required_columns']
        
        data = read_excel_file(file_path, required_columns)
        

        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"column '{col}' does not exist in the file '{file_path}' ")
        
        data_subset = pd.concat([data[required_columns[0]], data[required_columns[1]], data[required_columns[1]]],
                                axis=0)
        
        df[typee] = data_subset.values.tolist()
        

    bins = np.arange(0, 101, 10)
    bin_labels = [f'{i}-{i+10}' for i in bins[:-1]]

    counts = bin_data(df, types, bins, bin_labels)

    total_counts = {col: counts[col].sum() for col in types}
    print("The total number of each column:", total_counts)

    plot_title = "Overall Special Segmentation Coverage of the TopLow Three Attention Segments"
    fig = plot_donut_charts(counts, bin_labels, plot_title)


    output_pdf_path = out_dir + "/All Top-Low Nested Donut Chart.pdf"
    fig.savefig(output_pdf_path, format='pdf', bbox_inches='tight')

    print(f"The chart has been saved as {output_pdf_path}")