import pandas as pd
import os

kuuu = "uniprot" #uniprot  interpro
Top_list = [
    f'Identification_computational_methods_of_functional_AA_regions/database_{kuuu}_out/0-0/0-0_Top3_out.xlsx',
    f'Identification_computational_methods_of_functional_AA_regions/database_{kuuu}_out/0-1/0-1_Top3_out.xlsx',
    f'Identification_computational_methods_of_functional_AA_regions/database_{kuuu}_out/1-0/1-0_Top3_out.xlsx',
    f'Identification_computational_methods_of_functional_AA_regions/database_{kuuu}_out/1-1/1-1_Top3_out.xlsx',
]


for path in Top_list:

    file_path = path
    output_path = os.path.dirname(file_path) 
    df = pd.read_excel(file_path)
    df_copy = df.copy()

    TT = ["Top1", "Top2", "Top3", "Top"]
    thresholds = [25, 50, 75]
    if kuuu == "uniprot":
        structural_domains = ["Domain", "Region", "Compositional bias", "Repeat", "Motif"]
    if kuuu == "interpro":
        structural_domains = ["Domain", "Repeat", "Active_site", "Binding_site", "Binding_site", "Conserved_site", "Ptm"] # ["Homologous_superfamily", "Family", ]


    # Define a function to handle the colate column
    def process_colate_column(col_name, threshold):
        df_copy[col_name] = df[col_name].apply(lambda x: 1 if x >= threshold else 0)
        return df_copy[col_name].mean() * 100

    # Handle the columns corresponding to the "all" row
    def process_all_row(columns, threshold):
        df_copy['new_col'] = df[columns].apply(lambda row: 1 if any(row >= threshold) else 0, axis=1)
        return df_copy['new_col'].mean() * 100

    for threshold in thresholds:
        # Initialize the result dictionary
        result = {
            f"{TT[3]}_Thr{threshold}_%": [TT[0], TT[1], TT[2], "all"],
            **{domain: [0]*4 for domain in structural_domains},
            "ALL": [0, 0, 0, 0]
        }

        for domain in structural_domains:
            domain_columns = [f"{domain}_{TT[i]}_colate" for i in range(3)]
            
            for i, col in enumerate(domain_columns):
                result[domain][i] = process_colate_column(col, threshold)
            
            result[domain][3] = process_all_row(domain_columns, threshold)

        all_columns = [f"{TT[i]}_colate" for i in range(3)]
        for i, col in enumerate(all_columns):
            result["ALL"][i] = process_colate_column(col, threshold)
        result["ALL"][3] = process_all_row(all_columns, threshold)

        result_df = pd.DataFrame(result)
        
        file_name = f'{TT[3]}_Thr{threshold}_hit_result_output.xlsx'
        file_path = os.path.join(output_path, file_name)
        
        result_df.to_excel(file_path, index=False)
        print(f"The file has been saved to: {file_path}")
        print(result_df)
