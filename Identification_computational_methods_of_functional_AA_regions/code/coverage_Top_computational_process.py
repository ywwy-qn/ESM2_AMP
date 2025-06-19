import re
import pandas as pd
from code_tools import get_segment_interval, process_dataframe
import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
kuuu = "uniprot" #uniprot  interpro
file_list = [
    "Identification_computational_methods_of_functional_AA_regions/data/0-0_Low3.xlsx",
    "Identification_computational_methods_of_functional_AA_regions/data/0-0_Top3.xlsx",
    "Identification_computational_methods_of_functional_AA_regions/data/0-1_Low3.xlsx",
    "Identification_computational_methods_of_functional_AA_regions/data/0-1_Top3.xlsx",
    "Identification_computational_methods_of_functional_AA_regions/data/1-0_Low3.xlsx",
    "Identification_computational_methods_of_functional_AA_regions/data/1-0_Top3.xlsx",
    "Identification_computational_methods_of_functional_AA_regions/data/1-1_Low3.xlsx",
    "Identification_computational_methods_of_functional_AA_regions/data/1-1_Top3.xlsx"
]

file_index = [1, 3, 5, 7]
output_paths = []
for idx in file_index:

    protein_file_path = f"Identification_computational_methods_of_functional_AA_regions/Total_information_protein/{kuuu}_processed_proteins.xlsx"
    excel_file_path = file_list[idx]
    original_filename = excel_file_path.split('/')[-1].replace('.xlsx', '')
    file_prefix = original_filename.split('_')[0]
    # Create output directory structure
    output_dir = os.path.join(
        "Identification_computational_methods_of_functional_AA_regions",
        f"database_{kuuu}_out",
        file_prefix
    )
    os.makedirs(output_dir, exist_ok=True)
    # Create output file path with the new structure
    output_file_path = os.path.join(output_dir, f"{original_filename}_out.xlsx")

    df_fugai = pd.read_excel(excel_file_path)
    df_fugai = df_fugai.iloc[:, :20]
    df_protein_information = pd.read_excel(protein_file_path)
    if kuuu == "uniprot":
        interval_types = ["Domain", "Region", "Compositional bias", "Repeat", "Motif"] # ["Homologous_superfamily", "Family", ]
    if kuuu == "interpro":
        interval_types = ["Domain", "Repeat", "Active_site", "Binding_site", "Binding_site", "Conserved_site", "Ptm"] # ["Homologous_superfamily", "Family", ]

    ###1_Map the sequence length value based on the protein name==================================================
    # Map the Entry and Length columns from df_protein_information to ProteinA and ProteinB, respectively.
    df_protein_mapping = df_protein_information.set_index('Entry')['Length']
    # To map ProteinA and ProteinB to their corresponding Length values from df_protein_information
    df_fugai['A_length'] = df_fugai['ProteinA'].map(df_protein_mapping)
    df_fugai['B_length'] = df_fugai['ProteinB'].map(df_protein_mapping)
    del df_protein_mapping



    ###2_Based on segment mapping intervals====================================================
    for num in range(int(3)):
        
        col1 = f'TOP{num+1}'
        col2 = f'Top{num+1}_D'
        
        df_fugai[col2] = df_fugai[col2].astype(object)
        
        for index, row in df_fugai.iterrows():
            top1_value = row[col1]
            if top1_value.startswith('A_segment'):

                segment = top1_value.replace('A_', '')
                total_value = row['A_length']
                interval = get_segment_interval(total_value, segment)
                df_fugai.at[index, col2] = interval
                
            elif top1_value.startswith('B_segment'):
                segment = top1_value.replace('B_', '')
                total_value = row['B_length']
                interval = get_segment_interval(total_value, segment)
                df_fugai.at[index, col2] = interval



    ###3_Based on segment and interval mapping of special sequence information===========================================

    for interval_type in interval_types:

        interval_pattern = re.compile(r'\((\d+), (\d+)\]')
        domain_pattern = re.compile(rf'{interval_type}\[(\d+),(\d+)\]: /note="([^"]+)"')
        
        for num in range(int(3)):
            col1 = f'TOP{num+1}'
            col2 = f'Top{num+1}_D'
            col3 = f'Top{num+1}_cover' 
            
            df_fugai[col3] = df_fugai[col3].astype(object)
            

            for index, row in df_fugai.iterrows():
                if row[col1].startswith('A_segment'):
 
                    protein = row['ProteinA']
                elif row[col1].startswith('B_segment'):

                    protein = row['ProteinB']
                else:
                    continue
                

                top1_d = row[col2]
                match = interval_pattern.match(top1_d)
                if not match:
                    continue
                start_top1, end_top1 = map(int, match.groups())
                
                # To find the corresponding Entry values in the df_protein_information
                protein_info = df_protein_information[df_protein_information['Entry'] == protein]
                if protein_info.empty:
                    continue
                
                # Extract Domain information
                domain_str = protein_info[interval_type].values[0]
                
                if pd.isna(domain_str):
                    continue
                
                matches = domain_pattern.findall(domain_str)
                
                cover_domains = []
                for match in matches:
                    start_domain, end_domain, description = match
                    start_domain, end_domain = int(start_domain), int(end_domain)
                    
                    
                    if start_top1 < end_domain and start_domain <= end_top1:
                        cover_domains.append(f"{interval_type}[{start_domain},{end_domain}]: /note=\"{description}\"")
                        
                
                # Concatenate the Domain strings that meet the conditions and store them in the Top1_cover column
                if cover_domains:
                    if pd.isna(df_fugai.at[index, col3]):
                        df_fugai.at[index, col3] = "; ".join(cover_domains)
                    else:
                        df_fugai.at[index, col3] += "; " + "; ".join(cover_domains)



    ###4_Automatically calculate the coverage interval ratio======================================================

    Tops = ["Top1", "Top2", "Top3"]
    #Tops = ["Low1", "Low2", "Low3"]
    processed_data = df_fugai.copy()
    for Top in Tops:
        processed_data = process_dataframe(df=processed_data, top_col=Top, interval_types=interval_types)

    # Save
    processed_data.to_excel(output_file_path, index=False)
    print(f"The data has been successfully saved to {output_file_path}")
    output_paths.append(output_file_path)
