import re
import pandas as pd
from code_tools import get_segment_interval, process_dataframe

kuuu = "uniport" #uniport  interpro
file_list = [
    fr"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_{kuuu}_out\0-0\0-0_Low3.xlsx",
    fr"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_{kuuu}_out\0-0\0-0_Top3.xlsx",
    fr"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_{kuuu}_out\0-1\0-1_Low3.xlsx",
    fr"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_{kuuu}_out\0-1\0-1_Top3.xlsx",
    fr"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_{kuuu}_out\1-0\1-0_Low3.xlsx",
    fr"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_{kuuu}_out\1-0\1-0_Top3.xlsx",
    fr"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_{kuuu}_out\1-1\1-1_Low3.xlsx",
    fr"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_{kuuu}_out\1-1\1-1_Top3.xlsx"
]

file_index = [1, 3, 5, 7]
for idx in file_index:

    protein_file_path = fr"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\Tabel_information_protein\{kuuu}_processed_proteins.xlsx"
    excel_file_path = file_list[idx]
    output_file_path = excel_file_path.replace('.xlsx', '_information.xlsx')

    df_fugai = pd.read_excel(excel_file_path)
    df_fugai = df_fugai.iloc[:, :20]  # 保留前20列
    df_protein_information = pd.read_excel(protein_file_path)
    if kuuu == "uniport":
        interval_types = ["Domain", "Region", "Compositional bias", "Repeat", "Motif"] # ["Homologous_superfamily", "Family", ] 删除分类层面区域
    if kuuu == "interpro":
        interval_types = ["Domain", "Repeat", "Active_site", "Binding_site", "Binding_site", "Conserved_site", "Ptm"] # ["Homologous_superfamily", "Family", ] 删除分类层面区域

    ###1_基于蛋白名称映射序列长度值==================================================
    # 首先，将df_protein_information中的Entry和Length列分别映射到ProteinA和ProteinB
    df_protein_mapping = df_protein_information.set_index('Entry')['Length']
    # 使用map函数将ProteinA和ProteinB映射到对应的Length
    df_fugai['A_length'] = df_fugai['ProteinA'].map(df_protein_mapping)
    df_fugai['B_length'] = df_fugai['ProteinB'].map(df_protein_mapping)
    del df_protein_mapping



    ###2_基于segment分段映射区间====================================================
    for num in range(int(3)):
        
        col1 = f'TOP{num+1}'
        col2 = f'Top{num+1}_D'
        
        df_fugai[col2] = df_fugai[col2].astype(object)
        
        for index, row in df_fugai.iterrows():
            top1_value = row[col1]
            if top1_value.startswith('A_segment'):
                # 如果是 A_segment{0-9}，使用 A_length 列的值
                segment = top1_value.replace('A_', '')  # 去掉前缀 'A_'
                total_value = row['A_length']
                interval = get_segment_interval(total_value, segment)
                df_fugai.at[index, col2] = interval
            elif top1_value.startswith('B_segment'):
                # 如果是 B_segment{0-9}，使用 B_length 列的值
                segment = top1_value.replace('B_', '')  # 去掉前缀 'B_'
                total_value = row['B_length']
                interval = get_segment_interval(total_value, segment)
                df_fugai.at[index, col2] = interval



    ###3_基于segment和区间映射特殊片段信息===========================================
    # 遍历 interval_type
    for interval_type in interval_types:
        # 正则表达式提取区间信息
        interval_pattern = re.compile(r'\((\d+), (\d+)\]')  # 匹配 Top1_D 的区间
        domain_pattern = re.compile(rf'{interval_type}\[(\d+),(\d+)\]: /note="([^"]+)"')  # 匹配 Domain 的区间和描述
        
        for num in range(int(3)):
            col1 = f'TOP{num+1}'
            col2 = f'Top{num+1}_D'
            col3 = f'Top{num+1}_cover' 
            
            df_fugai[col3] = df_fugai[col3].astype(object)
            
            # 遍历 df_fugai 的每一行
            for index, row in df_fugai.iterrows():
                if row[col1].startswith('A_segment'):
                    # 处理 A_segment{0-9}
                    protein = row['ProteinA']
                elif row[col1].startswith('B_segment'):
                    # 处理 B_segment{0-9}
                    protein = row['ProteinB']
                else:
                    continue  # 如果不是 A_segment 或 B_segment，跳过
                
                # 提取 Top1_D 的区间
                top1_d = row[col2]
                match = interval_pattern.match(top1_d)
                if not match:
                    continue
                start_top1, end_top1 = map(int, match.groups())  # 获取区间起始和结束值
                
                # 查找 df_protein_information 中对应的 Entry
                protein_info = df_protein_information[df_protein_information['Entry'] == protein]
                if protein_info.empty:
                    continue
                
                # 提取 Domain 信息
                domain_str = protein_info[interval_type].values[0]
                
                # 检查 domain_str 是否为 NaN
                if pd.isna(domain_str):
                    continue  # 如果是 NaN，跳过
                
                matches = domain_pattern.findall(domain_str)  # 提取所有 Domain 区间和描述
                # print(f"protein:{protein} -> domain_str:{domain_str} -> matches:{matches}")
                
                # 判断区间是否重合（严格数学逻辑）
                cover_domains = []
                for match in matches:
                    start_domain, end_domain, description = match
                    start_domain, end_domain = int(start_domain), int(end_domain)
                    
                    # 判断 (start_top1, end_top1] 与 [start_domain, end_domain] 是否重合
                    if start_top1 < end_domain and start_domain <= end_top1:
                        cover_domains.append(f"{interval_type}[{start_domain},{end_domain}]: /note=\"{description}\"")
                        # cover_domains.append(f"{interval_type}[{start_domain},{end_domain}]:{description}")
                
                # 将符合条件的 Domain 字符串拼接并存储到 Top1_cover 列
                if cover_domains:
                    if pd.isna(df_fugai.at[index, col3]):
                        df_fugai.at[index, col3] = "; ".join(cover_domains)
                    else:
                        df_fugai.at[index, col3] += "; " + "; ".join(cover_domains)



    ###4_全自动计算覆盖区间比率======================================================
    # 处理DataFrame
    Tops = ["Top1", "Top2", "Top3"]
    #Tops = ["Low1", "Low2", "Low3"]
    processed_data = df_fugai.copy()
    for Top in Tops:
        processed_data = process_dataframe(df=processed_data, top_col=Top, interval_types=interval_types)

    # 将 DataFrame 保存为 Excel 文件
    processed_data.to_excel(output_file_path, index=False)
    print(f"数据已成功保存到 {output_file_path}")
