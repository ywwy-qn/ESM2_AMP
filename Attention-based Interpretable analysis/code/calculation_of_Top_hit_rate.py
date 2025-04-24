import pandas as pd
import os

kuuu = "uniport" #uniport  interpro
Top_list = [
    fr'E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_{kuuu}_out\0-0\0-0_Top3_information.xlsx',
    fr'E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_{kuuu}_out\0-1\0-1_Top3_information.xlsx',
    fr'E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_{kuuu}_out\1-0\1-0_Top3_information.xlsx',
    fr'E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_{kuuu}_out\1-1\1-1_Top3_information.xlsx',
]

# 读取Excel文件
for path in Top_list:

    file_path = path
    output_path = os.path.dirname(file_path) 
    df = pd.read_excel(file_path)
    df_copy = df.copy()

    TT = ["Top1", "Top2", "Top3", "Top"]
    thresholds = [25, 50, 75]
    if kuuu == "uniport":
        structural_domains = ["Domain", "Region", "Compositional bias", "Repeat", "Motif"]
    if kuuu == "interpro":
        structural_domains = ["Domain", "Repeat", "Active_site", "Binding_site", "Binding_site", "Conserved_site", "Ptm"] # ["Homologous_superfamily", "Family", ] 删除分类层面区域


    # 定义一个函数来处理colate列
    def process_colate_column(col_name, threshold):
        df_copy[col_name] = df[col_name].apply(lambda x: 1 if x >= threshold else 0)
        return df_copy[col_name].mean() * 100

    # 处理"all"行对应的列
    def process_all_row(columns, threshold):
        df_copy['new_col'] = df[columns].apply(lambda row: 1 if any(row >= threshold) else 0, axis=1)
        return df_copy['new_col'].mean() * 100

    for threshold in thresholds:
        # 初始化结果字典
        result = {
            f"{TT[3]}_Thr{threshold}_%": [TT[0], TT[1], TT[2], "all"],
            **{domain: [0]*4 for domain in structural_domains},  # 动态生成结构域键
            "ALL": [0, 0, 0, 0]
        }

        # 处理特殊结构域
        for domain in structural_domains:
            # 生成当前结构域的列名列表
            domain_columns = [f"{domain}_{TT[i]}_colate" for i in range(3)]
            
            # 处理各阈值列
            for i, col in enumerate(domain_columns):
                result[domain][i] = process_colate_column(col, threshold)
            
            # 处理all行
            result[domain][3] = process_all_row(domain_columns, threshold)

        # 处理ALL列（特殊处理）
        all_columns = [f"{TT[i]}_colate" for i in range(3)]
        for i, col in enumerate(all_columns):
            result["ALL"][i] = process_colate_column(col, threshold)
        result["ALL"][3] = process_all_row(all_columns, threshold)

        # 创建DataFrame并保存
        result_df = pd.DataFrame(result)
        
        # 定义输出路径和文件名
        file_name = f'{TT[3]}_Thr{threshold}_hit_result_output.xlsx'
        file_path = os.path.join(output_path, file_name)
        
        # 保存结果
        result_df.to_excel(file_path, index=False)
        print(f"文件已保存到: {file_path}")
        print(result_df)
