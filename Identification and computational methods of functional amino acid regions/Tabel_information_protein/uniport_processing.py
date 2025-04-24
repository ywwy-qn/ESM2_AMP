import pandas as pd
import re
import os

# 文件路径
file_path = r"E:\Working\Team\Sun_Yaven\Artcle_topic\高注意力片段与特殊片段覆盖信息\蛋白质信息表\uniprotkb_taxonomy_id_9606_2025_01_03.tsv"  # download from uniport

# 读取TSV文件为DataFrame
df = pd.read_csv(file_path, sep='\t')

# 删除列 "Domain [CC]"
df = df.drop(columns=["Domain [CC]"])

# 修改列名 "Domain [FT]" 为 "Domain"
df = df.rename(columns={"Domain [FT]": "Domain"})


# 定义替换函数
def format_column(text, pattern, replacement):
    # 如果值为 NaN 或 None，返回空字符串
    if pd.isna(text):
        return ""
    # 使用正则表达式匹配并替换
    formatted_text = re.sub(pattern, replacement, text)
    return formatted_text

# 定义每一列的正则表达式和替换格式
column_patterns = {
    "Domain": (r"DOMAIN (\d+)\.\.(\d+);", r"Domain[\1,\2]:"),
    "Compositional bias": (r"COMPBIAS (\d+)\.\.(\d+);", r"Compositional bias[\1,\2]:"),
    "Motif": (r"MOTIF (\d+)\.\.(\d+);", r"Motif[\1,\2]:"),
    "Region": (r"REGION (\d+)\.\.(\d+);", r"Region[\1,\2]:"),
    "Repeat": (r"REPEAT (\d+)\.\.(\d+);", r"Repeat[\1,\2]:")
}

# 对每一列进行处理
for col, (pattern, replacement) in column_patterns.items():
    # 将 NaN 值替换为空字符串
    df[col] = df[col].fillna("")
    # 应用替换函数到当前列
    df[col] = df[col].apply(format_column, args=(pattern, replacement))



# 修复路径转义问题（建议使用原始字符串）
file_list = [
    r"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_interpro_out\0-0\0-0_Low3.xlsx",
    r"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_interpro_out\0-0\0-0_Top3.xlsx",
    r"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_interpro_out\0-1\0-1_Low3.xlsx",
    r"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_interpro_out\0-1\0-1_Top3.xlsx",
    r"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_interpro_out\1-0\1-0_Low3.xlsx",
    r"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_interpro_out\1-0\1-0_Top3.xlsx",
    r"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_interpro_out\1-1\1-1_Low3.xlsx",
    r"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_interpro_out\1-1\1-1_Top3.xlsx"
]
all_dfs = []
for file in file_list:
    dfs = pd.read_excel(file, engine='openpyxl')
    all_dfs.append(dfs)
merged_df = pd.concat(all_dfs, axis=0, ignore_index=True)
# 合并ProteinA和ProteinB列并去重
protein_ids = pd.concat([merged_df['ProteinA'], merged_df['ProteinB']]).dropna().unique().tolist()
del merged_df, dfs, all_dfs


new_df = df[df['Entry'].isin(protein_ids)]
# 定义输出文件路径
output_file_path = r"E:\Working\Team\Sun_Yaven\Artcle_topic\高注意力片段与特殊片段覆盖信息\蛋白质信息表\uniport_processed_preteins.xlsx"
# 保存 DataFrame 为 Excel 文件
new_df.to_excel(output_file_path, index=False)
print(f"文件已保存到: {output_file_path}")
