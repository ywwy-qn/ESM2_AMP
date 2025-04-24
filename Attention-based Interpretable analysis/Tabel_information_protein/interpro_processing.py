'''
修改自InterPro官网上的代码
用于读取InterPro上的查找结果 export.tsv 并根据结果下载所有蛋白质的结构域信息
'''
# standard library modules
import sys, errno, re, json, ssl, os
import glob
import pandas as pd
from tqdm import tqdm
from urllib import request
from urllib.error import HTTPError
from collections import defaultdict
from time import sleep

# BASE_URL = "https://www.ebi.ac.uk:443/interpro/api/entry/InterPro/protein/reviewed/A0A024R1R8/?page_size=200"

def parse_items(items):
    if type(items)==list:
        return ",".join(items)
    return ""
def parse_member_databases(dbs):
    if type(dbs)==dict:
        return ";".join([f"{db}:{','.join(dbs[db])}" for db in dbs.keys()])
    return ""
def parse_go_terms(gos):
    if type(gos)==list:
        return ",".join([go["identifier"] for go in gos])
    return ""
def parse_locations(locations):
    if type(locations)==list:
        return ",".join(
        [",".join([f"{fragment['start']}..{fragment['end']}" 
                    for fragment in location["fragments"]
                    ])
        for location in locations
        ])
    return ""
def parse_group_column(values, selector):
    return ",".join([parse_column(value, selector) for value in values])

def parse_column(value, selector):
    if value is None:
        return ""
    elif "member_databases" in selector:
        return parse_member_databases(value)
    elif "go_terms" in selector: 
        return parse_go_terms(value)
    elif "children" in selector: 
        return parse_items(value)
    elif "locations" in selector:
        return parse_locations(value)
    return str(value)

def download_to_file(url, file_path):
    #disable SSL verification to avoid config issues
    context = ssl._create_unverified_context()

    next = url
    last_page = False

    
    attempts = 0
    while next:
        try:
            req = request.Request(next, headers={"Accept": "application/json"})
            res = request.urlopen(req, context=context)
            # If the API times out due a long running query
            if res.status == 408:
                # wait just over a minute
                sleep(61)
                # then continue this loop with the same URL
                continue
            elif res.status == 204:
                #no data so leave loop
                break
            payload = json.loads(res.read().decode())
            next = payload["next"]
            attempts = 0
            if not next:
                last_page = True
        except HTTPError as e:
            if e.code == 408:
                sleep(61)
                continue
            else:
                # If there is a different HTTP error, it wil re-try 3 times before failing
                if attempts < 3:
                    attempts += 1
                    sleep(61)
                    continue
                else:
                    sys.stderr.write("LAST URL: " + next)
                    raise e
        with open(file_path,"w+") as f:
            for i, item in enumerate(payload["results"]):
                f.write(parse_column(item["metadata"]["accession"], 'metadata.accession') + "\t")
                f.write(parse_column(item["metadata"]["name"], 'metadata.name') + "\t")
                f.write(parse_column(item["metadata"]["source_database"], 'metadata.source_database') + "\t")
                f.write(parse_column(item["metadata"]["type"], 'metadata.type') + "\t")
                f.write(parse_column(item["metadata"]["integrated"], 'metadata.integrated') + "\t")
                f.write(parse_column(item["metadata"]["member_databases"], 'metadata.member_databases') + "\t")
                f.write(parse_column(item["metadata"]["go_terms"], 'metadata.go_terms') + "\t")
                f.write(parse_column(item["proteins"][0]["accession"], 'proteins[0].accession') + "\t")
                f.write(parse_column(item["proteins"][0]["protein_length"], 'proteins[0].protein_length') + "\t")
                f.write(parse_column(item["proteins"][0]["entry_protein_locations"], 'proteins[0].entry_protein_locations') + "\t")
                f.write("\n")
        
    # Don't overload the server, give it time before asking for more
    sleep(1)



def process_dataframe(df, protein_id):
    
    if df.empty:
        # 处理空DataFrame的情况，如返回空DataFrame或记录错误
        print(f"警告: 蛋白质ID {protein_id} 无数据")
        return pd.DataFrame({"Entry": [protein_id]})  # 返回空的DataFrame或适当处理
    
    else:
        # 删除指定位置的列
        df.drop(df.columns[[0, 2, 4, 5, 6, 10]], axis=1, inplace=True)
        
        # 重命名列
        df.columns = ['function', 'type', 'Entry', 'Length', 'Interval']
        
        # 处理Interval列
        df["Interval"] = df["Interval"].str.split(",")
        df = df.explode("Interval")
        df["Interval"] = df["Interval"].str.replace(" ", "")
        df["Interval"] = df["Interval"].apply(lambda s: f"[{s.replace('..', ',')}]")  # 直接将 ".." 替换为 ", "，并包裹方括号

        # 生成type_information并删除冗余列
        df["type_information"] = df.apply(
            lambda row: f"{row['type']}{row['Interval']}: /note=\"{row['function']}\";", 
            axis=1
        )
        df.drop(['function', 'type', 'Interval'], axis=1, inplace=True)
        
        # 收集和处理类型信息
        entry_value = df['Entry'].iloc[0]
        length_value = df['Length'].iloc[0]
        type_collections = defaultdict(list)
        
        for type_info in df['type_information']:
            if match := re.match(r'^([^[]+)\[', str(type_info)):
                type_name = match.group(1).strip()
                column = type_name.capitalize()
                type_collections[column].append(str(type_info).capitalize())
        
        # 创建结果DataFrame
        merged_data = {k: ' '.join(v) for k, v in type_collections.items()}
        result_df = pd.DataFrame([{
            'Entry': protein_id,
            'Length': length_value,
            **merged_data
        }])
        
        return result_df



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

domain_path = r"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\Tabel_information_protein\domain"
output_file = os.path.join(r"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\Tabel_information_protein", "interpro_processed_proteins.xlsx")
os.makedirs(domain_path, exist_ok=True)

# 读取并合并所有Excel文件
all_dfs = []
for file in file_list:
    df = pd.read_excel(file, engine='openpyxl')
    all_dfs.append(df)
merged_df = pd.concat(all_dfs, axis=0, ignore_index=True)
# 合并ProteinA和ProteinB列并去重
protein_ids = pd.concat([merged_df['ProteinA'], merged_df['ProteinB']]).dropna().unique().tolist()
del merged_df, df, all_dfs

step = 0
for protein_id in tqdm(protein_ids, desc="Downloading proteins"):
    
    save_file = os.path.join(domain_path, f"{protein_id}.tsv")
    if os.path.exists(save_file):
        continue  # 跳过当前循环
    url = f"https://www.ebi.ac.uk:443/interpro/api/entry/InterPro/protein/reviewed/{protein_id}/?page_size=200"
    download_to_file(url,os.path.join('domain', protein_id+'.tsv'))


# 获取所有TSV文件列表
file_list = glob.glob(os.path.join(domain_path, "*.tsv"))
if not file_list:
    raise FileNotFoundError(f"No TSV files found in: {domain_path}")

# 读取并合并所有DataFrame
dfs = []
for file in tqdm(file_list, desc="processing"):

    protein_id = file.split("\\")[-1].split(".tsv")[0]  # 去除后缀
    df = pd.read_csv(file, sep='\t', header=None).rename(columns=lambda x: f'unnamed:{x}')
    df = process_dataframe(df, protein_id)
    dfs.append(df)

combined_df = pd.concat(dfs, axis=0, ignore_index=True)
combined_df.to_excel(output_file, index=False)
print(f"Successfully merged {len(file_list)} files. Output saved to: {output_file}")