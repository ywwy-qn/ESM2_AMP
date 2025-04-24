import numpy as np
import pandas as pd
from code_tools.visual import read_excel_file, bin_data, plot_donut_charts



kuuu = "uniport" #uniport  interpro
pre_list = ["0-0","0-1","1-0","1-1"]
for pre in pre_list:

    file_dict={
        "Top":{'file_path':fr"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_{kuuu}_out\{pre}\{pre}_Top3_information.xlsx",
        'required_columns':['Top1_colate', 'Top2_colate', 'Top3_colate']},
        "Low":{'file_path':fr"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_{kuuu}_out\{pre}\{pre}_Low3_information.xlsx",
        'required_columns':['Low1_colate', 'Low2_colate', 'Low3_colate']}
        }
    out_dir = fr"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_{kuuu}_out\{pre}"


    # 读取并验证数据
    types = ["Top", "Low"]
    df = pd.DataFrame([], columns=types)

    # 循环读取数据
    for typee in types:
        # 读取文件
        file_path = file_dict[typee]['file_path']
        required_columns = file_dict[typee]['required_columns']
        
        # 读取数据
        data = read_excel_file(file_path, required_columns)
        
        # 验证所需列是否存在
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"列 '{col}' 不存在于文件 '{file_path}' 中")
        
        # 提取所需列的数据
        data_subset = pd.concat([data[required_columns[0]], data[required_columns[1]], data[required_columns[1]]],
                                axis=0)
        
        # 将数据添加到 DataFrame 中
        df[typee] = data_subset.values.tolist()
        


    # 分段区间（倒序）
    bins = np.arange(0, 101, 10)
    bin_labels = [f'{i}-{i+10}' for i in bins[:-1]]  # 正序标签

    # 分箱计数
    counts = bin_data(df, types, bins, bin_labels)

    # 检查是否所有计数的总和一致，预防错误计数
    total_counts = {col: counts[col].sum() for col in types}
    print("各列的总计数:", total_counts)

    # 绘制图表
    plot_title = "Overall Special Segmentation Coverage of the TopLow Three Attention Segments"
    fig = plot_donut_charts(counts, bin_labels, plot_title)


    # 保存图表为PDF
    output_pdf_path = out_dir + "/All Top-Low Nested Donut Chart.pdf"
    fig.savefig(output_pdf_path, format='pdf', bbox_inches='tight')

    print(f"图表已保存为 {output_pdf_path}")