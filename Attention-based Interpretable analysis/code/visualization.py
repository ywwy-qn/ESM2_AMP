import pandas as pd
import numpy as np
from code_tools.visual import read_excel_file, bin_data, plot_donut_charts



kuuu = "uniport" #uniport  interpro
pre_list = ["0-0","0-1","1-0","1-1"]
for pre in pre_list:
    # 文件路径 
    types = ['Top', 'Low']
    file_dict = {
        'Top':{'file_path':fr"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_{kuuu}_out\{pre}\{pre}_Top3_information.xlsx",
        'required_columns':[['Top1_colate', 'Top2_colate', 'Top3_colate'],
                            ['Domain_Top1_colate', 'Domain_Top2_colate', 'Domain_Top3_colate'],
                            ['Region_Top1_colate', 'Region_Top2_colate', 'Region_Top3_colate'], #This line applies only to UniProt database; delete if for InterPro.
                            ['Repeat_Top1_colate', 'Repeat_Top2_colate', 'Repeat_Top3_colate'],
                            ]
        },
        
        'Low':{'file_path':fr"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_{kuuu}_out\{pre}\{pre}_Low3_information.xlsx",
        'required_columns':[['Low1_colate', 'Low2_colate', 'Low3_colate'],
                            ['Domain_Low1_colate', 'Domain_Low2_colate', 'Domain_Low3_colate'],
                            ['Region_Low1_colate', 'Region_Low2_colate', 'Region_Low3_colate'], #This line applies only to UniProt database; delete if for InterPro.
                            ['Repeat_Low1_colate', 'Repeat_Low2_colate', 'Repeat_Low3_colate'],
                            ]
        }
            }
    out_dir = fr"E:\Working\Team\Sun_Yaven\Artcle_topic\文章修改\高注意力片段与特殊片段覆盖信息\database_{kuuu}_out\{pre}"


    for i in types:
        file_path = file_dict[i]['file_path']
        for ii in range(3):
            required_columns = file_dict[i]['required_columns'][ii]

            # 读取并验证数据
            df = read_excel_file(file_path, required_columns)
            
            # 分段区间（倒序）
            bins = np.arange(0, 101, 10)
            bin_labels = [f'{i}-{i+10}' for i in bins[:-1]]  # 正序标签
            
            # 分箱计数
            counts = bin_data(df, required_columns, bins, bin_labels)
            
            # 检查是否所有计数的总和一致，预防错误计数
            total_counts = {col: counts[col].sum() for col in required_columns}
            print("各列的总计数:", total_counts)
            
            # 绘制图表
            plot_title = f"Overall Special Segmentation Coverage of the {required_columns[0].replace('1_colate', '')} Three Attention Segments"
            fig = plot_donut_charts(counts, bin_labels, plot_title)
            
            
            # 保存图表为PDF
            output_pdf_path = out_dir + f"/{required_columns[0].replace('1_colate', '')} Nested Donut Chart.pdf"
            fig.savefig(output_pdf_path, format='pdf', bbox_inches='tight')
            
            print(f"图表已保存为 {output_pdf_path}")