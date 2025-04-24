import pandas as pd
import math
import re

def get_segment_interval(total_value, segment):
    """
    根据输入的 total_value 和 segment，直接返回对应分段的区间字符串。

    参数:
    total_value (int): 需要分段的数值。
    segment (str): 目标分段名称，例如 "segment0"。

    返回:
    str: 对应分段的区间字符串，例如 "(0, 55]"。
    """
    # 解析 segment 名称，获取分段索引
    segment_index = int(segment.replace("segment", ""))

    # 计算每段的长度
    segment_length = total_value / 10
    base_length = math.floor(segment_length)  # 基础长度
    remainder = total_value % 10  # 余数

    # 定义分段长度，前 remainder 段长度加1
    segment_lengths = [base_length + 1 if i < remainder else base_length for i in range(10)]

    # 计算目标分段的区间
    start = sum(segment_lengths[:segment_index])  # 分段起始点
    end = start + segment_lengths[segment_index]  # 分段结束点

    # 返回格式化后的区间字符串
    return f"({start}, {end}]"



def parse_special_intervals(input_str):
    pattern = r"(\w+\s*\w*)\[(\d+),(\d+)\]:"
    matches = re.findall(pattern, input_str)
    special_intervals = []
    for match in matches:
        interval_type = match[0]
        x1, y1 = int(match[1]), int(match[2])
        special_intervals.append((interval_type, x1, y1))
    return special_intervals


# 计算覆盖比率
def calculate_coverage_ratio(target, input_str, interval_types):
    
    special_intervals = parse_special_intervals(input_str)
    
    x, y = target
    fragment_length = y - x
    
    special_ratios = {}
    total_covered_length = 0
    special_intersections = {}
    total_intersection_points = set()

    for i, (interval_type, x1, y1) in enumerate(special_intervals):
        # 确保interval_type是合法的
        if interval_type not in interval_types:
            continue  # 跳过非预期的类型
        
        start = max(x + 1, x1)
        end = min(y, y1)

        if interval_type not in special_intersections:
            special_intersections[interval_type] = set()

        if start < end:
            intersection_points = set(range(start, end + 1))
            special_intersections[interval_type].update(intersection_points)
            total_intersection_points.update(intersection_points)
            special_covered_length = len(special_intersections[interval_type])
        else:
            special_covered_length = 0
            special_intersections[interval_type] = set()
        
        special_ratios[interval_type] = special_covered_length / fragment_length if fragment_length != 0 else 0
        
    total_covered_length = len(total_intersection_points)
    total_ratio = total_covered_length / fragment_length if fragment_length != 0 else 0

    return total_ratio, special_ratios, special_intersections, total_intersection_points


# 处理DataFrame
def process_dataframe(df, top_col, interval_types):
    
    for idx, row in df.iterrows():
        target_interval_str = row[f"{top_col}_D"]
        input_str = row[f"{top_col}_cover"]

        # 将目标区间字符串转换为目标区间元组
        target_interval = tuple(map(int, target_interval_str[1:-1].split(',')))
        
        # 确保input_str为字符串类型
        if not isinstance(input_str, str):
            input_str = str(input_str)

        total_ratio, special_ratios, special_intersections, total_intersection_points = calculate_coverage_ratio(target_interval, input_str, interval_types)
        
        # 填充总覆盖比率列
        df.at[idx, f"{top_col}_colate"] = round(total_ratio * 100, 1)
        
        # 填充每个区间类型的覆盖比率
        for interval_type in interval_types:
            col_name = f"{interval_type.capitalize()}_{top_col}_colate"
            df.at[idx, col_name] = round(special_ratios.get(interval_type, 0)*100, 1)
    
    return df






