import pandas as pd
import math
import re

def get_segment_interval(total_value, segment):
    """
    Based on the input total_value and segment, directly return the interval string corresponding to the segment.

    parameters:
    total_value (int): The values that need to be segmented.
    segment (str): Target segment name, such as "segment0".

    return:
    str: The interval string corresponding to the segment, such as "(0, 55]".
    """

    segment_index = int(segment.replace("segment", ""))

    # Calculate the length of each section
    segment_length = total_value / 10
    base_length = math.floor(segment_length)
    remainder = total_value % 10


    segment_lengths = [base_length + 1 if i < remainder else base_length for i in range(10)]

    # Calculate the interval of the target segment
    start = sum(segment_lengths[:segment_index])
    end = start + segment_lengths[segment_index]


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


# Calculate the coverage rate
def calculate_coverage_ratio(target, input_str, interval_types):
    
    special_intervals = parse_special_intervals(input_str)
    
    x, y = target
    fragment_length = y - x
    
    special_ratios = {}
    total_covered_length = 0
    special_intersections = {}
    total_intersection_points = set()

    for i, (interval_type, x1, y1) in enumerate(special_intervals):

        if interval_type not in interval_types:
            continue
        
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


def process_dataframe(df, top_col, interval_types):
    
    for idx, row in df.iterrows():
        target_interval_str = row[f"{top_col}_D"]
        input_str = row[f"{top_col}_cover"]

        target_interval = tuple(map(int, target_interval_str[1:-1].split(',')))

        if not isinstance(input_str, str):
            input_str = str(input_str)

        total_ratio, special_ratios, special_intersections, total_intersection_points = calculate_coverage_ratio(target_interval, input_str, interval_types)
        
        df.at[idx, f"{top_col}_colate"] = round(total_ratio * 100, 1)
        

        for interval_type in interval_types:
            col_name = f"{interval_type.capitalize()}_{top_col}_colate"
            df.at[idx, col_name] = round(special_ratios.get(interval_type, 0)*100, 1)
    
    return df






