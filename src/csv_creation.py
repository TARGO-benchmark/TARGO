import json
import csv
import os

# 读取category.json文件
json_file = 'eval_results_train_full-medium-occlusion-1000/category.json'
with open(json_file, 'r') as f:
    category_map = json.load(f)

# 创建CSV文件
csv_file = 'object_categories.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    
    # 写入表头
    writer.writerow(["Name", "Category"])
    
    # 写入数据行
    for obj_id, category in category_map.items():
        writer.writerow([f'"{obj_id}"', category])

print(f"CSV文件已创建: {csv_file}")