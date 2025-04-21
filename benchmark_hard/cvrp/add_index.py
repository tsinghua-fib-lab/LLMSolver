import os
import json

def add_index_to_json_files(directory):
    # 遍历指定目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            
            # 读取 JSON 文件
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # 为每个元素添加 index 属性
            for idx, item in enumerate(data):
                item['index'] = idx
            
            # 保存修改后的 JSON 文件
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4, ensure_ascii=False)

# 指定 dataset 目录路径
dataset_directory = "/data1/shy/zgc/llm_solver/LLMSolver/benchmark_hard/cvrp/dataset"
add_index_to_json_files(dataset_directory)
