
import requests
import numpy as np
import json
from itertools import islice
import re
import os
import ast
import random
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "sk-xnflapkfclgydebocytvgkjvcnhngtbbtibxbetcwgwnaoco"
#MODEL_NAME = "THUDM/GLM-4-32B-0414"
MODEL_NAME = "deepseek-ai/DeepSeek-V3"
#MODEL_NAME ="deepseek-ai/DeepSeek-R1"
#filepath1='benchmark/td_new/VRPMBTW.npz'
filepath2='benchmark/knapsack/kp.json'
Sector=['Resource Allocation in Cloud Computing', 'Inventory Management for Retail', 'Military Equipment Deployment', 'Energy Grid Optimization', 'Software Feature Prioritization', 'Wildlife Conservation Planning', 'Conference Scheduling', 'Sports Team Formation', 'Academic Course Selection', 'Medical Treatment Prioritization', 'Ship Cargo Loading', 'Digital Storage Allocation', 'Adventure Backpack Packing', 'Music Festival Scheduling', 'Home Renovation Budgeting', 'Game Loot Selection', 'Movie Production Budgeting', 'Vaccine Distribution Planning', 'Crop Planting Strategy', 'Space Mission Payload Selection', 'Charitable Donation Allocation', 'Movie Theater Snack Menu Design', 'Product Line Optimization', 'Cybersecurity Patch Deployment', 'Academic Research Grant Allocation', 'Travel Itinerary Planning', 'Industrial Equipment Maintenance', 'Data Center Server Procurement', 'Construction Material Selection', 'Movie Merchandise Production', 'Emergency Relief Supplies Packing', 'Fashion Retail Store Stocking', 'Film Festival Programming', 'Logistics Truck Loading', 'AI Model Feature Inclusion', 'Custom Jewelry Design', 'Athlete Nutrition Planning', 'Vehicle Fleet Optimization', 'Gift Wrapping Efficiency', 'Job Interview Scheduling', 'Warehouse Rack Space Allocation', 'Chemical Compound Selection', 'Smartphone Component Assembly', 'E-commerce Order Shipping', 'Tourist Attraction Planning', 'Luxury Car Customization', 'Music Album Track Listing', 'VIP Event Guest Selection', 'Ancient Artifact Acquisition', 'Athlete Sponsorship Allocation', 'Startup Hiring Strategy', 'Catering Menu Planning', 'Real Estate Investment', 'Pharmaceutical Drug Development', 'Movie Script Selection', 'Startup Office Layout Design', 'Wedding Catering Menu Design', 'Digital Art Collection Curation', 'R&D Investment Allocation', 'Disaster Relief Packing', 'Luxury Watch Collection Curation', 'AI Training Data Selection', 'Urban Park Amenity Planning', 'Athlete Meal Prep Planning', 'Jewelry Thief Heist', 'Spacecraft Payload Optimization', 'Luxury Hotel Menu Design', 'Auction House Art Acquisition', 'Formula 1 Pit Stop Tool Selection', 'Supply Chain Container Loading', 'Hiking Gear Selection', 'Film Equipment Rental', 'Pharmacy Shelf Stocking', 'Personal Library Curation', 'Restaurant Menu Design', 'Luxury Yacht Interior Design', 'Fashion Week Model Selection', 'Cargo Plane Loading', 'Museum Exhibit Transport', 'Backcountry Research Kit Assembly', 'Smartphone Feature Integration', 'Emergency Shelter Supply Packing']
#problem_type="VRPMBTW"
with open("benchmark/knapsack/prompt.json", "r", encoding="utf-8") as f:
    data_prompt = json.load(f) 
with open("benchmark/knapsack/seed.json", "r", encoding="utf-8") as f:
    data_seed_prompt = json.load(f)    
def generate_uniform_numbers(n):
     return [random.uniform(0, 1) for _ in range(n)]
for key in ["knapsack"]:
    problem_type=key
    num_requests = 1  # 定义你想调用 API 的次数
    for i in range(num_requests-1,num_requests+5):      
        Sector_process=','.join(Sector)
        user_input = data_prompt[key]+data_seed_prompt[key]+"The scenario title for each generated example must not duplicate any in the following list, meaning each application scenario must be distinct and cannot be the same as or similar to the ones in the list below:"+Sector_process+" ,Note that the language style of different examples must vary, and it should not be apparent that they were asked by the same person."
        number=5
        user_input = user_input.format(number=number)
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "请按照我的要求生成不同的问题描述。"},
                {"role": "user", "content": user_input}
            ],
            "temperature": 1.3,
            "max_tokens": 8000
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        # 调用 API
        response = requests.post(API_URL, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"]
            total_tokens = response.json().get("usage", {}).get("total_tokens", None)
#            print(result)
            print(f"Total tokens used: {total_tokens}")
            resulttt = result.split('&&&')
#            for index, result in enumerate(resulttt, start=1):
#            print("resulttt:",resulttt)
            for  result in resulttt:
#            result = result.replace("\u2019", "'")
                match = re.search(r"##(.*?)##", result)
                title = match.group(1) if match else None
                if title==None:
                    print("null_exist;")
                else:
                   Sector.append(title)


# 删除双井号及其中的内容
                   text_without_title = re.sub(r"##.*?##", "", result).strip()

                   responses=[]
                   responses.append(text_without_title)  # 保存每次的回答
                   responses = [response.replace("\n", "") for response in responses]
                   responses = responses[0]
 #           responses=str(responses)
#						new_text = "6666新增的一条文本。"
# 追加模式 ('a') 打开文件，并写入新的 JSON 行
                   item_weight = generate_uniform_numbers(20)
                   item_value  = generate_uniform_numbers(20)
                   knapsack_capacity=5
                   desc_merge=responses.replace('<item_weight>', str(item_weight))
                   desc_merge=desc_merge.replace('<item_value>', str(item_value))
                   desc_merge=desc_merge.replace('<knapsack_capacity>', str(knapsack_capacity))
                   add_data = {
                        "title":title,
                        "desc_split": responses,
                        "desc_merge":desc_merge,                    
                        "data_template": {
                            "item_weight":item_weight,
                            "item_value": item_value,
                            "knapsack_capacity": knapsack_capacity
                        },
                        "label": problem_type
                       }
                   try:
                      with open(filepath2, "r") as file:
                          data_list = json.load(file)  # 读取文件中的数据
                   except (FileNotFoundError, json.JSONDecodeError):  # 文件不存在或为空时
                       data_list = []
# 向列表中追加新数据
                   data_list.append(add_data)

# 将更新后的数据写回 JSON 文件
                   with open(filepath2, "w") as file:
                      json.dump(data_list, file, indent=4)  #
            
#            if not os.path.exists("benchmark/description_generate/description/cvrp.json"):
#                     with open("benchmark/description_generate/description/cvrp.json", "w", encoding="utf-8") as f:
#                       json.dump([], f, ensure_ascii=False)
#            with open("benchmark/description_generate/description/cvrp.json", "a", encoding="utf-8") as f:
#                 json.dump({"label": "cvrp", "texts": add_data}, f, ensure_ascii=False)
#                 f.write("\n")  # 确保每条数据单独一行
            # 将结果写入文本文件
#            file.write(f"{result}\n\n")
            print(f"调用 {i + 1} 成功，结果已写入文件")
        else:
#            file.write(f"调用 {i + 1} 失败，错误代码: {response.status_code}\n\n")
            print(f"调用 {i + 1} 失败，错误代码: {response.status_code}")
        print(Sector)