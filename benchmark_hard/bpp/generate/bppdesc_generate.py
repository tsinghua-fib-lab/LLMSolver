#-----------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------

'''
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
#filepath2='benchmark/bpp/2DOFBPP.json'
Sector=[]
#problem_type="VRPMBTW"
with open("benchmark/bpprompt.json", "r", encoding="utf-8") as f:
    data_prompt = json.load(f) 
with open("benchmark/bppseedprompt.json", "r", encoding="utf-8") as f:
    data_seed_prompt = json.load(f)    
def generate_random_list(n):
    return [[random.randint(1, 10) for _ in range(2)] for _ in range(n)]
#"VRPLTW","OVRPLTW","VRPBLTW","OVRPBLTW","VRPMBL","VRPMBLTW", "OVRPMBLTW"
for key in ["2DOFBPPR","2DONBPP","2DONBPPR"]:
    problem_type=key
    if problem_type=="2DOFBPPR":
       Sector=[]
       bin_status="false",
       can_rotate="true",
       dimension="2D"
       filepath2=f'benchmark/bpp/{key}.json'
    if problem_type=="2DONBPP":
       Sector=[]
       bin_status="true",
       can_rotate="false",
       dimension="2D"
       filepath2=f'benchmark/bpp/{key}.json'
    if problem_type=="2DONBPPR":
       Sector=[]
       bin_status="true",
       can_rotate="true",
       dimension="2D"
       filepath2=f'benchmark/bpp/{key}.json'
    num_requests = 1  # 定义你想调用 API 的次数
    for i in range(num_requests-1,num_requests+22):      
#        print("Sectorlist:",Sector)
        Sector_process=','.join(Sector)
#        print("Sectorlist:",Sector)
#        print("Sector_process:",Sector_process)
        user_input = data_prompt[key]+data_seed_prompt[key]+"The industry scenario title for each generated example must not duplicate any in the following list:"+Sector_process
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



                   text_without_title = re.sub(r"##.*?##", "", result).strip()

                   responses=[]
                   responses.append(text_without_title)  # 保存每次的回答
                   responses = [response.replace("\n", "") for response in responses]
                   responses = responses[0]
 #           responses=str(responses)

                   n_items = random.randint(20, 25)
                   items_size=generate_random_list(n_items)
                   desc_merge=responses.replace('<bin_size>', str([10,10]))
                   desc_merge=desc_merge.replace('<items_size>', str(items_size))
                   add_data = {
                        "title":title,
                        "desc_split": responses,
                        "desc_merge":desc_merge,
                        "data_template": {
                            "bin_size":[10,10],
                            "items_size": items_size,
                            "bin_status":bin_status,
                            "can_rotate":can_rotate,
                            "dimension":dimension,
                        },
                        "label": problem_type
                       }
                   try:
                      with open(filepath2, "r") as file:
                          data_list = json.load(file)  # 读取文件中的数据
                   except (FileNotFoundError, json.JSONDecodeError):  # 文件不存在或为空时
                       data_list = []

                   data_list.append(add_data)


                   with open(filepath2, "w") as file:
                      json.dump(data_list, file, indent=4)  #
            
            print(f"调用 {i + 1} 成功，结果已写入文件")
        else:
#            file.write(f"调用 {i + 1} 失败，错误代码: {response.status_code}\n\n")
            print(f"调用 {i + 1} 失败，错误代码: {response.status_code}")
