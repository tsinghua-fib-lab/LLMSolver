#-----------------------------------------------------------------------------------------
'''
import requests
import numpy as np
from itertools import islice
import re
def get_lines_as_array(filepath, start, num_lines):
    with open(filepath, 'r', encoding='utf-8') as file:
        return [line.strip().split() for line in islice(file, start, start + num_lines)]
filepathloc='data/cvrp/val/chai/locs.txt'
filepathlinehau='data/cvrp/val/chai/demand_linehaul.txt'
responses=[]
result = "A waste management company needs to collect recyclable materials from 10 neighborhoods in a city. The central recycling facility and the neighborhoods are located at coordinates represented as aaa, and the demand for recyclable materials from each neighborhood is represented as bbb. The collection trucks have a maximum capacity of 1 ton, and each truck must start and end its route at the central facility. The goal is to plan the routes to ensure all neighborhoods are visited exactly once without exceeding the truck capacity."
aa=get_lines_as_array(filepathloc, 1, 21)
#        aa = np.array([[0.191519, 0.622109], [0.437728,0.785359],[0.779976,0.272593]])
#        bb=np.array([1, 2])
bb=get_lines_as_array(filepathlinehau,1, 20) #列表
aa = np.array(aa, dtype=float)
bb = np.array(bb, dtype=float)
result = result.replace('aaa', str(aa))  # 将aaa替换为数组array_aaa
result = result.replace('bbb', str(bb))  # 将bbb替换为数组array_bbb
responses.append(result)  # 保存每次的回答

responses = [response.replace("\n", "") for response in responses]
with open("intput.txt", "a", encoding="utf-8") as file:
    for response in responses:
        file.write(f"{response}\n")  # 每个回答后加上换行符，使其在文件中占一行
#print(responses)

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
#   cvrp     user_input = f"""假设有一个vrp问题,有一个仓库和{n}个客户,仓库和客户的坐标为a={aa},车辆最大载重为1,各个客户的需求分别为b={bb},每辆车必须从仓库出发并最终返回仓库,目标是安排车辆路线,确保所有客户被访问且仅访问一次.请按照这个模板生成一个新的具体现实应用下的vrp问题.请用一段自然语言描述,要求应用场景和语言描述和之前不能重复或者相近，可以举任意的例子,语言描述风格和方式顺序要求多变,把a和b中的所有数据都准确无误地完整列出来,其中坐标和客户需求统一用列表表示，不要把仓库坐标单独列出，不要生成列表的解释,开头直接开始定义问题,生成的问题要求之前回答的不一样，用英文回答
#          """  # 可以根据需求修改输入
#ovrp       user_input = f"""有一个仓库和{n}个客户,仓库和客户坐标信息统一表示为aaa,客户的运输需求量表示为bbb,每个汽车容量为1,车速为1,每辆车必须从仓库出发但不用返回仓库,目标是安排车辆路线,确保所有客户被访问且仅访问一次同时总距离最短.要求:1.根据以上模板用英文生成一个具体现实应用下的问题描述.2.问题描述中仓库和客户坐标信息统一表示为一个符号aaa,客户需求量表示为符号bbb,3.描述中必须包含坐标aaa,客户需求量bbb,车辆容量为1,车速为1,不用返回仓库等信息,仓库坐标和客户坐标不要分开给出.4.描述中aaa,bbb仅出现一次,5.只生成问题描述,用一段话输出.6.要求每次生成的实际应用场景必须不能重复,语言描述千万不要重复"""
#vrpl        user_input = f"""有一个仓库和{n}个客户,仓库和客户坐标信息统一表示为aaa,客户的运输需求量表示为bbb,每条路线的距离限制表示为ccc,每个汽车容量为1,车速为1,每辆车必须从仓库出发并返回仓库,目标是安排车辆路线,确保所有客户被访问且仅访问一次同时总距离最短.要求:1.根据以上模板用英文生成一个具体现实应用下的问题描述.2.问题描述中仓库和客户坐标信息统一表示为一个符号aaa,客户需求量表示为符号bbb,距离限制表示为ccc,3.描述中必须包含坐标aaa,客户需求量bbb,距离限制ccc,车辆容量为1,车速为1,必须返回仓库等信息,仓库坐标和客户坐标不要分开给出.4.描述中aaa,bbb,ccc都仅出现一次,5.只生成问题描述,用一段话输出.6.要求每次生成的实际应用场景必须不能重复,语言描述千万不要重复"""
#VRPTW        user_input = f"""有一个仓库和{n}个客户,仓库和客户坐标信息统一表示为aaa,客户的运输需求量表示为bbb,客户的服务时间为ccc,客户的时间窗为ddd,每个汽车容量为1,车速为1,每辆车必须从仓库出发并返回仓库,目标是安排车辆路线,确保所有客户被访问且仅访问一次同时总距离最短.要求:1.根据以上模板用英文生成一个具体现实应用下的问题描述.2.问题描述中仓库和客户坐标信息统一表示为一个符号aaa,客户需求量表示为符号bbb,客户服务时间用符号ccc表示,时间窗用符号ddd表示,3.描述中必须包含坐标aaa,客户需求量bbb,客户服务时间ccc和时间窗ddd,车辆容量为1,车速为1,必须返回仓库等信息,仓库坐标和客户坐标不要分开给出.4.描述中aaa,bbb,ccc,ddd都仅出现一次,5.只生成问题描述,用一段话输出.6.要求每次生成的实际应用场景必须不能重复,语言描述千万不要重复"""   
#vrpb        user_input = f"""有一个仓库和{n}个客户,仓库和客户坐标信息统一表示为aaa,客户的干线运输需求(仓库向客户运输)表示为bbb,回城运输需求(客户向仓库运输)表示为ccc,每个汽车容量为1,车速为1,每辆车必须从仓库出发并返回仓库,且回城运输必须在干线运输完成之后才能进行,目标是安排车辆路线,确保所有客户被访问且仅访问一次同时总距离最短.要求:1.根据以上模板用英文生成一个具体现实应用下的问题描述.2.问题描述中仓库和客户坐标信息统一表示为一个符号aaa,客户干线需求表示为符号bbb,回城运输需求表示为ccc,3.描述中必须包含坐标aaa,干线运输需求bbb,回城运输ccc,车辆容量为1,车速为1,必须返回仓库等信息,仓库坐标和客户坐标不要分开给出.4.只生成问题描述,用一段话输出.5.要求每次生成的实际应用场景必须不能重复,语言描述千万不要重复,6.描述中aaa,bbb,ccc都仅出现一次,要求只出现一次一定要遵守,"""
#cvrp        user_input = f"""有一个仓库和{n}个客户,仓库和客户坐标信息统一表示为aaa,客户的运输需求量表示为bbb,每个汽车容量为1,车速为1,每辆车必须从仓库出发并返回仓库,目标是安排车辆路线,确保所有客户被访问且仅访问一次同时总距离最短.要求:1.根据以上模板用英文生成一个具体现实应用下的问题描述.2.问题描述中仓库和客户坐标信息统一表示为一个符号aaa,客户需求量表示为符号bbb,3.描述中必须包含坐标aaa,客户需求量bbb,车辆容量为1,车速为1,必须返回仓库等信息,仓库坐标和客户坐标不要分开给出.4.描述中aaa,bbb仅出现一次,5.只生成问题描述,用一段话输出.6.要求每次生成的实际应用场景必须不能重复,语言描述千万不要重复"""
#ovrptw        user_input = f"""在某个实际问题中有一个仓库和{n}个客户,仓库和客户坐标信息统一表示为aaa,客户的运输需求量表示为bbb,客户的服务时间为ccc,客户的时间窗为ddd,每个汽车容量为1,车速为1,每辆车必须从仓库出发但不用返回仓库,目标是安排车辆路线,确保所有客户被访问且仅访问一次同时总距离最短.要求:1.根据以上模板用英文生成一个具体现实应用下的问题描述.2.问题描述中仓库和客户坐标信息统一表示为一个符号aaa,客户需求量表示为符号bbb,客户服务时间和时间窗分别用符号ccc和ddd表示,3.描述中必须包含坐标aaa,客户需求量bbb,客户服务时间ccc和时间窗ddd,车辆容量为1,车速为1,不用返回仓库等信息,仓库坐标和客户坐标不要分开给出.4.描述中aaa,bbb,ccc,ddd都仅出现一次,5.只生成问题描述,用一段话输出.6.要求每次生成的实际应用场景必须不能重复,语言描述千万不要重复"""
#        user_input = f"""有一个仓库和{n}个客户,仓库和客户坐标信息统一表示为aaa,客户的干线运输需求(仓库向客户运输量)表示为bbb,回城运输需求(客户向仓库运输)表示为ccc,客户的服务时间为ddd,客户的时间窗为eee,每个汽车容量为1,车速为1,每辆车必须从仓库出发并返回仓库,回城运输和干线运输可以交替进行,目标是安排车辆路线,确保所有客户被访问且仅访问一次同时总距离最短.要求:1.根据以上模板用英文生成一个具体现实应用下的问题描述.2.问题描述中仓库和客户坐标信息统一表示为一个符号aaa,客户干线需求表示为符号bbb,回城运输需求表示为ccc,客户的服务时间为ddd,客户的时间窗为eee,3.描述中必须包含坐标aaa,干线运输需求bbb,回城运输ccc,服务时间ddd,时间窗eee,车辆容量为1,车速为1,必须返回仓库以及回城运输和干线运输可以交替进行等信息,仓库坐标和客户坐标不要分开给出.4.描述中aaa,bbb,ccc,ddd,eee都仅出现一次,一定要保证.5.只生成问题描述,用一段话输出.6.要求每次生成的实际应用场景必须不能重复,语言描述千万不要重复"""
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