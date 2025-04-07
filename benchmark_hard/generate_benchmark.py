import os
import re
import json
import time
from typing import List
from tqdm import tqdm

from LLM import LLM_api

problem_type_dict = {
    'cvrp': "Capacitated Vehicle Routing Problem (CVRP)",
    'ovrp': "Open Vehicle Routing Problem (OVRP)",
    'vrpl': "Distance Constrained Capacitated Vehicle Routing Problem (DCVRPB)",
    'vrpb': "Vehicle Routing Problem with Backhauls (VRPB)",
    'vrpmb': "Vehicle Routing Problem with Mixed Backhauls (VRPMB)",
    'vrptw': "Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)",
}


def generate_problem_desc(problem_type: str, generate_num: int = 5, another_title_list=None) -> str:
    if problem_type == 'cvrp':
        from seed_cvrp import cvrp_data_seed
        problem_data_seed = cvrp_data_seed
    elif problem_type == 'ovrp':
        from seed_ovrp import ovrp_data_seed
        problem_data_seed = ovrp_data_seed
    elif problem_type == 'vrpl':
        from seed_vrpl import vrpl_data_seed
        problem_data_seed = vrpl_data_seed
    elif problem_type == 'vrpb':
        from seed_vrpb import vrpb_data_seed
        problem_data_seed = vrpb_data_seed
    elif problem_type == 'vrpmb':
        from seed_vrpmb import vrpmb_data_seed
        problem_data_seed = vrpmb_data_seed
    elif problem_type == 'vrptw':
        from seed_vrptw import vrptw_data_seed
        problem_data_seed = vrptw_data_seed
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")
    title_list = [scenario['title'] for scenario in problem_data_seed]
    title_list.extend(another_title_list)
    title_prompt = "**Existing Scenarios:**\n"
    for title in title_list:
        title_prompt += f"- {title}\n"
    problem_type_prompt = (
            f"I need to generate diverse real-world application scenarios for the **{problem_type_dict[problem_type]}**. While I already have some problem instances, I want to expand their variety.\n\n" +
            f"{title_prompt}\n" +
            f"**Request:**\n" +
            f"Please generate **{generate_num} additional scenario titles and descriptions**, formatted as follows:\n" +
            f"```\n**Scenario <ID>:** <Title>\n<Description>\n---\n```\n" +
            f"Describe these scenarios like you're telling a story to your neighbor - keep it simple, practical, and grounded in everyday experiences.\n"
            f"Here are a few examples:\n\n"
    )

    for data_idx, data in enumerate(problem_data_seed, start=1):
        problem_type_prompt += (f"**Scenario {data_idx}:** {data['title']}\n\n")
        problem_type_prompt += (f"{data['desc_split']}\n\n")
        problem_type_prompt += (f"---\n\n")

    return problem_type_prompt


def extract_templates(text: str) -> List[dict]:
    """Extract CVRP templates from formatted text"""
    scenario_list = []
    separator = '---'
    segments = [segment.strip() for segment in text.split(separator) if segment.strip()]
    pattern = r"Scenario \d+:"
    for segment in segments:
        seg_lines = segment.splitlines()
        start_idx = 0
        scenario_title = ""
        for line_idx, line in enumerate(seg_lines):
            if re.search(pattern, line):
                scenario_title = line.split(":")[1]
                start_idx = line_idx + 1
                break
        if len(scenario_title) > 0:
            scenario_contents = "\n".join(seg_lines[start_idx:])
            tags = re.findall(r"<(\w+)>", scenario_contents)
            scenario_list.append({
                "title": scenario_title.strip(' *'),
                "content": scenario_contents.strip(),
                "tags": tags,
            })

    return scenario_list


if __name__ == '__main__':
    problem_type = 'cvrp'
    llm = LLM_api(model="deepseek-chat", )
    time_start = time.time()
    generated_problem_type_path = f"./data/{problem_type}_meta.json"
    total_generate_num = 100
    batch_generate_num = 5
    iter_num = (total_generate_num + batch_generate_num - 1) // batch_generate_num
    for iter_id in tqdm(range(iter_num)):
        existed_scenario_list = []
        if os.path.exists(generated_problem_type_path):
            with open(generated_problem_type_path, "r") as f:
                existed_scenario_list = json.load(f)
        existed_title_list = [scenario['title'] for scenario in existed_scenario_list]

        iter_generate_num = min(batch_generate_num, total_generate_num - iter_id * batch_generate_num)
        prompt = generate_problem_desc(problem_type=problem_type, generate_num=iter_generate_num,
                                       another_title_list=existed_title_list)
        # print(prompt)
        text = llm.get_text(content=prompt)
        end_time = time.time()
        # print(text)
        print(f"Time taken: {end_time - time_start}")
        generated_scenario_list = extract_templates(text)
        existed_scenario_list.extend(generated_scenario_list)
        with open(generated_problem_type_path, "w") as f:
            json.dump(existed_scenario_list, f, indent=4)
        for scenario in generated_scenario_list:
            print(f"Generating scenario: {scenario['title']}")
