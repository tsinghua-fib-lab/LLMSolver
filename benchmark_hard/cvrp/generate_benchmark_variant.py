import os
import re
import json
import time
from typing import List
from tqdm import tqdm

from benchmark_hard.LLM import LLM_api

features_desc = {
    "C": "*Capacity (C)*: Each vehicle has a maximum capacity :math:`Q`, restricting the total load that can be in the vehicle at any point of the route. The route must be planned such that the sum of demands and pickups for all customers visited does not exceed this capacity.",
    "TW": "*Time Windows (TW)*: Every node :math:`i` has an associated time window :math:`[e_i, l_i]` during which service must commence. Additionally, each node has a service time :math:`s_i`. Vehicles must reach node :math:`i` within its time window; early arrivals must wait at the node location until time :math:`e_i`.",
    "O": "*Open Routes (O)*: You should emphasize in your description that once the vehicle has completed its route and served all its customers, it does not need to return to the depot",
    "B": "*Backhauls (B)*: Backhauls generalize demand to account for return shipments. Customers are categorized as either linehaul or backhaul. Linehaul customers require delivery of goods from the depot to the customer, while backhaul customers require pickup of goods that are transported from the client back to the depot. You should emphasize in your description that any linehaul customers must precede the backhaul customers in the route, ensuring that deliveries are made before pickups are scheduled.",
    "L": "*Distance Limits (L)*: Restricts each route to a maximum allowable travel distance, ensuring equitable distribution of route lengths across vehicles.",
    "MB": "*Mixed (M) Backhaul (B)*: Linehaul (deliveries) and backhaul (pickups) can be sequenced in any order along a route. You should emphasize in your description that customers could be served in any order.",
    "MD": "*Multi-depot (MD)*: Generalizes single-depot (m = 1) variants with multiple depot nodes m > 1 from which vehicles can start their tour. Each vehicle must return to its starting depot. This variant requires decisions about depot-customer assignments, making the problem more realistic for organizations operating from multiple facilities.",
}
problem_type_base = ['cvrp', 'ovrp', 'vrpb', 'vrpl', 'vrptw', 'vrpmb', 'mdcvrp']
problem_type_dict = {
    'cvrp': "Capacitated Vehicle Routing Problem (CVRP)",
    'ovrp': "Open Vehicle Routing Problem (OVRP)",
    'vrpb': "Vehicle Routing Problem with Backhauls (VRPB)",
    'vrpl': "Vehicle Routing Problem with Duration Limits (VRPL)",
    'vrptw': "Vehicle Routing Problem with Time Windows (VRPTW)",
    'ovrptw': "Open Vehicle Routing Problem with Time Windows (OVRPTW)",
    'ovrpb': "Open Vehicle Routing Problem with Backhauls (OVRPB)",
    'ovrpl': "Open Vehicle Routing Problem with Duration Limits (OVRPL)",
    'vrpbl': "Vehicle Routing Problem with Backhauls and Duration Limits (VRPBL)",
    'vrpbtw': "Vehicle Routing Problem with Backhauls and Time Windows (VRPBTW)",
    'vrpltw': "Vehicle Routing Problem with Duration Limits and Time Windows (VRPLTW)",
    'ovrpbl': "Open Vehicle Routing Problem with Backhauls and Duration Limits (OVRPBL)",
    'ovrpbtw': "Open Vehicle Routing Problem with Backhauls and Time Windows (OVRPBTW)",
    'ovrpltw': "Open Vehicle Routing Problem with Duration Limits and Time Windows (OVRPLTW)",
    'vrpbltw': "Vehicle Routing Problem with Backhauls, Duration Limits and Time Windows (VRPBLTW)",
    'ovrpbltw': "Open Vehicle Routing Problem with Backhauls, Duration Limits and Time Windows (OVRPBLTW)",
    'vrpmb': "Vehicle Routing Problem with Mixed Backhauls (VRPMB)",
    'ovrpmb': "Open Vehicle Routing Problem with Mixed Backhauls (OVRPMB)",
    'vrpmbl': "Vehicle Routing Problem with Mixed Backhauls and Duration Limits (VRPMBL)",
    'vrpmbtw': "Vehicle Routing Problem with Mixed Backhauls and Time Windows (VRPMBTW)",
    'ovrpmbl': "Open Vehicle Routing Problem with Mixed Backhauls and Duration Limits (OVRPMBL)",
    'ovrpmbtw': "Open Vehicle Routing Problem with Mixed Backhauls and Time Windows (OVRPMBTW)",
    'vrpmbltw': "Vehicle Routing Problem with Mixed Backhauls, Duration Limits and Time Windows (VRPMBLTW)",
    'ovrpmbltw': "Open Vehicle Routing Problem with Mixed Backhauls, Duration Limits and Time Windows (OVRPMBLTW)",

    'mdcvrp': "Multi-Depots Capacitated Vehicle Routing Problem (MDCVRP)",
    'mdovrp': "Multi-Depots Open Vehicle Routing Problem (MDOVRP)",
    'mdvrpb': "Multi-Depots Vehicle Routing Problem with Backhauls (MDVRPB)",
    'mdvrpl': "Multi-Depots Vehicle Routing Problem with Duration Limits (MDVRPL)",
    'mdvrptw': "Multi-Depots Vehicle Routing Problem with Time Windows (MDVRPTW)",
    'mdovrptw': "Multi-Depots Open Vehicle Routing Problem with Time Windows (MDOVRPTW)",
    'mdovrpb': "Multi-Depots Open Vehicle Routing Problem with Backhauls (MDOVRPB)",
    'mdovrpl': "Multi-Depots Open Vehicle Routing Problem with Duration Limits (MDOVRPL)",
    'mdvrpbl': "Multi-Depots Vehicle Routing Problem with Backhauls and Duration Limits (MDVRPBL)",
    'mdvrpbtw': "Multi-Depots Vehicle Routing Problem with Backhauls and Time Windows (MDVRPBTW)",
    'mdvrpltw': "Multi-Depots Vehicle Routing Problem with Duration Limits and Time Windows (MDVRPLTW)",
    'mdovrpbl': "Multi-Depots Open Vehicle Routing Problem with Backhauls and Duration Limits (MDOVRPBL)",
    'mdovrpbtw': "Multi-Depots Open Vehicle Routing Problem with Backhauls and Time Windows (MDOVRPBTW)",
    'mdovrpltw': "Multi-Depots Open Vehicle Routing Problem with Duration Limits and Time Windows (MDOVRPLTW)",
    'mdvrpbltw': "Multi-Depots Vehicle Routing Problem with Backhauls, Duration Limits and Time Windows (MDVRPBLTW)",
    'mdovrpbltw': "Multi-Depots Open Vehicle Routing Problem with Backhauls, Duration Limits and Time Windows (MDOVRPBLTW)",
    'mdvrpmb': "Multi-Depots Vehicle Routing Problem with Mixed Backhauls (MDVRPMB)",
    'mdovrpmb': "Multi-Depots Open Vehicle Routing Problem with Mixed Backhauls (MDOVRPMB)",
    'mdvrpmbl': "Multi-Depots Vehicle Routing Problem with Mixed Backhauls and Duration Limits (MDVRPMBL)",
    'mdvrpmbtw': "Multi-Depots Vehicle Routing Problem with Mixed Backhauls and Time Windows (MDVRPMBTW)",
    'mdovrpmbl': "Multi-Depots Open Vehicle Routing Problem with Mixed Backhauls and Duration Limits (MDOVRPMBL)",
    'mdovrpmbtw': "Multi-Depots Open Vehicle Routing Problem with Mixed Backhauls and Time Windows (MDOVRPMBTW)",
    'mdvrpmbltw': "Multi-Depots Vehicle Routing Problem with Mixed Backhauls, Duration Limits and Time Windows (MDVRPMBLTW)",
    'mdovrpmbltw': "Multi-Depots Open Vehicle Routing Problem with Mixed Backhauls, Duration Limits and Time Windows (MDOVRPMBLTW)",
}

problem_type_features_dict = {
    'cvrp': ["C"],
    'ovrp': ["C", "O"],
    'vrpb': ["C", "B"],
    'vrpl': ["C", "L"],
    'vrptw': ["C", "TW"],
    'ovrptw': ["C", "O", "TW"],
    'ovrpb': ["C", "O", "B"],
    'ovrpl': ["C", "O", "L"],
    'vrpbl': ["C", "B", "L"],
    'vrpbtw': ["C", "B", "TW"],
    'vrpltw': ["C", "L", "TW"],
    'ovrpbl': ["C", "O", "B", "L"],
    'ovrpbtw': ["C", "O", "B", "TW"],
    'ovrpltw': ["C", "O", "L", "TW"],
    'vrpbltw': ["C", "B", "L", "TW"],
    'ovrpbltw': ["C", "O", "B", "L", "TW"],
    'vrpmb': ["C", "MB"],
    'ovrpmb': ["C", "O", "MB"],
    'vrpmbl': ["C", "MB", "L"],
    'vrpmbtw': ["C", "MB", "TW"],
    'ovrpmbl': ["C", "O", "MB", "L"],
    'ovrpmbtw': ["C", "O", "MB", "TW"],
    'vrpmbltw': ["C", "MB", "L", "TW"],
    'ovrpmbltw': ["C", "O", "MB", "L", "TW"],

    'mdcvrp': ["C", "MD"],
    'mdovrp': ["C", "MD", "O"],
    'mdvrpb': ["C", "MD", "B"],
    'mdvrpl': ["C", "MD", "L"],
    'mdvrptw': ["C", "MD", "TW"],
    'mdovrptw': ["C", "MD", "O", "TW"],
    'mdovrpb': ["C", "MD", "O", "B"],
    'mdovrpl': ["C", "MD", "O", "L"],
    'mdvrpbl': ["C", "MD", "B", "L"],
    'mdvrpbtw': ["C", "MD", "B", "TW"],
    'mdvrpltw': ["C", "MD", "L", "TW"],
    'mdovrpbl': ["C", "MD", "O", "B", "L"],
    'mdovrpbtw': ["C", "MD", "O", "B", "TW"],
    'mdovrpltw': ["C", "MD", "O", "L", "TW"],
    'mdvrpbltw': ["C", "MD", "B", "L", "TW"],
    'mdovrpbltw': ["C", "MD", "O", "B", "L", "TW"],
    'mdvrpmb': ["C", "MD", "MB"],
    'mdovrpmb': ["C", "MD", "O", "MB"],
    'mdvrpmbl': ["C", "MD", "MB", "L"],
    'mdvrpmbtw': ["C", "MD", "MB", "TW"],
    'mdovrpmbl': ["C", "MD", "O", "MB", "L"],
    'mdovrpmbtw': ["C", "MD", "O", "MB", "TW"],
    'mdvrpmbltw': ["C", "MD", "MB", "L", "TW"],
    'mdovrpmbltw': ["C", "MD", "O", "MB", "L", "TW"],
}


def generate_problem_base_desc(feature: str):
    if feature == "C":
        from benchmark_hard.cvrp.seed_cvrp import cvrp_data_seed
        problem_data_seed = cvrp_data_seed
    elif feature == 'O':
        from benchmark_hard.cvrp.seed_ovrp import ovrp_data_seed
        problem_data_seed = ovrp_data_seed
    elif feature == 'L':
        from benchmark_hard.cvrp.seed_vrpl import vrpl_data_seed
        problem_data_seed = vrpl_data_seed
    elif feature == 'B':
        from benchmark_hard.cvrp.seed_vrpb import vrpb_data_seed
        problem_data_seed = vrpb_data_seed
    elif feature == 'MB':
        from benchmark_hard.cvrp.seed_vrpmb import vrpmb_data_seed
        problem_data_seed = vrpmb_data_seed
    elif feature == 'TW':
        from benchmark_hard.cvrp.seed_vrptw import vrptw_data_seed
        problem_data_seed = vrptw_data_seed
    elif feature == 'MD':
        from benchmark_hard.cvrp.seed_mdcvrp import mdcvrp_data_seed
        problem_data_seed = mdcvrp_data_seed
    else:
        raise ValueError(f"Unknown problem feature: {feature}")

    problem_base_prompt = f"Here are a few examples for feature ({feature}):\n\n"
    for data_idx, data in enumerate(problem_data_seed, start=1):
        problem_base_prompt += (f"**Scenario {data_idx}:** {data['title']}\n\n")
        problem_base_prompt += (f"{data['desc_split']}\n\n")
        problem_base_prompt += (f"---\n\n")

    return problem_base_prompt


def generate_problem_desc(problem_type: str, generate_num: int = 5, title_list=None) -> str | None:
    if problem_type in problem_type_base:
        return None
    problem_features = problem_type_features_dict[problem_type]
    problem_feature_prompt = f"The Problem has the following features: {problem_features}\n\n"
    for feature in problem_features:
        problem_feature_prompt += f"{features_desc[feature]}\n"
        problem_feature_prompt += generate_problem_base_desc(feature)

    problem_feature_prompt += "Besides, you should **not include** the following constraints:\n"
    for feature in features_desc:
        if feature in problem_features:
            continue
        problem_feature_prompt += f"- (Not Include) {features_desc[feature]}\n"

    title_prompt = "**Existing Scenarios:**\n"
    for title in title_list:
        title_prompt += f"- {title}\n"
    problem_type_prompt = (
            f"I need to generate diverse real-world application scenarios for the **{problem_type_dict[problem_type]}**. While I already have some problem instances, I want to expand their variety.\n\n" +
            f"{title_prompt}\n" +
            f"**Request:**\n" +
            f"Please generate **{generate_num} additional scenario titles and descriptions**, formatted as follows:\n" +
            f"```\n**Scenario <ID>:** <Title>\n<Description>\n---\n```\n" +
            f"Please use diverse language to describe these scenarios. All scenarios must be described in English.\n\n" +
            problem_feature_prompt
    )

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
    var_problem_type_list = ['cvrp', 'ovrp', 'vrpb', 'vrpl', 'vrptw', 'ovrptw', 'ovrpb', 'ovrpl', 'vrpbl', 'vrpbtw',
                             'vrpltw', 'ovrpbl', 'ovrpbtw', 'ovrpltw', 'vrpbltw', 'ovrpbltw', 'vrpmb', 'ovrpmb',
                             'vrpmbl', 'vrpmbtw', 'ovrpmbl', 'ovrpmbtw', 'vrpmbltw', 'ovrpmbltw',

                             'mdcvrp', 'mdovrp', 'mdvrpb', 'mdvrpl', 'mdvrptw', 'mdovrptw', 'mdovrpb', 'mdovrpl',
                             'mdvrpbl', 'mdvrpbtw', 'mdvrpltw', 'mdovrpbl', 'mdovrpbtw', 'mdovrpltw', 'mdvrpbltw',
                             'mdovrpbltw', 'mdvrpmb', 'mdovrpmb', 'mdvrpmbl', 'mdvrpmbtw', 'mdovrpmbl', 'mdovrpmbtw',
                             'mdvrpmbltw', 'mdovrpmbltw']

    var_problem_type_list_1 = ['mdvrpbl', 'mdvrpbtw', 'mdvrpltw', 'mdovrpbl', 'mdovrpbtw', ]
    var_problem_type_list_2 = ['mdovrpbltw', 'mdvrpmb', 'mdovrpmb', 'mdvrpmbl', 'mdvrpmbtw', ]
    var_problem_type_list_3 = ['mdovrpltw', 'mdvrpbltw', 'mdvrpmbltw', 'mdovrpmbl', 'mdovrpmbtw', 'mdovrpmbltw']
    for problem_type in var_problem_type_list_3:
        print(problem_type)
        if problem_type in problem_type_base:
            continue
        llm = LLM_api(model="deepseek-chat", )
        time_start = time.time()
        generated_problem_type_path = f"./data_var/{problem_type}_meta.json"
        if os.path.exists(generated_problem_type_path):
            continue
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
                                           title_list=existed_title_list)
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
