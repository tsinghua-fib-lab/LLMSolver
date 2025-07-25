import os
import re
import json
import time
from typing import List
from tqdm import tqdm

from utils.LLM import LLM_api

problem_type_dict = {
    'cvrp': "Capacitated Vehicle Routing Problem (CVRP)",
    'ovrp': "Open Vehicle Routing Problem (OVRP)",
    'vrpl': "Distance Constrained Capacitated Vehicle Routing Problem (DCVRPB)",
    'vrpb': "Vehicle Routing Problem with Backhauls (VRPB)",
    'vrpmb': "Vehicle Routing Problem with Mixed Backhauls (VRPMB)",
    'vrptw': "Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)",
    'mdcvrp': "Multi-Depots Capacitated Vehicle Routing Problem (MDCVRP)",
}

features_desc = {
    "C": "*Capacity (C)*: Each vehicle has a maximum capacity :math:`Q`, restricting the total load that can be in the vehicle at any point of the route. The route must be planned such that the sum of demands and pickups for all customers visited does not exceed this capacity.",
    "TW": "*Time Windows (TW)*: Every node :math:`i` has an associated time window :math:`[e_i, l_i]` during which service must commence. Additionally, each node has a service time :math:`s_i`. Vehicles must reach node :math:`i` within its time window; early arrivals must wait at the node location until time :math:`e_i`.",
    "O": "*Open Routes (O)*: Vehicles are not required to return to the depot after serving all customers. Note that this does not need to be counted as a constraint since it can be modelled by setting zero costs on arcs returning to the depot :math:`c_{i0} = 0` from any customer :math:`i \\in C`, and not counting the return arc as part of the route.",
    "B": "*Backhauls (B)*: Backhauls generalize demand to also account for return shipments. Customers are either linehaul or backhaul customers. Linehaul customers require delivery of a demand :math:`q_i > 0` that needs to be transported from the depot to the customer, whereas backhaul customers need a pickup of an amount :math:`p_i > 0` that is transported from the client back to the depot. It is possible for vehicles to serve a combination of linehaul and backhaul customers in a single route, but then any linehaul customers must precede the backhaul customers in the route.",
    "L": "*Distance Limits (L)*: Restricts each route to a maximum allowable travel distance, ensuring equitable distribution of route lengths across vehicles.",
    "MB": "*Mixed (M) Backhaul (B)*: A backhaul constraint variant allowing linehaul (deliveries) and backhaul (pickups) customers to be sequenced in any order along a route, provided. You should emphasize in your description that customers may be served in any order",
    "MD": "*Multi-depot (MD)*: Generalizes single-depot (m = 1) variants with multiple depot nodes m > 1 from which vehicles can start their tour. Each vehicle must return to its starting depot. This variant requires decisions about depot-customer assignments, making the problem more realistic for organizations operating from multiple facilities.",

}
problem_type_features_dict = {
    'cvrp': ["C"],
    'ovrp': ["C", "O"],
    'vrpb': ["C", "B"],
    'vrpl': ["C", "L"],
    'vrpmb': ["C", "MB"],
    'vrptw': ["C", "TW"],
    'mdcvrp': ["C", "MD"],
}


def generate_problem_desc(problem_type: str, generate_num: int = 5, another_title_list=None) -> str:
    if problem_type == 'cvrp':
        from benchmark.cvrp.seed.seed_cvrp import cvrp_data_seed
        problem_data_seed = cvrp_data_seed
    elif problem_type == 'ovrp':
        from benchmark.cvrp.seed.seed_ovrp import ovrp_data_seed
        problem_data_seed = ovrp_data_seed
    elif problem_type == 'vrpl':
        from benchmark.cvrp.seed.seed_vrpl import vrpl_data_seed
        problem_data_seed = vrpl_data_seed
    elif problem_type == 'vrpb':
        from benchmark.cvrp.seed.seed_vrpb import vrpb_data_seed
        problem_data_seed = vrpb_data_seed
    elif problem_type == 'vrpmb':
        from benchmark.cvrp.seed.seed_vrpmb import vrpmb_data_seed
        problem_data_seed = vrpmb_data_seed
    elif problem_type == 'vrptw':
        from benchmark.cvrp.seed.seed_vrptw import vrptw_data_seed
        problem_data_seed = vrptw_data_seed
    elif problem_type == 'mdcvrp':
        from benchmark.cvrp.seed.seed_mdcvrp import mdcvrp_data_seed
        problem_data_seed = mdcvrp_data_seed
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")
    title_list = [scenario['title'] for scenario in problem_data_seed]
    title_list.extend(another_title_list)
    title_prompt = "**Existing Scenarios:**\n"
    for title in title_list:
        title_prompt += f"- {title}\n"

    problem_features = problem_type_features_dict[problem_type]
    feature_prompt = "Your description should **include** the following constraints:\n"
    for feature in problem_features:
        feature_prompt += f"- {features_desc[feature]}\n"
    feature_prompt += "Besides, you should **not include** the following constraints:\n"
    for feature in features_desc:
        if feature in problem_features:
            continue
        feature_prompt += f"- (Not Include) {features_desc[feature]}\n"

    problem_type_prompt = (
            f"I need to generate diverse real-world application scenarios for the **{problem_type_dict[problem_type]}**. While I already have some problem instances, I want to expand their variety.\n\n" +
            f"{title_prompt}\n" +
            f"**Request:**\n" +
            f"Please generate **{generate_num} additional scenario titles and descriptions**, formatted as follows:\n" +
            f"```\n**Scenario <ID>:** <Title>\n<Description>\n---\n```\n" +
            feature_prompt +
            f"Please use diverse language to describe these scenarios. Here are a few examples:\n\n"
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
    problem_type_base = ['cvrp', 'ovrp', 'vrpb', 'vrpl', 'vrptw', 'vrpmb', 'mdcvrp']
    for problem_type in problem_type_base:
        print(problem_type)
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
