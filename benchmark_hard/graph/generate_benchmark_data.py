import json

from envs.graph.generator import GraphGenerator


def get_user_template_data(data_template_dict: dict, tags: list) -> dict:
    """Extract user template data from the given dictionary"""
    user_template_dict = {}
    for tag in tags:
        if tag in data_template_dict:
            user_template_dict[tag] = data_template_dict[tag]
        else:
            raise NotImplementedError(f"Tag {tag} not implemented")
    return user_template_dict


def generate_data(problem_type):
    generator = GraphGenerator(problem_type=problem_type)
    data = generator.generate(1)[0]
    return data


problem_type_list = ['maxcut', 'mis', 'mvc']


if __name__ == '__main__':
    problem_type_dir = "data"
    for problem_type in problem_type_list:
        print(problem_type)

        generated_problem_type_path = f"./{problem_type_dir}/{problem_type}_meta.json"
        with open(generated_problem_type_path, "r") as f:
            scenario_list = json.load(f)
        scenario_data_list = []
        for scenario_idx, scenario in enumerate(scenario_list):
            # Generate data for each scenario
            data_dict = generate_data(problem_type)
            print(scenario["title"])
            scenario_data = {
                "title": scenario["title"],
                "desc_split": scenario["content"],
                "data_template": data_dict,
                "user_template": get_user_template_data(data_dict, scenario["tags"]),
                "label": problem_type,
                "index": scenario_idx,
            }
            scenario_data_list.append(scenario_data)
        scenario_problem_type_path = f"./{problem_type_dir}/{problem_type}.json"
        with open(scenario_problem_type_path, "w") as f:
            json.dump(scenario_data_list, f, indent=4)
