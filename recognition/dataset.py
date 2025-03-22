import os
import json
import datetime
import numpy as np
from LLM import LLM_api
from rl4co.data.utils import load_npz_to_tensordict
from tensordict.tensordict import TensorDict
from routefinder.envs.mtvrp import MTVRPGenerator

# Initialize LLM API
llm = LLM_api(model="deepseek-reasoner", key_idx=0)

# Function to generate the natural language prompt
def generate_prompt(instance_data, scenario):
    prompt = f"You are a problem modeler and you are provided with an example of a VRP problem below, please translate it into a plain natural language description in {scenario} and output the description directly.\n"
    prompt += f"Specifically, the problem has the following parameters:\n"
    prompt += f"Locations (depot + {len(instance_data['locs']) - 1} customers): {instance_data['locs']}\n"
    prompt += f"Backhaul Demand: {instance_data['demand_backhaul']},\n"
    prompt += f"Linehaul Demand: {instance_data['demand_linehaul']}\n"
    prompt += f"Backhaul Class: {instance_data['backhaul_class']}, where '1' means classic backhaul and '2' means mixed backhaul.\n"
    prompt += f"Distance Limit: {instance_data['distance_limit']}\n"
    prompt += f"Time Windows: {instance_data['time_windows']}\n"
    prompt += f"Service Time: {instance_data['service_time']}\n"
    prompt += f"Vehicle Capacity: {instance_data['capacity_original']}, there is no capacity limitation if the capacity is equal to 0.\n"
    prompt += f"Open Route: {instance_data['open_route']}, where 'True' means the routes could be open while 'False' means the routes should be closed \n"
    prompt += f"Speed: {instance_data['speed']}\n"
    prompt += f"Please notice that your description should be complete enough to recognize the problem type, you can omit specific parameter values from the description.\n"
    prompt += f"Return the description directly without any additional explanation or information.\n"
    return prompt

# Function to generate natural language descriptions using LLM
def generate_natural_language_description(instance_data, scenario, llm):
    # Generate the prompt
    prompt = generate_prompt(instance_data, scenario)
    
    # Request the LLM to generate the description
    response = llm.get_text(content=prompt)
    return response

# Function to convert TensorDict to a simple dictionary
def convert_tensor_dict(tensor_dict):
    # Assuming tensor_dict is a dictionary-like object
    return {key: value.tolist() if hasattr(value, 'tolist') else value for key, value in tensor_dict.items()}

# Function to save the generated descriptions as a dataset
def save_to_dataset(descriptions, variant_names, td_data, pclass='VRP'):
    time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{pclass}_{time_stamp}.json"
    
    # Convert TensorDict to a JSON-serializable format
    serialized_td_data = [convert_tensor_dict(data) for data in td_data]
    print(len(serialized_td_data))
    print(len(descriptions))
    print(len(variant_names))
    
    data = [{"variant_name": variant_names[i], "description": descriptions[i], 'data': serialized_td_data[i]} for i in range(len(descriptions))]
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Generate instances directly from the generator and save them as JSON
def generate_and_save_vrp_instances(generator, scenario_list, plist, num_instances=100, pclass='VRP'):
    descriptions = []
    variant_names = []
    td_datas = []

    # Generate and process each problem type in plist
    for problem_type in plist:
        # Set the variant preset for the generator
        generator.variant_preset = problem_type.lower()  # Match problem type with preset

        # Generate the data for the given problem type
        td_data = generator._generate(batch_size=(num_instances,))

        # Append the problem type to the variant names
        variant_names.extend([problem_type] * num_instances)

        # Generate natural language descriptions for each instance
        for idx in range(len(td_data)):
            print(f"Processing variant {problem_type}...")

            # Randomly select a scenario
            scenario = np.random.choice(scenario_list)
            
            # Generate natural language description
            description = generate_natural_language_description(td_data[idx], scenario, llm)
            
            # Print or save the description
            print(f"Description: {description}")
            
            # Append the description to the list
            descriptions.append(description)
            td_datas.append(td_data[idx])

    # Save all descriptions, variant names, and TensorDict data to a JSON file
    save_to_dataset(descriptions, variant_names, td_datas, pclass)


# Define a real-world scenario list
scenario_list = ['field service management', 'waste collection', 'emergency response', 'mobile surveillance', 'goods delivery', 'advertising placement', 'personnel scheduling']

# Define the generator and problem types (plist)
generator = MTVRPGenerator(num_loc=50, variant_preset="all")
# plist = ['cvrp', 'cvrpl', 'cvrptw', 'ovrp', 'vrpb']  # Example problem types/
plist = ['CVRP', 'CVRPL', 'CVRPTW', 'OVRP', 'VRPB']  # Example problem types

# Generate and save the VRP instances as a JSON file
generate_and_save_vrp_instances(generator, scenario_list, plist, num_instances=2, pclass='VRP')
