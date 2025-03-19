import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from rl4co.utils.trainer import RL4COTrainer
from routefinder.models import RouteFinderBase, RouteFinderPolicy
from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator
from routefinder.utils import rollout, greedy_policy, evaluate

from LLM import LLM_api  # Assuming LLM API is available
import json
import numpy as np
import datetime


# Define a function to generate the natural language prompt
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
    prompt += f"Please notice that your description should be complete enough to recover the whole problem instance, namely, all the parameters should be included in your description.\n"
    prompt += f"Return the description directly without any additional explanation or information.\n"
    return prompt

# Function to generate natural language descriptions using LLM
def generate_natural_language_description(instance_data, scenario, llm):
    # Generate the prompt
    prompt = generate_prompt(instance_data, scenario)
    
    # Request the LLM to generate the description
    response = llm.get_text(content=prompt)
    return response

# Function to save the generated descriptions as a dataset
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
    
    data = [{"variant_name": variant_names[i], "description": descriptions[i], 'data': serialized_td_data[i]} for i in range(len(descriptions))]
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)



# Assuming you have the following setup
generator = MTVRPGenerator(num_loc=10, variant_preset="all")
env = MTVRPEnv(generator, check_solution=False)

# Define the real-world scenario (choose one from the options
scenario_list = ['field service management', 'waste collection', 'emergency response', 'mobile surveillance', 'goods delivery', 'advertising placement', 'personnel scheduling']

# Generate data (mixed variants)
instance_num = 1
td_data = env.generator(instance_num)
variant_names = env.get_variant_names(td_data)
scenarios = np.random.choice(scenario_list, instance_num)

# LLM API initialization
llm = LLM_api(model="deepseek-reasoner", key_idx=0)

# List to store descriptions
plist = ['CVRP', 'CVRPL', 'CVRPW', 'OVRP', 'VRPB']


descriptions = []

# Loop through each instance and generate a description
for idx in range(len(td_data)):
    
    print(f"Processing variant {variant_names[idx]}...")
    
    description = generate_natural_language_description(td_data[idx], scenarios[idx], llm)
    
    # Print or save the description
    print(f"Description: {description}")
    
    # Append the description to the list
    descriptions.append(description)

# Save the descriptions to a file
save_to_dataset(descriptions, variant_names, td_data)