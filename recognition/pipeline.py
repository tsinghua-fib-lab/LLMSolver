
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
from tensordict.tensordict import TensorDict
from agent import Classifier, Checker, Extractor
from LLM import LLM_api
from envs.mtvrp import MTVRPEnv, MTVRPGenerator
from solver.solver_pool import SolverPool

# Initialize the LLM API
llm = LLM_api(model="deepseek-reasoner", key_idx=0)

think_flag = False

# Initialize the Classifier and Checker
classifier = Classifier(llm, think=think_flag)
checker = Checker(llm, think=think_flag)
extractor = Extractor(llm, think=think_flag)

# # Load the dataset of VRP problem descriptions and their labels
# def load_dataset(filename="VRP_20250312162103.json"):
#     with open(filename, 'r') as file:
#         return json.load(file)

# Function to calculate the success rate
def precog(text):
    correct_predictions = 0

    # Iterate through the dataset and classify each VRP problem
    problem_desc = text

    max_rounds = 3  # Limit the maximum number of iterations to avoid infinite loops

    for _ in range(max_rounds):
        # Classify the VRP problem
        classification = classifier.run(problem_desc)

        # Check the classification
        is_correct, reason = checker.run(problem_desc, classification)

        if is_correct:
            print(f"\n[Final Output] VRP Problem Type: {classification}")
            break
        else:
            print(f"\n[Checker] The classification {classification} is incorrect. Reason: {reason}")
            # Append the reason to the problem description, so the classifier can reconsider
            problem_desc += f"\n(Your original classification {classification} might be wrong with feedback: {reason} from the checker, please reconsider.)"
            time.sleep(1)  # Delay to avoid rapid repetitive calls
    else:
        # If we exit the loop without agreement
        print("[Warning] Reached the maximum number of iterations without agreement.")
        print("Last classification:", classification)

    # Extract the relevant information
    td_data = extractor.run(problem_desc)
    print(f"\n[Extractor] Extracted Information: {td_data}")

    return td_data

# Load the dataset
# dataset = load_dataset("VRP_20250312234814.json")
text = "In a city's food delivery service, there is a central kitchen and 15 customers scattered across different neighborhoods. The coordinates of the central kitchen and the customers are represented as a matrix: [[4, 12], [18, 52], [22, 38], [36, 30], [45, 60], [55, 55], [50, 35], [40, 40], [30, 50], [25, 65], [20, 20], [10, 25], [15, 45], [35, 25], [60, 45], [65, 35]]. Each customer has a specific demand for food packages, given by the matrix: [0.2, 0.3, 0.1, 0.4, 0.2, 0.3, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.1, 0.2]. Delivery vehicles have a maximum load capacity of 1, and each vehicle must start and end its route at the central kitchen. The objective is to design vehicle routes that ensure all customers are visited exactly once, minimizing the total distance traveled while respecting the vehicle capacity constraints."

# Calculate the success rate


model_type = 'rf-transformer'
policy_dir = f'/data1/shy/zgc/llm_solver/LLMSolver/solver/model_checkpoints/100'
lkh_path = f'/data1/shy/zgc/llm_solver/LLMSolver/solver/lkh_solver/LKH-3.0.13/LKH'
lkh_num_runs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

solver_pool = SolverPool(lkh_path=lkh_path, lkh_num_runs=lkh_num_runs,
                            policy_dir=policy_dir, device=device)
# problem_type = "cvrp"
# for problem_type in ['cvrp', 'ovrp', 'vrpb', 'vrpl', 'vrptw']:
for problem_type in ['cvrp']:
    print(problem_type)
    generator = MTVRPGenerator(num_loc=30, variant_preset=problem_type)
    env = MTVRPEnv(generator, check_solution=False)
    # print(td_data)
    td_data = precog(text)
    td_test = env.reset(td_data)
    for solver_name in ['rf-transformer', 'lkh', 'pyvrp', 'ortools']:
        actions = solver_pool.solve(td_test.clone(), solver_name=solver_name)
        rewards = env.get_reward(td_test.clone().cpu(), actions.clone().cpu())
        print(solver_name)
        print(actions)
        print(rewards)