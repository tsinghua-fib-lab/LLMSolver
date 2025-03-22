import os
import sys
import logging
import datetime
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recognition.agent import Classifier, Checker
from recognition.LLM import LLM_api

# Set up logging configuration
log_file = f"./log/{datetime.datetime.now().strftime('%Y-%m-%d')}.log"
logging.basicConfig(
    filename=log_file,  # Log file path
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    filemode='w'  # 'w' means overwrite the log file, 'a' would append to it
)

# Initialize the LLM API
# llm = LLM_api(model="Qwen/QwQ-32B", key_idx=0)
# llm = LLM_api(model="deepseek-chat", key_idx=0)
llm = LLM_api(model="deepseek-reasoner", key_idx=0)

think_flag = False

# Initialize the Classifier and Checker
classifier = Classifier(llm, think=False)
checker = Checker(llm, think=False)

# Load the dataset from text files
# def load_dataset(directory="/data1/shy/zgc/llm_solver/LLMSolver/benchmark"):
#     dataset = {}
#     for filename in os.listdir(directory):
#         if filename.endswith(".txt"):
#             problem_type = filename.replace(".txt", "")
#             with open(os.path.join(directory, filename), 'r') as file:
#                 descriptions = file.readlines()
#             dataset[problem_type] = descriptions
#     return dataset


def load_dataset(directory="/data1/shy/zgc/llm_solver/LLMSolver/benchmark", filename="VRP_20250320075347.json"):
    dataset = {}
    with open(os.path.join(directory, filename), 'r') as file:
        data = json.load(file)
    
    # Loop through each entry in the JSON file
    for entry in data:
        problem_type = entry.get("variant_name")  # Extract the problem type from "variant_name"
        description = entry.get("description")  # Extract the description
        
        # Ensure the problem type exists in the dataset dictionary
        if problem_type not in dataset:
            dataset[problem_type] = []
        
        # Append the description to the corresponding problem type
        dataset[problem_type].append(description)
    
    return dataset



# Function to calculate the success rate for each problem type and overall
def calculate_success_rate(dataset):
    error_excution = 0
    correct_predictions = 0
    total_predictions = 0

    problem_type_success = {}

    for problem_type, descriptions in dataset.items():
        correct_predictions_for_type = 0
        total_for_type = len(descriptions)

        logging.info(f"\nProcessing {problem_type}... Total descriptions: {total_for_type}")

        for problem_desc in descriptions:
            problem_desc = problem_desc.strip()  # Remove any trailing newlines/whitespace
            max_rounds = 3  # Limit the maximum number of iterations to avoid infinite loops

            for _ in range(max_rounds):
                # Classify the VRP problem
                try:
                    classification = classifier.run(problem_desc)
                    if classification == problem_type:
                        correct_predictions_for_type += 1
                    else:
                        logging.info(f"[Error] Classifier predicted: {classification}, Expected: {problem_type}, Description: {problem_desc}")
                    break
                    
                except Exception as e:
                    error_excution += 1
                    logging.error(f"[Error] Exception occurred: {e}")

            # Store the success rate for this problem type
        success_rate_for_type = correct_predictions_for_type / total_for_type
        problem_type_success[problem_type] = success_rate_for_type
        correct_predictions += correct_predictions_for_type
        total_predictions += total_for_type

    # Calculate the overall success rate
    overall_success_rate = correct_predictions / total_predictions
    return problem_type_success, overall_success_rate

# Load the dataset
dataset = load_dataset()

# Calculate the success rate
problem_type_success, overall_success_rate = calculate_success_rate(dataset)

# Print the success rates
logging.info("\nSuccess rates per problem type:")
for problem_type, success_rate in problem_type_success.items():
    logging.info(f"{problem_type}: {success_rate * 100:.2f}%")

logging.info(f"\nOverall Success Rate: {overall_success_rate * 100:.2f}%")