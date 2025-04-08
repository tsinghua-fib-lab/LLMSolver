import json
from agent import Classifier, Checker
from LLM import LLM_api
import time

llm = LLM_api(model="deepseek-chat", key_idx=0)

think_flag = False

# Initialize the Classifier and Checker
classifier = Classifier(llm, think=False)
checker = Checker(llm, think=False)

# Load the dataset of VRP problem descriptions and their labels
def load_dataset(filename="benchmark/cvpr.txt"):
    with open(filename, 'r') as file:
        return json.load(file)

# Function to calculate the success rate
def calculate_success_rate(dataset):
    correct_predictions = 0
    total_predictions = len(dataset)
    print(f"Total predictions: {total_predictions}")

    # Iterate through the dataset and classify each VRP problem
    for data in dataset:
        variant_name = data["variant_name"]
        problem_desc = data["description"]
        print(variant_name)

        max_rounds = 3  # Limit the maximum number of iterations to avoid infinite loops

        for _ in range(max_rounds):
            # Classify the VRP problem
            classification = classifier.run(problem_desc)

            # Check the classification
            is_correct, reason = checker.run(problem_desc, classification)

            if is_correct:
                print(f"\n[Final Output] VRP Problem Type: {classification}")
                if classification == variant_name:
                    correct_predictions += 1
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
            if classification == variant_name:
                correct_predictions += 1

    # Calculate the success rate
    success_rate = correct_predictions / total_predictions
    return success_rate

# Load the dataset
dataset = load_dataset("VRP_20250312234814.json")

# Calculate the success rate
success_rate = calculate_success_rate(dataset)

# Print the success rate
print(f"\nLLM Classification Success Rate: {success_rate * 100:.2f}%")