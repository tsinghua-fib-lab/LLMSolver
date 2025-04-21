from tensordict.tensordict import TensorDict
import torch
import time
from recognition.LLM import LLM_api
from collections import defaultdict
from recognition.variants import VRP_VARIANTS, SP_VARIANTS


class Agent:
    """Base agent class: Encapsulates the common LLM calling logic."""
    def __init__(self, llm_api):
        """
        :param llm_api: An instance of LLM_api
        """
        self.llm_api = llm_api

    def get_result(self, text):
        """
        Parses the LLM output to extract the JSON result.
        :param text: The LLM output
        :return: A dictionary containing the parsed result
        """
        try:
            # Remove markdown code block syntax if present
            text = text.replace('```json', '').replace('```', '').strip()
            json_format = eval(text)
            result = json_format.get('result', None)
            reason = json_format.get('reason', None)
            return (result, reason)
        except Exception as e:
            return (f"<Error> parsing JSON: {e}", text)

    def run(self, *args, **kwargs):
        """Subclasses must implement the run method."""
        raise NotImplementedError("Please implement the run method in subclasses.")


class Classifier(Agent):
    """
    A VRP problem classifier agent: 
    Given a VRP problem description in natural language, 
    """
    def add_prompt(self, problem_desc: str, reason=None):
        # prompt = f"You are a VRP problem judger agent. The user has provided the following VRP problem description:\n"
        # prompt += f"{problem_desc}\n"
        # prompt += "You need to choose one category for the description from the following list:\n"
        # prompt += f"{VRP_VARIANTS}\n"
        # prompt += "Where:\n"
        # prompt += "- 'VRP' is the basic Vehicle Routing Problem\n"
        # prompt += "- 'O' represents Open Route (routes can be open)\n"
        # prompt += "- 'B' represents backhaul demand, and the backhaul demand can be only satisfied after all the linehual demand are satisfied. \n"
        # prompt += "- 'MB' represents mixed backhaul, which means backhaul demand and linehaul demand can be sttisfied simultaneously.\n"
        # prompt += "- 'L' represents Distance Limits\n"
        # prompt += "- 'TW' represents Time Window constraints\n"
        # prompt += "If the description contains any attribute in 'O', 'B', 'MB', 'L' or 'TW', the most likely corresponding classification is given, otherwise is 'C'.\n"
        # prompt += "Your output should be a string in json format as follows:\n"
        # prompt += "{\n"
        # prompt += "    'reason': <string>,\n"
        # prompt += "    'result': <string>\n"
        # prompt += "}\n"
        # prompt += "Where 'reason' is a string explaining why you chose this category, and 'result' must be one of the elements in the list.\n"
        
        
        prompt = f"You are a Scheduling Problem judger agent. The user has provided the following scheduling problem description:\n"
        prompt += f"{problem_desc}\n"
        prompt += "You need to choose one category for the description from the following list:\n"
        prompt += f"{SP_VARIANTS}\n"
        prompt += "Where:\n"
        prompt += "- 'Job Shop Scheduling Problem (JSSP) - fixed machine per operation, predefined operation order'\n"
        prompt += "- 'Flexible Job Shop Scheduling Problem (FJSSP) - flexible machine choice per operation, predefined operation order'\n"
        prompt += "- 'Flow-Shop Scheduling Problem (FSSP) - all jobs follow the same machine sequence'\n"
        prompt += "- 'Hybrid Flow-shop Scheduling Problem (HFSSP) - multiple stages with parallel machines in at least one stage'\n"
        prompt += "- 'Open Shop Scheduling Problem (OSSP) - no predefined operation order'\n"
        prompt += "- 'Assembly Scheduling Problem (ASP) - operations have precedence constraints (DAG)'\n"
        prompt += "Carefully read the problem description and identify the key features that match one of the scheduling problem variants listed above.\n"
        prompt += "Your output should be a string in JSON format as follows:\n"
        prompt += "{\n"
        prompt += "    'reason': <string>,\n"
        prompt += "    'result': <string>\n"
        prompt += "}\n"
        prompt += f"Where 'reason' is a string explaining why you chose this category, and 'result' must be one of the {SP_VARIANTS}.\n"
        
        return prompt
    
    def run(self, problem_desc: list, reason=None, num_threads=1) -> list:
        prompt = [self.add_prompt(problem_desc[i], reason) for i in range(min(num_threads, len(problem_desc)))]
        text = self.llm_api.get_multi_text(prompt, max_workers=num_threads)
        
        tuples = []
        for t in text:
            result = self.get_result(t)
            if isinstance(result, tuple):
                choice, reason = result
                tuples.append((choice.strip(), reason.strip()))
            else:
                tuples.append(("<Error>", str(result)))
        return tuples


class Checker(Agent):
    """
    A VRP problem checker agent:
    Given the user's VRP problem description and the classifier output, 
    it verifies whether the classification is correct.
    If correct, it should return True and an empty string.
    If incorrect, it should return False and a reason.
    """
    def run(self, problem_desc: str, classification: str) -> str:
        """
        :param problem_desc: The user's natural language VRP problem description
        :param classification: The classification result from the Classifier
        :return:
            bool: True if correct, False otherwise
            str: The reason if it is incorrect, or empty if correct
        """
        
        prompt = f"You are a VRP problem checker agent.\n"
        prompt += f"User's description: {problem_desc}\n"
        prompt += f"Classifier's classification: {classification}\n"
        prompt += f"Candidate categories: [CVRP, OVRP, VRPB, VRPL, VRPTW]\n"
        prompt += "If you believe this classification is correct, 1 if correct, 0 if incorrect.\n"
        prompt += 'Below are the letters that may appear in the VRP category and their corresponding meanings:\n'
        prompt += " 'C' represents Capacity, which means the vehicle has a capacity limit."
        prompt += " 'O' represents Open Route, which means routes can be open."
        prompt += " 'B' represents backhaul demand."
        prompt += " 'L' represents Distance Limits."
        prompt += " 'TW' represents Time Window, which means the vehicle has a time window to visit each customer.\n"
        prompt += "If the description contains any attribute in 'O', 'B', 'L' or 'TW', the most likely corresponding classification is given, otherwise is 'C'.\n"
        prompt += "Your output should be in json format as follows:\n"
        prompt += "{\n"
        prompt += "    'reason': <string>,\n"
        prompt += "    'result': <string>\n"
        prompt += "}\n"
        prompt += "Where 'reason' is a string explaining why the classification is correct or incorrect, and 'result' is a string that should be either '1' or '0'.\n"
        
        text = self.llm_api.get_text(content=prompt).strip()
        print(f"[Checker] LLM response: {text}")
        result = self.get_result(text)

        if isinstance(result, tuple):
            correct, reason = result
            if correct == "1":
                return (True, reason.strip())
            elif correct == "0":
                return (False, reason.strip())
        else:
            return f"<Error>: {result}"

class Extractor(Agent):
    """
    A VRP problem extractor agent:
    Given a natural language VRP problem description and a prompt to extract specific data,
    it processes the description and extracts relevant information in a structured format.
    """
    def run(self, problem_desc: str) -> TensorDict:
        """
        :param problem_desc: The user's natural language VRP problem description
        :param prompt: The prompt to guide the extraction process
        :return: A dictionary containing the extracted data
        """
        # Create a structured prompt to instruct the LLM on what data to extract
        prompt = f"You are a VRP data extractor agent.\n"
        prompt += f"The user has provided the following VRP problem description:\n{problem_desc}\n"
        prompt += f"Based on this description, extract the following data in a json format as follows :\n"
        prompt += "{\n"
        prompt += "    'locs': <list of location coordinates, length should be equal to the number of locations>,\n"
        prompt += "    'demand_backhaul': <list of demand values for backhaul, length should equal to the number of locations - 1>,\n"
        prompt += "    'demand_linehaul': <list of demand values for linehaul, length should equal to the number of locations - 1>,\n"
        prompt += "    'backhaul_class': <list with a single value, [1] for classic backhaul and [2] for mixed backhaul. Default value is [1]>,\n"
        prompt += "    'distance_limit': <list with a single value, or a default 'inf if not provided>,\n"
        prompt += "    'time_windows': <list of time windows, length should match number of locations, default value [0, 'inf'] if not specified>,\n"
        prompt += "    'service_time': <list of service times, length should match number of locations, default value 0 if not specified>,\n"
        prompt += "    'vehicle_capacity': <list with a single value for vehicle capacity, or default 1 if not specified>,\n"
        prompt += "    'capacity_original': <list with a single value for the original capacity, or default 30 if not specified>,\n"
        prompt += "    'open_route': <list with a single value, 1 if open routes are allowed, otherwise 0. Default value is False if not specified>,\n"
        prompt += "    'speed': <list with a single value for vehicle speed, or default 1 if not specified>\n"
        prompt += "}\n"
        prompt += "Your output should be in json format as follows:\n"
        prompt += "{\n"
        prompt += "    'reason': <string>,\n"
        prompt += "    'result': <json>\n"
        prompt += "}\n"
        prompt += "Where 'reason' is a string explaining the extracted data, and 'result' should be the extracted data in JSON format.\n"

        # Get the extracted text from LLM
        text = self.llm_api.get_text(content=prompt)
        print(f"[Extractor] LLM response: {text}")
        result = self.get_result(text)

        if isinstance(result, tuple):
            extracted_dict, reason = result
            try:
                print(extracted_dict)
                td_data =  TensorDict(
                {
                    # normalize and add 1 dim at the start 
                    'locs': (torch.tensor(extracted_dict['locs'])).float().unsqueeze(0),
                    'demand_backhaul': torch.tensor(extracted_dict['demand_backhaul']).float().unsqueeze(0),
                    'demand_linehaul': torch.tensor(extracted_dict['demand_linehaul']).float().unsqueeze(0),
                    'backhaul_class': torch.tensor(extracted_dict['backhaul_class']).float().unsqueeze(0),
                    'distance_limit': torch.tensor(extracted_dict['distance_limit']).float().unsqueeze(0),
                    'time_windows': torch.tensor(extracted_dict['time_windows']).float().unsqueeze(0),
                    'service_time': torch.tensor(extracted_dict['service_time']).float().unsqueeze(0),
                    'vehicle_capacity': torch.tensor(extracted_dict['vehicle_capacity']).float().unsqueeze(0),
                    'capacity_original': torch.tensor(extracted_dict['capacity_original']).float().unsqueeze(0),
                    'open_route': torch.tensor(extracted_dict['open_route']).bool().unsqueeze(0),
                    'speed': torch.tensor(extracted_dict['speed']).float().unsqueeze(0)
                },
                batch_size = 1
            )
                return (td_data, reason.strip())
            except Exception as e:
                return f"<Error> parsing JSON for td: {e}"
        else:
            return f"<Error>: {result}"

def main():
    """
    A sample workflow that:
    1) Initializes the LLM API
    2) Creates a Classifier, Checker, and Extractor
    3) Obtains the VRP problem description from the user
    4) Classifies the problem, checks the classification, and extracts relevant
    data based on the problem description.
    """
    # 1) Initialize LLM_api (adjust parameters as needed)
    llm = LLM_api(
        model="Qwen/Qwen2.5-7B-Instruct",
        key_idx=0,
    )

    # 2) Create the Classifier (judger) and Checker
    classifier = Classifier(llm)
    checker = Checker(llm)
    extractor = Extractor(llm)

    # 3) Obtain the VRP problem description from the user
    problem_desc = input("Please enter your VRP problem description: ")

    max_rounds = 5  # Limit the maximum number of iterations to avoid infinite loops

    for _ in range(max_rounds):
        # (a) Classify the VRP problem
        classification = classifier.run(problem_desc)

        # (b) Check the classification
        is_correct, reason = checker.run(problem_desc, classification)

        if is_correct:
            print(f"\n[Final Output] VRP Problem Type: {classification}")
            break
        else:
            print(f"\n[Checker] The classification is incorrect. Reason: {reason}")
            # Append the reason to the problem description, so the classifier can reconsider
            problem_desc += f"\n(Checker feedback: {reason})"
            time.sleep(1)  # Delay to avoid rapid repetitive calls
    else:
        # If we exit the loop without agreement, you can handle it accordingly
        print("[Warning] Reached the maximum number of iterations without agreement.")

    extracted_data = extractor.run(problem_desc)
    
    print("\n[Extracted Data] VRP Instance Data:")
    for key, value in extracted_data.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
