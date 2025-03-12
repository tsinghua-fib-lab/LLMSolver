import time
from LLM import LLM_api

class Agent:
    """Base agent class: Encapsulates the common LLM calling logic."""
    def __init__(self, llm_api):
        """
        :param llm_api: An instance of LLM_api
        """
        self.llm_api = llm_api

    def run(self, *args, **kwargs):
        """Subclasses must implement the run method."""
        raise NotImplementedError("Please implement the run method in subclasses.")


class Classifier(Agent):
    """
    A VRP problem classifier agent: 
    Given a VRP problem description in natural language, 
    it returns one category from 
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRP Variant  || Capacity (C) | Open Route (O) | Backhaul (B) | Duration Limit (L) | Time Window (TW) |
    +==============++==============+================+==============+====================+==================+
    | CVRP         || ✔            |                |              |                    |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRP         || ✔            | ✔              |              |                    |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPB         || ✔            |                | ✔            |                    |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPL         || ✔            |                |              | ✔                  |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPTW        || ✔            |                |              |                    | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPTW       || ✔            | ✔              |              |                    | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPB        || ✔            | ✔              | ✔            |                    |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPL        || ✔            | ✔              |              | ✔                  |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPBL        || ✔            |                | ✔            | ✔                  |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPBTW       || ✔            |                | ✔            |                    | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPLTW       || ✔            |                |              | ✔                  | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPBL       || ✔            | ✔              | ✔            | ✔                  |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPBTW      || ✔            | ✔              | ✔            |                    | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPLTW      || ✔            | ✔              |              | ✔                  | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPBLTW      || ✔            |                | ✔            | ✔                  | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPBLTW     || ✔            | ✔              | ✔            | ✔                  | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    """
    
    def run(self, problem_desc: str) -> str:
        """
        :param problem_desc: The user's natural language VRP problem description
        :return: A string, one of [CVRP, OVRP, VRPB, VRPL, VRPTW, OVRPTW, OVRPB, OVRPL, VRPBL, VRPBTW, VRPLTW, OVRPBL, OVRPBTW, OVRPLTW, VRPBLTW, OVRPBLTW]
        """
        prompt = f"You are a VRP problem judger agent. The user has provided the following VRP problem description:\n"
        prompt += f"{problem_desc}\n"
        prompt += "You only need to return one category from [CVRP, OVRP, VRPB, VRPL, VRPTW, OVRPTW, OVRPB, OVRPL, VRPBL, VRPBTW, VRPLTW, OVRPBL, OVRPBTW, OVRPLTW, VRPBLTW, OVRPBLTW].\n"
        prompt += "Where 'O' represents Open Route, 'B' represents Backhaul, 'L' represents Duration Limit, 'TW' represents Time Window.\n"
        prompt += "Return nothing else."

        result = self.llm_api.get_text(content=prompt)
        return result.strip()


class Checker(Agent):
    """
    A VRP problem checker agent:
    Given the user's VRP problem description and the classifier output, 
    it verifies whether the classification is correct.
    If correct, it should return True and an empty string.
    If incorrect, it should return False and a reason.
    """
    def run(self, problem_desc: str, classification: str) -> (bool, str):
        """
        :param problem_desc: The user's natural language VRP problem description
        :param classification: The classification result from the Classifier
        :return:
            bool: True if correct, False otherwise
            str: The reason if it is incorrect, or empty if correct
        """
        prompt = f"""You are a VRP problem checker agent.
User's description: {problem_desc}
Classifier's output: {classification}

If you believe this classification is correct, return only: "CORRECT".
If you believe this classification is incorrect, return only: "INCORRECT: <reason>".
Do not add any additional text beyond these formats."""
        
        result = self.llm_api.get_text(content=prompt).strip()

        if result.startswith("CORRECT"):
            return True, ""
        elif result.startswith("INCORRECT"):
            # Parse the reason after "INCORRECT:"
            if ":" in result:
                reason = result.split(":", 1)[1].strip()
                return False, reason
            else:
                return False, "No reason provided"
        else:
            # If the response does not follow the specified format, treat it as incorrect
            return False, f"Invalid output format: {result}"


def main():
    """
    A sample workflow that:
      1) Takes the user's VRP description as input.
      2) Uses the Classifier to determine the problem type.
      3) Uses the Checker to verify if the classification is correct.
      4) If incorrect, appends the checker feedback to the original description 
         and retries until both agents agree or the maximum number of iterations is reached.
    """
    # 1) Initialize LLM_api (adjust parameters as needed)
    llm = LLM_api(
        model="Qwen/Qwen2.5-7B-Instruct",
        key_idx=0,
    )

    # 2) Create the Classifier (judger) and Checker
    classifier = Classifier(llm)
    checker = Checker(llm)

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


if __name__ == "__main__":
    main()
