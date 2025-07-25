import json
import re
import random
from utils.LLM import LLM_api

model_name = "deepseek-chat"
Sector = []
with open("benchmark/bpprompt.json", "r", encoding="utf-8") as f:
    data_prompt = json.load(f)
with open("benchmark/bppseedprompt.json", "r", encoding="utf-8") as f:
    data_seed_prompt = json.load(f)


def generate_random_list(n):
    return [[random.randint(1, 10) for _ in range(2)] for _ in range(n)]


# Initialize LLM instance
llm = LLM_api(
    model=model_name,
    max_tokens=8000,
    temperature=1.3
)

for key in ["2DOFBPPR", "2DONBPP", "2DONBPPR"]:
    problem_type = key
    if problem_type == "2DOFBPPR":
        Sector = []
        bin_status = "false",
        can_rotate = "true",
        dimension = "2D"
        filepath2 = f'benchmark/bpp/{key}.json'
    if problem_type == "2DONBPP":
        Sector = []
        bin_status = "true",
        can_rotate = "false",
        dimension = "2D"
        filepath2 = f'benchmark/bpp/{key}.json'
    if problem_type == "2DONBPPR":
        Sector = []
        bin_status = "true",
        can_rotate = "true",
        dimension = "2D"
        filepath2 = f'benchmark/bpp/{key}.json'
    else:
        raise NotImplementedError(problem_type)

    num_requests = 1  # Define the number of API calls you want to make
    for i in range(num_requests - 1, num_requests + 22):
        Sector_process = ','.join(Sector)
        user_input = data_prompt[key] + data_seed_prompt[
            key] + "The industry scenario title for each generated example must not duplicate any in the following list:" + Sector_process
        number = 5
        user_input = user_input.format(number=number)

        # Use LLM class to get response
        result = llm.get_text(content=user_input)

        if result:  # Check if response is not empty
            print(f"Total tokens used: {llm.get_token()}")
            resulttt = result.split('&&&')
            for result in resulttt:
                match = re.search(r"##(.*?)##", result)
                title = match.group(1) if match else None
                if title == None:
                    print("null_exist;")
                else:
                    Sector.append(title)
                    text_without_title = re.sub(r"##.*?##", "", result).strip()
                    responses = []
                    responses.append(text_without_title)  # Save each response
                    responses = [response.replace("\n", "") for response in responses]
                    responses = responses[0]
                    n_items = random.randint(20, 25)
                    items_size = generate_random_list(n_items)
                    desc_merge = responses.replace('<bin_size>', str([10, 10]))
                    desc_merge = desc_merge.replace('<items_size>', str(items_size))
                    add_data = {
                        "title": title,
                        "desc_split": responses,
                        "desc_merge": desc_merge,
                        "data_template": {
                            "bin_size": [10, 10],
                            "items_size": items_size,
                            "bin_status": bin_status,
                            "can_rotate": can_rotate,
                            "dimension": dimension,
                        },
                        "label": problem_type
                    }
                    try:
                        with open(filepath2, "r") as file:
                            data_list = json.load(file)  # Read data from file
                    except (FileNotFoundError, json.JSONDecodeError):  # When file doesn't exist or is empty
                        data_list = []
                    data_list.append(add_data)
                    with open(filepath2, "w") as file:
                        json.dump(data_list, file, indent=4)  #
            print(f"API call {i + 1} successful, results written to file")
        else:
            print(f"API call {i + 1} failed, empty response returned")
