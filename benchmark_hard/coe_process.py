import os
import json
import glob

def create_code_example(data_template, problem_label, index):
    """
    Create a code_example.py file based on the data_template
    
    Args:
        data_template (dict): The data template containing input parameters
        problem_label (str): The label of the problem
        index (int): The index of the problem
    
    Returns:
        str: The content of the code_example.py file
    """
    # Get the keys from data_template
    param_keys = list(data_template.keys())
    
    # Create function parameters string
    params_str = ", ".join(param_keys)
    
    # Create function docstring with args
    args_str = "\n    ".join([f"Args:"] + [f"    {key}: description of {key}" for key in param_keys])
    
    # Create the function template
    function_name = f"prob_{problem_label}_{index}"
    code_content = f"""def {function_name}({params_str}):
    \"\"\"
    {args_str}
    
    Returns:
        obj: an integer representing the optimal objective value
    \"\"\"
    obj = 0
    # To be implemented
    return obj
"""
    return code_content

def create_sample_json(data_template):
    """
    Create a sample.json file based on the data_template
    
    Args:
        data_template (dict): The data template containing input parameters
    
    Returns:
        str: The content of the sample.json file
    """
    sample = {
        "input": data_template,
        "output": [0]
    }
    return json.dumps([sample], indent=4)

def process_json_files():
    """
    Process JSON files from benchmark_hard and create folder structure with code_example.py,
    description.txt, and sample.json files for each problem
    """
    # Problem types directories
    problem_types = ['graph', 'cvrp', 'msp', 'kp', 'bpp']
    
    # Main output directory
    output_base_dir = os.path.join(os.path.dirname(__file__), 'processed_problems')
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each problem type
    for problem_type in problem_types:
        print(f"Processing problem type: {problem_type}")
        
        dataset_dir = os.path.join(os.path.dirname(__file__), problem_type, 'dataset')
        if not os.path.exists(dataset_dir):
            print(f"  - Dataset directory not found for {problem_type}, skipping")
            continue
            
        # Find all JSON files in the dataset directory
        json_files = glob.glob(os.path.join(dataset_dir, '**', '*.json'), recursive=True)
        
        for json_file in json_files:
            print(f"  - Processing file: {os.path.basename(json_file)}")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Handle both list and dict formats
                    if isinstance(data, dict):
                        data = [data]
                    
                    # Process each problem in the file
                    for problem in data:
                        if (isinstance(problem, dict) and 
                            'desc_split' in problem and 
                            'data_template' in problem and 
                            'label' in problem and 
                            'index' in problem):
                            
                            # Create folder for this problem
                            problem_dir = os.path.join(
                                output_base_dir, 
                                f"prob_{problem['label']}_{problem['index']}"
                            )
                            os.makedirs(problem_dir, exist_ok=True)
                            
                            # Create code_example.py
                            code_content = create_code_example(
                                problem['data_template'], 
                                problem['label'], 
                                problem['index']
                            )
                            with open(os.path.join(problem_dir, 'code_example.py'), 'w', encoding='utf-8') as f:
                                f.write(code_content)
                            
                            # Create description.txt
                            with open(os.path.join(problem_dir, 'description.txt'), 'w', encoding='utf-8') as f:
                                f.write(problem['desc_split'])
                            
                            # Create sample.json
                            sample_content = create_sample_json(problem['data_template'])
                            with open(os.path.join(problem_dir, 'sample.json'), 'w', encoding='utf-8') as f:
                                f.write(sample_content)
                            
                            print(f"    - Created problem dir: {os.path.basename(problem_dir)}")
                            
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")

if __name__ == '__main__':
    process_json_files() 