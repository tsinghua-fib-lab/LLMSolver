import os
import json
import glob

def process_individual_json():
    """
    Process each JSON file individually and save them separately in the AutoCOBench directory.
    Each input JSON file will generate a corresponding processed JSON file.
    """
    # Problem types directories
    problem_types = ['graph', 'cvrp', 'msp', 'kp', 'bpp']
    problem_id = 1
    
    # Create output directory if it doesn't exist
    output_dir = '/data1/shy/zgc/llm_solver/ORLM/dataset/AutoCOBench'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each problem type
    for problem_type in problem_types:
        dataset_dir = os.path.join(os.path.dirname(__file__), problem_type, 'dataset')
        if not os.path.exists(dataset_dir):
            print(f"Skipping {problem_type}: directory not found")
            continue
            
        # Find all JSON files in the dataset directory
        json_files = glob.glob(os.path.join(dataset_dir, '**', '*.json'), recursive=True)
        print(f"Processing {len(json_files)} files for {problem_type}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Handle both list and dict formats
                    if isinstance(data, dict):
                        data = [data]
                    
                    # Process problems in this file
                    processed_problems = []
                    for problem in data:
                        if isinstance(problem, dict) and 'desc_split' in problem:
                            if 'user_template' in problem:
                                new_problem = {
                                    'en_question': problem['desc_split'] + '\n And the following is the user template: \n' + json.dumps(problem['user_template']),    
                                    'en_answer': '0',  # Set answer to 0
                                    'difficulty': 'Hard',  # Set difficulty to Hard
                                    'id': problem_id,
                                    'problem_type': problem['label']
                                }
                            else:
                                new_problem = {
                                    'en_question': problem['desc_split'] + '\n And the following is the description of the problem: \n' + json.dumps(problem['data_template']),
                                    'en_answer': '0',  # Set answer to 0
                                    'difficulty': 'Hard',  # Set difficulty to Hard
                                    'id': problem_id,
                                    'problem_type': problem['label']
                                }
                            processed_problems.append(new_problem)
                            problem_id += 1
                    
                    if processed_problems:
                        # Generate output filename based on original file
                        base_name = os.path.basename(json_file)
                        output_name = f"{problem_type}_{base_name}"
                        output_path = os.path.join(output_dir, output_name)
                        
                        # Save processed problems to new file
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(processed_problems, f, ensure_ascii=False, indent=2)
                        
                        print(f"Processed {len(processed_problems)} problems from {base_name} -> {output_name}")
                            
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")

if __name__ == '__main__':
    process_individual_json() 