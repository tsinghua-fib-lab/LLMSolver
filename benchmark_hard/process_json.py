import os
import json
import glob

def process_json_files():
    # Problem types directories
    problem_types = ['graph', 'cvrp', 'msp', 'kp', 'bpp']
    all_problems = []
    problem_id = 1

    # Process each problem type
    for problem_type in problem_types:
        dataset_dir = os.path.join(os.path.dirname(__file__), problem_type, 'dataset')
        if not os.path.exists(dataset_dir):
            continue

        # Find all JSON files in the dataset directory
        json_files = glob.glob(os.path.join(dataset_dir, '**', '*.json'), recursive=True)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Handle both list and dict formats
                    if isinstance(data, dict):
                        data = [data]
                    
                    # Process each problem in the file
                    for problem in data:
                        if isinstance(problem, dict) and 'desc_merge' in problem:
                            new_problem = {
                                'en_question': problem['desc_merge'],
                                'en_answer': '0',  # Set answer to 0
                                'difficulty': 'Hard',  # Set difficulty to Hard
                                'id': problem_id,
                                'problem_type': problem['label']
                            }
                            all_problems.append(new_problem)
                            problem_id += 1
                            
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")

    # Save all problems to AutoCOBench.json
    output_file = os.path.join(os.path.dirname(__file__), 'AutoCOBench.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_problems, f, ensure_ascii=False, indent=2)

    print(f"Successfully processed {len(all_problems)} problems into AutoCOBench.json")

if __name__ == '__main__':
    process_json_files() 