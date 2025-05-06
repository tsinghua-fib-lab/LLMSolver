import os
import json
import time
import argparse
from tqdm import tqdm
from collections import defaultdict
from recognition.LLM import LLM_api
from recognition.agent import Classifier, Checker, Extractor

def load_json_file(file_path):
    """Load and parse a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def test_model_on_vrp_variant(json_path, model_name, log_file, num_samples=None, use_checker=True, max_rounds=3):
    """
    Test a specific language model on a VRP variant.
    
    Args:
        json_path: Path to the JSON file containing VRP problems
        model_name: Name of the language model to test
        log_file: File to log errors
        num_samples: Number of samples to test (None for all)
        use_checker: Whether to use the Checker to verify classifications
        max_rounds: Maximum number of classification rounds with feedback
    
    Returns:
        Tuple of (correct_count, total_count, problem_type, checker_corrections)
    """
    print(f"Testing model {model_name} on {json_path}")
    
    # Initialize LLM API
    llm = LLM_api(model=model_name, max_tokens=16384)
    
    # Initialize classifier and checker
    classifier = Classifier(llm_api=llm)
    checker = Checker(llm_api=llm) if use_checker else None
    
    # Load problems
    problems = load_json_file(json_path)
    
    # Determine the true label from the file
    variant_name = os.path.basename(json_path).split('.')[0]
    if variant_name.lower() in ['cvrp', 'ovrp', 'vrpb', 'vrpl', 'vrptw', 'vrpmb']:
        true_label = variant_name.upper()
    else:
        # Try to find the label in the first problem
        try:
            true_label = problems[0].get('label', variant_name.upper())
        except:
            true_label = variant_name.upper()
    
    # Limit samples if specified
    if num_samples is not None:
        problems = problems[:num_samples]
    
    correct_count = 0
    total_count = len(problems)
    checker_corrections = 0  # Count problems corrected by the checker
    
    # Process each problem
    for i, problem in enumerate(tqdm(problems, desc=f"Testing {variant_name}")):
        desc = problem.get('desc_split', '')
        
        # Skip empty descriptions
        if not desc:
            total_count -= 1
            continue
        
        problem_desc = desc
        final_classification = None
        
        # First round of classification
        result = classifier.run(problem_desc)
        
        if not isinstance(result, tuple):
            # Log error
            with open(log_file, 'a') as f:
                f.write(f"Model: {model_name}, Problem Type: {true_label}, Index: {i}, Error: {result}\n")
                f.write("-" * 80 + "\n")
            total_count -= 1
            continue
        
        choice, reason = result
        initial_choice = choice.strip()
        
        # If using checker, verify the classification
        if use_checker:
            rounds = 0
            current_choice = initial_choice
            current_reason = reason
            
            while rounds < max_rounds:
                # Check if classification is already correct
                if current_choice == true_label:
                    final_classification = current_choice
                    if rounds > 0:
                        checker_corrections += 1
                    break
                
                # Check the classification
                check_result = checker.run(problem_desc, current_choice)
                
                if not isinstance(check_result, tuple):
                    # Checker failed, use the current classification
                    final_classification = current_choice
                    break
                
                is_correct, check_reason = check_result
                
                # If checker says it's correct, we'll accept it (even if it differs from true_label)
                if is_correct:
                    final_classification = current_choice
                    break
                
                # Append the feedback and try again
                problem_desc += f"\n(Your original classification {current_choice} might be wrong with feedback: {check_reason} from the checker, please reconsider.)"
                
                # Get a new classification with the feedback
                new_result = classifier.run(problem_desc)
                
                if not isinstance(new_result, tuple):
                    # Classification failed, use the current classification
                    final_classification = current_choice
                    break
                
                new_choice, new_reason = new_result
                current_choice = new_choice.strip()
                current_reason = new_reason
                
                rounds += 1
            
            # If we exit the while loop without setting final_classification
            if final_classification is None:
                final_classification = current_choice
        else:
            # Not using checker, just use the initial classification
            final_classification = initial_choice
        
        # Check if final classification is correct
        if final_classification == true_label:
            correct_count += 1
        else:
            # Log incorrect classification
            with open(log_file, 'a') as f:
                f.write(f"Model: {model_name}, Problem Type: {true_label}, Index: {i}\n")
                f.write(f"Description: {desc}\n")
                f.write(f"Initial Prediction: {initial_choice}\n")
                f.write(f"Final Prediction: {final_classification}\n")
                f.write(f"Initial Reason: {reason}\n")
                if use_checker and initial_choice != final_classification:
                    f.write(f"Checker corrected but still wrong\n")
                f.write("-" * 80 + "\n")
    
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Model: {model_name}, Problem Type: {true_label}")
    print(f"Accuracy: {correct_count}/{total_count} = {accuracy:.4f}")
    if use_checker:
        print(f"Checker corrections: {checker_corrections}")
    
    return correct_count, total_count, true_label, checker_corrections

def run_all_tests(data_path, models, num_samples=None, use_checker=True):
    """
    Run tests for all VRP variants on all specified models.
    
    Args:
        data_path: Path to directory containing VRP JSON files
        models: List of model names to test
        num_samples: Number of samples per variant to test
        use_checker: Whether to use the Checker to verify classifications
    """
    results = {}
    all_jsons = [f for f in os.listdir(data_path) if f.endswith('.json') and not f.endswith('_meta.json')]
    
    for model_name in models:
        model_results = defaultdict(list)
        overall_correct = 0
        overall_total = 0
        overall_corrections = 0
        
        # Log file for recording misclassifications
        log_file = f"error_log_{model_name.replace('/', '_')}.txt"
        with open(log_file, 'w') as f:
            f.write(f"Error log for model: {model_name}\n")
            f.write("=" * 80 + "\n")
        
        for json_file in all_jsons:
            json_path = os.path.join(data_path, json_file)
            correct, total, problem_type, corrections = test_model_on_vrp_variant(
                json_path, model_name, log_file, num_samples, use_checker
            )
            
            model_results[problem_type].append((correct, total, corrections))
            overall_correct += correct
            overall_total += total
            overall_corrections += corrections
        
        # Calculate aggregate results
        results[model_name] = {
            'by_type': {},
            'overall': (overall_correct, overall_total, 
                         overall_correct / overall_total if overall_total > 0 else 0,
                         overall_corrections)
        }
        
        for problem_type, counts in model_results.items():
            type_correct = sum(c[0] for c in counts)
            type_total = sum(c[1] for c in counts)
            type_corrections = sum(c[2] for c in counts)
            results[model_name]['by_type'][problem_type] = (
                type_correct, 
                type_total, 
                type_correct / type_total if type_total > 0 else 0,
                type_corrections
            )
    
    # Print and save overall results
    with open("classification_results_with_checker.json", 'w') as f:
        json_results = {}
        
        for model_name, model_data in results.items():
            json_results[model_name] = {
                'overall': {
                    'correct': model_data['overall'][0],
                    'total': model_data['overall'][1],
                    'accuracy': model_data['overall'][2],
                    'checker_corrections': model_data['overall'][3]
                },
                'by_type': {}
            }
            
            print(f"\n=== Model: {model_name} ===")
            print(f"Overall Accuracy: {model_data['overall'][0]}/{model_data['overall'][1]} = {model_data['overall'][2]:.4f}")
            if use_checker:
                print(f"Overall Checker Corrections: {model_data['overall'][3]}")
            
            for problem_type, (correct, total, accuracy, corrections) in model_data['by_type'].items():
                json_results[model_name]['by_type'][problem_type] = {
                    'correct': correct,
                    'total': total,
                    'accuracy': accuracy,
                    'checker_corrections': corrections
                }
                print(f"{problem_type} Accuracy: {correct}/{total} = {accuracy:.4f}")
                if use_checker:
                    print(f"{problem_type} Checker Corrections: {corrections}")
        
        json.dump(json_results, f, indent=2)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LLM accuracy on VRP classification with Checker")
    parser.add_argument("--data_path", type=str, default="/data1/shy/zgc/llm_solver/LLMSolver/benchmark_hard/data", 
                      help="Path to the directory containing VRP JSON files")
    parser.add_argument("--models", type=str, nargs="+", default=["Qwen/QwQ-32B"], 
                      help="List of model names to test")
    parser.add_argument("--samples", type=int, default=None, 
                      help="Number of samples to test per variant (default: all)")
    parser.add_argument("--no_checker", action="store_true", 
                      help="Disable using the Checker to verify classifications")
    
    args = parser.parse_args()
    
    # Run tests
    results = run_all_tests(args.data_path, args.models, args.samples, not args.no_checker)
    
    print("\nClassification testing complete. Results saved to classification_results_with_checker.json") 