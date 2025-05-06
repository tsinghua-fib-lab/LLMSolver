import os
import json
import time
import argparse
from tqdm import tqdm
from collections import defaultdict
from recognition.LLM import LLM_api
from recognition.agent import Classifier, Checker, Extractor
import datetime
from recognition.problems import SP_VARIANTS as P_VARIANTS

def load_json_file(file_path):
    """Load and parse a JSON file."""
    print(f"Loading JSON file from {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def process_batch(classifier, problems, start_idx, end_idx, true_label, log_file, model_name, num_threads):
    """Process a batch of problems using either multi-threaded or single-threaded API calls."""
    batch_problems = problems[start_idx:end_idx]
    descriptions = [p.get('desc_split', '') for p in batch_problems]
    
    # Filter out empty descriptions and keep track of their indices
    valid_indices = [i for i, desc in enumerate(descriptions) if desc]
    valid_descriptions = [descriptions[i] for i in valid_indices]
    
    if not valid_descriptions:
        return 0, 0

    try:
        # 获取元组列表
        tuples = classifier.run(valid_descriptions, num_threads=num_threads)
    except Exception as e:
        print(f"Error in classifier.run: {e}")
        return 0, 0
    
    correct_count = 0
    total_count = len(valid_descriptions)
    
    for idx, (choice, reason) in enumerate(tuples):
        original_idx = valid_indices[idx] + start_idx
        desc = valid_descriptions[idx]
        
        try:
            if choice == "<Error>":
                # 处理分类器返回的错误
                with open(log_file, 'a') as f:
                    f.write(f"Model: {model_name}, Problem Type: {true_label}, Index: {original_idx}\n")
                    f.write(f"Description: {desc}\n")
                    f.write(f"Error: {reason}\n")
                    f.write("-" * 80 + "\n")
                total_count -= 1  # 错误样本不计入总数
            elif choice.strip() == true_label:
                correct_count += 1
            else:
                # 预测错误，记录日志
                with open(log_file, 'a') as f:
                    f.write(f"Model: {model_name}, Problem Type: {true_label}, Index: {original_idx}\n")
                    f.write(f"Description: {desc}\n")
                    f.write(f"Predicted: {choice.strip()}\n")
                    f.write(f"Reason: {reason}\n")
                    f.write("-" * 80 + "\n")
        except Exception as e:
            # 处理意外异常
            with open(log_file, 'a') as f:
                f.write(f"Model: {model_name}, Problem Type: {true_label}, Index: {original_idx}\n")
                f.write(f"Description: {desc}\n")
                f.write(f"Unexpected Error: {str(e)}\n")
                f.write("-" * 80 + "\n")
            total_count -= 1
    
    return correct_count, total_count

def test_model_on_vrp_variant(json_path, model_name, log_file, num_samples=None, num_threads=1):
    """
    Test a specific language model on a VRP variant using batched API calls.
    
    Args:
        json_path: Path to the JSON file containing VRP problems
        model_name: Name of the language model to test
        log_file: File to log errors
        num_samples: Number of samples to test (None for all)
        batch_size: Number of problems to process in each batch
        num_threads: Number of threads to use for API calls
    """
    print(f"Testing model {model_name} on {json_path}")
    
    # Initialize LLM API
    llm = LLM_api(model=model_name, max_tokens=8192)
    
    # Initialize classifier
    classifier = Classifier(llm_api=llm)
    
    # Load problems
    problems = load_json_file(json_path)
    
    # Determine the true label from the file
    variant_name_list = P_VARIANTS
    variant_name = os.path.basename(json_path).split('.')[0]
    if variant_name.upper() in variant_name_list:
        true_label = variant_name.upper()
    else:
        return 0, 0, variant_name
    
    # Limit samples if specified
    if num_samples is not None:
        problems = problems[:num_samples]
    
    total_problems = len(problems)
    correct_count = 0
    total_count = 0
    
    # Process problems in batches
    for start_idx in tqdm(range(0, total_problems, num_threads), desc=f"Testing {variant_name}"):
        end_idx = min(start_idx + num_threads, total_problems)
        batch_correct, batch_total = process_batch(
            classifier, problems, start_idx, end_idx, 
            true_label, log_file, model_name, num_threads
        )
        correct_count += batch_correct
        total_count += batch_total
    
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Model: {model_name}, Problem Type: {true_label}")
    print(f"Accuracy: {correct_count}/{total_count} = {accuracy:.4f}")
    
    return correct_count, total_count, true_label

def run_all_tests(data_path, models, num_samples=None, num_threads=1):
    """
    Run tests for all VRP variants on all specified models.
    
    Args:
        data_path: Path to directory containing VRP JSON files
        models: List of model names to test
        num_samples: Number of samples per variant to test (default: None)
        num_threads: Number of threads to use for API calls
    """
    results = {}
    all_jsons = [f for f in os.listdir(data_path) if f.endswith('.json') and not f.endswith('_meta.json')]
    
    for model_name in models:
        model_results = defaultdict(list)
        overall_correct = 0
        overall_total = 0
        
        # Log file for recording misclassifications
        log_file = f"log/error_log_{model_name.replace('/', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_file, 'w') as f:
            f.write(f"Error log for model: {model_name}\n")
            f.write("=" * 80 + "\n")
        
        for json_file in all_jsons:
            json_path = os.path.join(data_path, json_file)
            correct, total, problem_type = test_model_on_vrp_variant(
                json_path, model_name, log_file, num_samples, num_threads
            )
            
            model_results[problem_type].append((correct, total))
            overall_correct += correct
            overall_total += total
        
        # Calculate aggregate results
        results[model_name] = {
            'by_type': {},
            'overall': (overall_correct, overall_total, overall_correct / overall_total if overall_total > 0 else 0)
        }
        
        for problem_type, counts in model_results.items():
            type_correct = sum(c[0] for c in counts)
            type_total = sum(c[1] for c in counts)
            results[model_name]['by_type'][problem_type] = (
                type_correct, 
                type_total, 
                type_correct / type_total if type_total > 0 else 0
            )
    
    # Print and save overall results
    with open(f"log/classification_results_{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json_results = {}
        
        for model_name, model_data in results.items():
            json_results[model_name] = {
                'overall': {
                    'correct': model_data['overall'][0],
                    'total': model_data['overall'][1],
                    'accuracy': model_data['overall'][2]
                },
                'by_type': {}
            }
            
            print(f"\n=== Model: {model_name} ===")
            print(f"Overall Accuracy: {model_data['overall'][0]}/{model_data['overall'][1]} = {model_data['overall'][2]:.4f}")
            
            for problem_type, (correct, total, accuracy) in model_data['by_type'].items():
                json_results[model_name]['by_type'][problem_type] = {
                    'correct': correct,
                    'total': total,
                    'accuracy': accuracy
                }
                print(f"{problem_type} Accuracy: {correct}/{total} = {accuracy:.4f}")
        
        json.dump(json_results, f, indent=2)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LLM accuracy on VRP classification")
    parser.add_argument("--data_path", type=str, default="/data1/shy/zgc/llm_solver/LLMSolver/recognition/test/test_data", 
                        help="Path to the directory containing VRP JSON files")
    parser.add_argument("--models", type=str, nargs="+", default=["Qwen/QwQ-32B", "deepseek-reasoner"], 
                        help="List of model names to test")
    parser.add_argument("--samples", type=int, default=None, 
                        help="Number of samples to test per variant (default: all)")
    parser.add_argument("--num_threads", type=int, default=1, 
                        help="Number of threads to use for API calls")
    
    args = parser.parse_args()
    
    # Run tests
    results = run_all_tests(args.data_path, args.models, args.samples, args.num_threads)
    
    print("\nClassification testing complete. Results saved to classification_results.json")