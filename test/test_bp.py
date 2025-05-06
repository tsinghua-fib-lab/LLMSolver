import json
import threading
import queue
import time
import os
from tqdm import tqdm
import argparse
from datetime import datetime
from collections import defaultdict
from recognition.agent import Classifier
from recognition.LLM import LLM_api

def load_problems(json_file):
    """Load problems from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def classify_problem(classifier, problem):
    """Classify a single problem."""
    index = problem.get('index', -1)
    desc = problem.get('desc_split', '')
    true_label = problem.get('label', '').upper()
    
    if not desc:
        return {
            'index': index,
            'true_label': true_label,
            'pred_label': "<Error>",
            'reason': "Missing problem description",
            'correct': False
        }
    
    try:
        pred_label, reason = classifier.run(desc)
        
        return {
            'index': index,
            'desc': desc,
            'true_label': true_label,
            'pred_label': pred_label,
            'reason': reason,
            'correct': true_label == pred_label
        }
    except Exception as e:
        return {
            'index': index,
            'desc': desc,
            'true_label': true_label,
            'pred_label': "<Error>",
            'reason': str(e),
            'correct': False
        }

def worker_function(llm_api, task_queue, log_file, lock, progress_bar, results):
    """Worker thread function."""
    classifier = Classifier(llm_api)
    
    while True:
        try:
            problem = task_queue.get(timeout=1)
            result = classify_problem(classifier, problem)
            with lock:
                with open(log_file, 'a') as f:
                    if not result['correct']:
                        f.write(f'Label: {result["true_label"]}, Predicted: {result["pred_label"]}, Index: {result["index"]}\n')
                        f.write(f'Description: {result["desc"]}\n')
                        f.write(f'Reason: {result["reason"]}\n')
                        f.write("-" * 80 + "\n")
                    
                progress_bar.update(1)
                results.append(result)
                
            task_queue.task_done()
        except queue.Empty:
            break
        except Exception as e:
            print(f"Worker error: {e}")
            task_queue.task_done()

def calculate_statistics(results):
    """Calculate accuracy and misclassification statistics."""
    stats = {
        'by_type': defaultdict(lambda: {'correct': 0, 'total': 0, 'misclassifications': defaultdict(int)}),
        'overall': {'correct': 0, 'total': 0}
    }
    
    for result in results:
        true_label = result['true_label']
        pred_label = result['pred_label']
        stats['overall']['total'] += 1
        stats['by_type'][true_label]['total'] += 1
        
        if result['correct']:
            stats['overall']['correct'] += 1
            stats['by_type'][true_label]['correct'] += 1
        else:
            stats['by_type'][true_label]['misclassifications'][pred_label] += 1
    
    return stats

def write_statistics_to_log(log_file, stats):
    """Write accuracy and misclassification statistics to the log file."""
    with open(log_file, 'a') as f:
        f.write("\n=== Statistics ===\n")
        
        # Write overall accuracy
        overall_correct = stats['overall']['correct']
        overall_total = stats['overall']['total']
        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
        f.write(f"Overall Accuracy: {overall_correct}/{overall_total} = {overall_accuracy:.4f}\n")
        
        # Write per-type statistics
        for problem_type, data in stats['by_type'].items():
            correct = data['correct']
            total = data['total']
            accuracy = correct / total if total > 0 else 0
            f.write(f"\nProblem Type: {problem_type}\n")
            f.write(f"Accuracy: {correct}/{total} = {accuracy:.4f}\n")
            
            # Write top 3 misclassifications
            misclassifications = sorted(data['misclassifications'].items(), key=lambda x: x[1], reverse=True)[:3]
            f.write("Top 3 Misclassifications:\n")
            for pred_label, count in misclassifications:
                f.write(f"  Predicted: {pred_label}, Count: {count}\n")
        f.write("=" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Multithreaded VRP Problem Classifier')
    parser.add_argument('--json_file', type=str, default='/data1/shy/zgc/llm_solver/LLMSolver/benchmark_hard/cvrp/dataset/ovrpltw.json', 
                        help='Path to the JSON file containing problems')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of worker threads')
    parser.add_argument('--models', type=str, nargs='+', default=["deepseek-reasoner", "deepseek-chat", 'qwen-max', 'qwen-plus'],     #["Qwen/QwQ-32B", "deepseek-chat", "deepseek-reasoner"],
                        help='List of LLM models to use')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output file for results')
    args = parser.parse_args()

    problems = load_problems(args.json_file)
    
    valid_problems = []
    for problem in problems:
        if 'desc_split' in problem and 'label' in problem:
            valid_problems.append(problem)
    
    # Get the number of tasks
    num_tasks = len(valid_problems)
    print(f"Loaded {num_tasks} valid problems from {args.json_file}")
    
    # Process problems for each model
    all_results = {}
    for model in args.models:
        print(f"\nClassifying problems using model {model}...")
        
        llm_api = LLM_api(
            model=model,
            key_idx=0,
        )
        
        task_queue = queue.Queue()
        for problem in valid_problems:
            task_queue.put(problem)
        
        # Create a unique log file for this model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = "/data1/shy/zgc/llm_solver/LLMSolver/log"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"classification_log_{model.replace('/', '_')}_{timestamp}.txt")
        
        # Shared resources
        lock = threading.Lock()
        results = []
        
        # Progress bar
        progress_bar = tqdm(total=num_tasks, desc=f"Model {model} classification")
        
        # Create and start worker threads
        threads = []
        for i in range(args.num_workers):
            thread = threading.Thread(
                target=worker_function, 
                args=(llm_api, task_queue, log_file, lock, progress_bar, results)
            )
            thread.daemon = True
            threads.append(thread)
            thread.start()
        
        # Wait for the queue to be empty
        task_queue.join()
        
        # Close the progress bar
        progress_bar.close()
        
        print(f"\nModel {model} classification completed. Results saved to {log_file}")
        
        # Calculate statistics
        stats = calculate_statistics(results)
        
        # Write statistics to the log file
        write_statistics_to_log(log_file, stats)
        
        # Save results to dictionary
        all_results[model] = {
            'log_file': log_file
        }
    
    # Save all model results to file
    if args.output:
        output_file = args.output
    else:
        output_file = "classification_results.json"  # Default filename if none is provided

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll model results saved to {output_file}")

if __name__ == "__main__":
    main()