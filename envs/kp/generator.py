import random
from enum import Enum
from collections import defaultdict
from typing import Dict, List, Tuple, Union, OrderedDict
import json
import os

from benchmark.msp.machine_scheduing_config import problem_type_param


class KPProblemType(Enum):
    """Knapsack Problem types"""
    KNAPSACK = 'knapsack'  # Standard Knapsack Problem


class KPGenerator:
    def __init__(self,
                 problem_type: str,
                 min_items: int = 50,
                 max_items: int = 100,
                 min_capacity: int = 5,
                 max_capacity: int = 20,
                 min_weight: float = 0.0,
                 max_weight: float = 1.0,
                 min_value: float = 0.0,
                 max_value: float = 1.0):
        """
        Initialize KP Generator
        
        Args:
            problem_type: Type of knapsack problem
            min_items: Minimum number of items to generate
            max_items: Maximum number of items to generate
            min_capacity: Minimum knapsack capacity
            max_capacity: Maximum knapsack capacity
            min_weight: Minimum item weight
            max_weight: Maximum item weight
            min_value: Minimum item value
            max_value: Maximum item value
        """
        if problem_type == 'knapsack':
            self.problem_type = KPProblemType.KNAPSACK
        else:
            raise NotImplementedError(f'{problem_type} is not implemented')

        self.min_items = min_items
        self.max_items = max_items
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_value = min_value
        self.max_value = max_value

    def _generate_uniform_numbers(self, n: int, min_val: float, max_val: float) -> List[float]:
        """Generate n uniform random numbers between min_val and max_val"""
        return [round(random.uniform(min_val, max_val), 4) for _ in range(n)]

    def _data_generation(self) -> Dict:
        """Generate a single knapsack problem instance"""
        # Generate random number of items
        n_items = random.randint(self.min_items, self.max_items)
        
        # Generate item weights and values
        item_weights = self._generate_uniform_numbers(n_items, self.min_weight, self.max_weight)
        item_values = self._generate_uniform_numbers(n_items, self.min_value, self.max_value)
        
        # Generate knapsack capacity
        knapsack_capacity = random.randint(self.min_capacity, self.max_capacity)
        
        return {
            "item_weight": item_weights,
            "item_value": item_values,
            "knapsack_capacity": knapsack_capacity
        }

    def generate(self, batch_size: int) -> List[Dict]:
        """
        Generate a batch of knapsack problem instances
        
        Args:
            batch_size: Number of instances to generate
            
        Returns:
            List of dictionaries containing problem instances
        """
        instances = []
        
        for _ in range(batch_size):
            data = self._data_generation()
            instances.append(data)
            
        return instances

    def generate_from_benchmark(self, batch_size: int, dataset_path: str = "benchmark/dataset/kp.json") -> List[Dict]:
        """
        Generate instances from benchmark dataset
        
        Args:
            batch_size: Number of instances to generate
            dataset_path: Path to the benchmark dataset
            
        Returns:
            List of dictionaries containing problem instances
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            
        # Randomly sample from dataset
        if batch_size > len(dataset):
            batch_size = len(dataset)
            
        selected_instances = random.sample(dataset, batch_size)
        
        # Extract data_template from each instance
        instances = []
        for instance in selected_instances:
            if "data_template" in instance:
                instances.append(instance["data_template"])
            else:
                # Fallback to generating new data if template not found
                instances.append(self._data_generation())
                
        return instances


# Example usage
if __name__ == "__main__":
    # Test the generator
    generator = KPGenerator(problem_type="knapsack")
    
    # Generate synthetic instances
    synthetic_instances = generator.generate(batch_size=5)
    print("Generated synthetic instances:")
    for i, instance in enumerate(synthetic_instances):
        print(f"Instance {i+1}: {len(instance['item_weight'])} items, capacity {instance['knapsack_capacity']}")
    
    # Generate from benchmark dataset
    try:
        benchmark_instances = generator.generate_from_benchmark(batch_size=3)
        print("\nGenerated from benchmark:")
        for i, instance in enumerate(benchmark_instances):
            print(f"Instance {i+1}: {len(instance['item_weight'])} items, capacity {instance['knapsack_capacity']}")
    except FileNotFoundError as e:
        print(f"Benchmark dataset not found: {e}")


