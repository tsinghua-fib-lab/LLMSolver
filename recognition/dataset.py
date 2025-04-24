import os
import json
import random
from typing import Dict, List, Optional, Union, Tuple
import torch
from torch.utils.data import Dataset as TorchDataset

class Dataset(TorchDataset):
    """
    Dataset class: Used to load VRP problem descriptions and labels from JSON files.
    """
    def __init__(
        self,
        data_dir: str = "/data1/shy/zgc/llm_solver/LLMSolver/benchmark_hard/cvrp/dataset",
        problem_counts: int = 100,
        seed: int = 42,
        shuffle: bool = False
    ):
        """
        Initialize the dataset by loading problem descriptions from JSON files.
        
        Args:
            data_dir: Directory containing JSON files with problem descriptions.
            problem_counts: Number of samples to load for each problem type.
            seed: Random seed for reproducibility.
            shuffle: Whether to shuffle the dataset.
        """
        super().__init__()
        self.data_dir = data_dir
        self.problem_counts = problem_counts
        self.shuffle = shuffle
        
        random.seed(seed)
        
        self.samples = self._load_data()
        
        # Shuffle samples if required
        if self.shuffle:
            random.shuffle(self.samples)
        
    def _load_data(self) -> List[Dict]:
        """
        Load problem descriptions and labels from JSON files.
        
        Returns:
            A list of dictionaries containing problem information.
        """
        samples = []
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        
        for filename in json_files:
            problem_type = os.path.splitext(filename)[0]
            file_path = os.path.join(self.data_dir, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                all_samples = []
                if isinstance(data, list):
                    for item in data:
                        if 'desc_split' in item and 'label' in item and 'index' in item:
                            sample = {
                                'desc_split': item['desc_split'],
                                'label': item['label'],
                                'index': item['index'],
                            }
                            all_samples.append(sample)
                elif isinstance(data, dict):
                    if 'desc_split' in data and 'label' in data and 'index' in data:
                        sample = {
                            'description': data['desc_split'],
                            'label': data['label'],
                            'index': data['index']
                        }
                        all_samples.append(sample)
                    else:
                        for key, value in data.items():
                            if isinstance(value, dict) and 'desc_split' in value and 'label' in value and 'index' in value:
                                sample = {
                                    'description': value['desc_split'],
                                    'label': value['label'],
                                    'index': value['index']
                                }
                                all_samples.append(sample)
                
                if not all_samples:
                    print(f"Warning: No valid samples found in {filename}")
                    continue
                
                if self.problem_counts:
                    num_samples = self.problem_counts
                    if len(all_samples) < num_samples:
                        print(f"Warning: {problem_type} has fewer samples ({len(all_samples)}) than required ({num_samples}). Using all available samples.")
                        num_samples = len(all_samples)
                    all_samples = random.sample(all_samples, num_samples)
                
                samples.extend(all_samples)
                
            except Exception as e:
                print(f"Error loading data from {filename}: {e}")
        
        return samples
    
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Return a sample from the dataset.
        
        Args:
            idx: Index of the sample to return.
            
        Returns:
            A dictionary containing problem description and label.
        """
        return self.samples[idx]
    
    def get_problems_by_type(self) -> Dict[str, int]:
        """
        Get the distribution of problems by type in the dataset.
        
        Returns:
            A dictionary mapping problem types to their counts.
        """
        problem_counts = {}
        for sample in self.samples:
            variant_name = sample['label']
            if variant_name in problem_counts:
                problem_counts[variant_name] += 1
            else:
                problem_counts[variant_name] = 1
        return problem_counts
    
    def create_subset(self, indices: List[int]) -> 'Dataset':
        """
        Create a subset of the dataset using specified indices.
        
        Args:
            indices: List of indices to include in the subset.
            
        Returns:
            A new Dataset instance containing only the specified samples.
        """
        subset = Dataset(self.data_dir, None)
        subset.samples = [self.samples[i] for i in indices]
        return subset
    
    def save_to_file(self, filename: str) -> None:
        """
        Save the current dataset to a JSON file.
        
        Args:
            filename: Name of the file to save the dataset.
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.samples, f, ensure_ascii=False, indent=4)
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'Dataset':
        """
        Load a dataset from a JSON file.
        
        Args:
            filename: Name of the file to load the dataset from.
            
        Returns:
            A new Dataset instance loaded from the file.
        """
        dataset = cls(None, None)
        dataset.data_dir = None
        with open(filename, 'r', encoding='utf-8') as f:
            dataset.samples = json.load(f)
        return dataset
    
if __name__ == '__main__':
    problem_counts = 1
    dataset = Dataset(
        data_dir="/data1/shy/zgc/llm_solver/LLMSolver/benchmark_hard/cvrp/dataset",
        shuffle=True,
        problem_counts=problem_counts,
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Problem type distribution: {dataset.get_problems_by_type()}")
    
    # subset = dataset.create_subset([0, 1, 2])
    # print(f"Subset size: {len(subset)}")
    
    dataset.save_to_file(f"vrp_{problem_counts}.json")
    # loaded_dataset = Dataset.load_from_file("dataset.json")
    # print(f"Loaded dataset size: {len(loaded_dataset)}")
