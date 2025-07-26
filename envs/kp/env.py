import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Tuple, Optional


class KPEnv:
    def __init__(self, problem_type: str = "knapsack"):
        """
        Initialize Knapsack Problem Environment
        
        Args:
            problem_type: Type of knapsack problem (currently only supports "knapsack")
        """
        if problem_type not in ["knapsack"]:
            raise NotImplementedError(f'{problem_type} is not implemented')
            
        self.problem_type = problem_type

    def is_valid(self, instances: dict, solution: dict) -> bool:
        """
        Check if a solution is valid for the knapsack problem
        
        Args:
            instances: Problem instance containing item weights, values, and capacity
            solution: Solution containing selected items
            
        Returns:
            bool: True if solution is valid, False otherwise
        """
        try:
            # Extract problem data
            item_weights = instances['item_weight']
            knapsack_capacity = instances['knapsack_capacity']
            
            # Extract solution data
            selected_items = solution.get('selected_items', [])
            
            # Check if all selected item indices are valid
            if not isinstance(selected_items, list):
                return False
                
            for item_idx in selected_items:
                if not isinstance(item_idx, int) or item_idx < 0 or item_idx >= len(item_weights):
                    return False
            
            # Check if total weight exceeds capacity
            total_weight = sum(item_weights[item_idx] for item_idx in selected_items)
            
            return total_weight <= knapsack_capacity
            
        except (KeyError, TypeError, IndexError):
            return False

    def get_reward(self, instances: dict, solution: dict) -> float:
        """
        Calculate the reward (total value) for a knapsack solution
        
        Args:
            instances: Problem instance containing item weights, values, and capacity
            solution: Solution containing selected items
            
        Returns:
            float: Total value of selected items, or -inf if invalid
        """
        # First check if solution is valid
        if not self.is_valid(instances, solution):
            return float('-inf')
        
        # Extract problem data
        item_values = instances['item_value']
        selected_items = solution.get('selected_items', [])
        
        # Calculate total value
        total_value = sum(item_values[item_idx] for item_idx in selected_items)
        
        return total_value

    def get_solution_info(self, instances: dict, solution: dict) -> Dict:
        """
        Get detailed information about a solution
        
        Args:
            instances: Problem instance
            solution: Solution to analyze
            
        Returns:
            Dict: Information about the solution including weight, value, utilization
        """
        if not self.is_valid(instances, solution):
            return {
                'valid': False,
                'total_weight': 0,
                'total_value': 0,
                'capacity_utilization': 0,
                'num_selected': 0
            }
        
        item_weights = instances['item_weight']
        item_values = instances['item_value']
        knapsack_capacity = instances['knapsack_capacity']
        selected_items = solution.get('selected_items', [])
        
        total_weight = sum(item_weights[item_idx] for item_idx in selected_items)
        total_value = sum(item_values[item_idx] for item_idx in selected_items)
        capacity_utilization = total_weight / knapsack_capacity if knapsack_capacity > 0 else 0
        
        return {
            'valid': True,
            'total_weight': total_weight,
            'total_value': total_value,
            'capacity_utilization': capacity_utilization,
            'num_selected': len(selected_items),
            'capacity': knapsack_capacity
        }

    def plot_solution(self, instances: dict, solution: dict, title: str = "Knapsack Solution"):
        """
        Visualize the knapsack solution
        
        Args:
            instances: Problem instance
            solution: Solution to visualize
            title: Plot title
        """
        if not self.is_valid(instances, solution):
            print("Invalid solution - cannot plot")
            return
        
        item_weights = instances['item_weight']
        item_values = instances['item_value']
        knapsack_capacity = instances['knapsack_capacity']
        selected_items = solution.get('selected_items', [])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Weight vs Value scatter plot
        all_weights = np.array(item_weights)
        all_values = np.array(item_values)
        
        # Plot all items
        ax1.scatter(all_weights, all_values, alpha=0.6, color='lightblue', label='All Items')
        
        # Highlight selected items
        if selected_items:
            selected_weights = [item_weights[i] for i in selected_items]
            selected_values = [item_values[i] for i in selected_items]
            ax1.scatter(selected_weights, selected_values, color='red', s=100, label='Selected Items')
        
        ax1.set_xlabel('Weight')
        ax1.set_ylabel('Value')
        ax1.set_title('Item Weight vs Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add capacity line
        ax1.axvline(x=knapsack_capacity, color='green', linestyle='--', label=f'Capacity: {knapsack_capacity}')
        
        # Plot 2: Solution summary
        solution_info = self.get_solution_info(instances, solution)
        
        # Create bar chart
        categories = ['Total Weight', 'Total Value', 'Capacity Utilization']
        values = [
            solution_info['total_weight'],
            solution_info['total_value'],
            solution_info['capacity_utilization']
        ]
        
        colors = ['orange', 'green', 'blue']
        bars = ax2.bar(categories, values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        ax2.set_ylabel('Value')
        ax2.set_title('Solution Summary')
        ax2.grid(True, alpha=0.3)
        
        # Add capacity line for utilization
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='100% Utilization')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def get_optimal_solution(self, instances: dict) -> dict:
        """
        Get the optimal solution using branch and bound for floating-point weights
        
        Args:
            instances: Problem instance
            
        Returns:
            dict: Optimal solution
        """
        item_weights = instances['item_weight']
        item_values = instances['item_value']
        capacity = instances['knapsack_capacity']
        
        n = len(item_weights)
        
        # For small instances, use branch and bound
        if n <= 50:  # Reasonable limit for exact solution
            # Sort items by value-to-weight ratio for better pruning
            items = [(i, item_weights[i], item_values[i]) for i in range(n)]
            items.sort(key=lambda x: x[2]/x[1], reverse=True)
            
            best_value = 0
            best_solution = []
            
            def branch_and_bound(index, current_weight, current_value, selected):
                nonlocal best_value, best_solution
                
                # Prune if we've exceeded capacity
                if current_weight > capacity:
                    return
                
                # Update best solution if current is better
                if current_value > best_value:
                    best_value = current_value
                    best_solution = selected.copy()
                
                # Prune if we can't improve
                if index >= n:
                    return
                
                # Calculate upper bound (greedy relaxation)
                remaining_capacity = capacity - current_weight
                upper_bound = current_value
                for i in range(index, n):
                    if items[i][1] <= remaining_capacity:
                        upper_bound += items[i][2]
                        remaining_capacity -= items[i][1]
                    else:
                        upper_bound += items[i][2] * (remaining_capacity / items[i][1])
                        break
                
                if upper_bound <= best_value:
                    return
                
                # Try including current item
                if current_weight + items[index][1] <= capacity:
                    branch_and_bound(index + 1, 
                                   current_weight + items[index][1], 
                                   current_value + items[index][2], 
                                   selected + [items[index][0]])
                
                # Try excluding current item
                branch_and_bound(index + 1, current_weight, current_value, selected)
            
            # Start branch and bound
            branch_and_bound(0, 0, 0, [])
            
            return {
                'selected_items': sorted(best_solution),
                'optimal_value': best_value
            }
        else:
            # For large instances, return greedy solution as approximation
            return self.get_greedy_solution(instances)

    def get_greedy_solution(self, instances: dict) -> dict:
        """
        Get a greedy solution based on value-to-weight ratio
        
        Args:
            instances: Problem instance
            
        Returns:
            dict: Greedy solution
        """
        item_weights = instances['item_weight']
        item_values = instances['item_value']
        capacity = instances['knapsack_capacity']
        
        # Calculate value-to-weight ratios
        ratios = [(i, item_values[i] / item_weights[i]) for i in range(len(item_weights))]
        ratios.sort(key=lambda x: x[1], reverse=True)
        
        selected_items = []
        remaining_capacity = capacity
        
        for item_idx, ratio in ratios:
            if item_weights[item_idx] <= remaining_capacity:
                selected_items.append(item_idx)
                remaining_capacity -= item_weights[item_idx]
        
        return {
            'selected_items': selected_items,
            'greedy_value': sum(item_values[i] for i in selected_items)
        }


def test():
    """Test the KPEnv class"""
    # Create environment
    env = KPEnv(problem_type="knapsack")
    
    # Create a test instance
    test_instance = {
        'item_weight': [0.5, 0.3, 0.8, 0.2, 0.6],
        'item_value': [0.8, 0.5, 1.2, 0.3, 0.9],
        'knapsack_capacity': 1.0
    }
    
    # Test valid solution
    valid_solution = {
        'selected_items': [1, 3]  # Items with weights 0.3, 0.2 = 0.5 (invalid)
    }
    
    print("Testing valid solution:")
    print(f"Is valid: {env.is_valid(test_instance, valid_solution)}")
    print(f"Reward: {env.get_reward(test_instance, valid_solution)}")
    
    # Test invalid solution (exceeds capacity)
    invalid_solution = {
        'selected_items': [0, 1, 2]  # Items with weights 0.5, 0.3, 0.8 = 1.6 > 1.0
    }
    
    print("\nTesting invalid solution:")
    print(f"Is valid: {env.is_valid(test_instance, invalid_solution)}")
    print(f"Reward: {env.get_reward(test_instance, invalid_solution)}")
    
    # Test greedy solution
    greedy_solution = env.get_greedy_solution(test_instance)
    print(f"\nGreedy solution: {greedy_solution}")
    print(f"Greedy reward: {env.get_reward(test_instance, greedy_solution)}")
    
    # Test optimal solution
    optimal_solution = env.get_optimal_solution(test_instance)
    print(f"\nOptimal solution: {optimal_solution}")
    print(f"Optimal reward: {env.get_reward(test_instance, optimal_solution)}")
    
    # Test solution info
    solution_info = env.get_solution_info(test_instance, optimal_solution)
    print(f"\nSolution info: {solution_info}")
    
    # Plot solution
    try:
        env.plot_solution(test_instance, optimal_solution, "Test Knapsack Solution")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == '__main__':
    test()
