import time
from typing import Dict, List, Tuple


##########################################################################################
# format

def format_instance(instance: Dict) -> Dict:
    """
    Format a single instance for the DP solver.

    Args:
        instance: Problem instance dict

    Returns:
        Dict: Formatted instance
    """
    if 'item_weight' not in instance or 'item_value' not in instance or 'knapsack_capacity' not in instance:
        raise ValueError("Instance must contain 'item_weight', 'item_value', and 'knapsack_capacity'")

    weights = list(instance['item_weight'])
    values = list(instance['item_value'])
    capacity = float(instance['knapsack_capacity'])

    if len(weights) != len(values):
        raise ValueError("Number of weights must equal number of values")
    if capacity <= 0:
        raise ValueError("Knapsack capacity must be positive")

    return {
        'item_weight': weights,
        'item_value': values,
        'knapsack_capacity': capacity
    }


def format_result(selected_items: List[int], instance: Dict) -> Dict:
    """
    Format the DP solution result.

    Args:
        selected_items: List of selected item indices
        instance: Original problem instance

    Returns:
        Dict: Formatted result
    """
    weights = instance['item_weight']
    values = instance['item_value']

    total_weight = sum(weights[i] for i in selected_items)
    total_value = sum(values[i] for i in selected_items)

    return {
        'selected_items': selected_items,
        'total_weight': total_weight,
        'total_value': total_value,
        'capacity_utilization': total_weight / instance['knapsack_capacity']
    }


##########################################################################################
# DP algorithm implementation

def knapsack_dp(weights: List[float], values: List[float], capacity: float, precision: int = 4) -> Tuple[float, List[int], float]:
    """
    Dynamic Programming solution for 0-1 Knapsack Problem with floating-point weights.

    Args:
        weights: List of item weights
        values: List of item values
        capacity: Knapsack capacity
        precision: Decimal precision for floating-point handling

    Returns:
        Tuple[float, List[int], float]: (max_value, selected_items, total_weight)
    """
    scale = 10 ** precision
    int_weights = [int(w * scale + 0.5) for w in weights]
    int_capacity = int(capacity * scale + 0.5)
    n = len(weights)

    dp = [[0] * (int_capacity + 1) for _ in range(n + 1)]
    keep = [[False] * (int_capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w = int_weights[i - 1]
        v = values[i - 1]
        for j in range(int_capacity + 1):
            if w > j:
                dp[i][j] = dp[i - 1][j]
            else:
                if dp[i - 1][j] < dp[i - 1][j - w] + v:
                    dp[i][j] = dp[i - 1][j - w] + v
                    keep[i][j] = True
                else:
                    dp[i][j] = dp[i - 1][j]

    res_value = dp[n][int_capacity]
    res_items = []
    total_weight_int = 0
    j = int_capacity
    for i in range(n, 0, -1):
        if keep[i][j]:
            res_items.append(i - 1)
            total_weight_int += int_weights[i - 1]
            j -= int_weights[i - 1]
    res_items.reverse()
    total_weight = total_weight_int / scale
    return res_value, res_items, total_weight


##########################################################################################
# main

class DPSolver:
    def __init__(self, problem_type: str = "knapsack"):
        """
        Initialize DP Solver for Knapsack Problems.

        Args:
            problem_type: Type of problem (currently only supports "knapsack")
        """
        if problem_type not in ["knapsack"]:
            raise NotImplementedError(f'{problem_type} is not implemented')
        self.problem_type = problem_type

    def solve(self, inst: Dict, **params) -> Dict:
        """
        Solve a single knapsack problem instance.

        Args:
            inst: Problem instance
            **params: Additional parameters

        Returns:
            Dict: Solution
        """
        formatted_instance = format_instance(inst)
        start_time = time.time()
        precision = params.get('precision', 4)
        
        max_value, selected_items, total_weight = knapsack_dp(
            formatted_instance['item_weight'],
            formatted_instance['item_value'],
            formatted_instance['knapsack_capacity'],
            precision
        )
        
        solve_time = time.time() - start_time
        result = format_result(selected_items, formatted_instance)
        result.update({
            'solve_time': solve_time,
            'algorithm': 'dynamic_programming',
            'optimal': True
        })
        return result

    def solve_batch(self, batch_inst: List[Dict], **params) -> List[Dict]:
        """
        Solve multiple knapsack problem instances.

        Args:
            batch_inst: List of problem instances
            **params: Additional parameters

        Returns:
            List[Dict]: List of solutions
        """
        return [self.solve(inst, **params) for inst in batch_inst]


##########################################################################################
# Test

def test_solver():
    """Test the DP solver with various instances"""
    import random
    from envs.kp.env import KPEnv

    # Create test instances
    test_instances = [
        {
            'item_weight': [0.5, 0.3, 0.8, 0.2, 0.6],
            'item_value': [0.8, 0.5, 1.2, 0.3, 0.9],
            'knapsack_capacity': 1.0
        },
        {
            'item_weight': [round(random.uniform(0.1, 1.0), 2) for _ in range(10)],
            'item_value': [round(random.uniform(0.1, 1.0), 2) for _ in range(10)],
            'knapsack_capacity': 3.0
        },
        {
            'item_weight': [round(random.uniform(0.1, 1.0), 2) for _ in range(20)],
            'item_value': [round(random.uniform(0.1, 1.0), 2) for _ in range(20)],
            'knapsack_capacity': 5.0
        }
    ]

    solver = DPSolver(problem_type="knapsack")
    print("Testing DP Solver for Knapsack Problems")
    print("=" * 50)

    for i, instance in enumerate(test_instances):
        print(f"\nInstance {i + 1}:")
        print(f"Number of items: {len(instance['item_weight'])}")
        print(f"Capacity: {instance['knapsack_capacity']}")
        solution = solver.solve(instance)
        print(f"Selected items: {solution['selected_items']}")
        print(f"Total weight: {solution['total_weight']:.4f}")
        print(f"Total value: {solution['total_value']:.4f}")
        print(f"Capacity utilization: {solution['capacity_utilization']:.4f}")
        print(f"Solve time: {solution['solve_time']:.6f} seconds")
        print(f"Optimal: {solution['optimal']}")

    # Test batch solving
    print(f"\n{'=' * 50}")
    print("Testing batch solving:")
    batch_solutions = solver.solve_batch(test_instances)
    print(f"Solved {len(batch_solutions)} instances in batch")

    env = KPEnv(problem_type="knapsack")
    for instance, solution in zip(test_instances, batch_solutions):
        print(f"Valid: {env.is_valid(instance, solution)}")
        print(f"Reward: {env.get_reward(instance, solution)}")
        print(f"Selected items: {solution['selected_items']}")
        env.plot_solution(instance, solution)
    total_time = sum(sol['solve_time'] for sol in batch_solutions)
    print(f"Total solve time: {total_time:.6f} seconds")


if __name__ == '__main__':
    test_solver()
