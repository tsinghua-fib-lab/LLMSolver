from dp.dp_solver import DPSolver
import random
from typing import Dict, List


class kpSolverPool:
    def __init__(self):
        """Initialize the Knapsack Problem Solver Pool"""
        self.dp_solver = DPSolver(problem_type="knapsack")

    def solve(self, inst: Dict, solver_name: str = "dp", problem_type: str = "knapsack", **kwargs) -> Dict:
        """
        Solve knapsack problem using specified solver

        Args:
            inst: Problem instance dictionary
            solver_name: Name of solver to use ("dp", "genetic", etc.)
            problem_type: Type of problem (currently only "knapsack")
            **kwargs: Additional solver parameters

        Returns:
            Dict: Solution with detailed information
        """
        if solver_name == "dp":
            # Use the DP solver directly for single instance
            return self.dp_solver.solve(inst, **kwargs)
        elif solver_name == "genetic":
            # Placeholder for genetic algorithm solver
            raise NotImplementedError("Genetic solver not yet implemented")
        else:
            raise ValueError(f"Unknown solver: {solver_name}")

    def solve_batch(self, batch_inst: List[Dict], solver_name: str = "dp", **kwargs) -> List[Dict]:
        """
        Solve multiple knapsack problem instances

        Args:
            batch_inst: List of problem instances
            solver_name: Name of solver to use
            **kwargs: Additional solver parameters

        Returns:
            List[Dict]: List of solutions
        """
        if solver_name == "dp":
            # Use the DP solver's batch method
            return self.dp_solver.solve_batch(batch_inst, **kwargs)
        elif solver_name == "genetic":
            # Placeholder for genetic algorithm solver
            raise NotImplementedError("Genetic solver not yet implemented")
        else:
            raise ValueError(f"Unknown solver: {solver_name}")


def test_dp_solver():
    """Test the solver pool with various instances"""
    # Test example
    data = {
        "item_weight": [round(random.uniform(0, 1), 4) for _ in range(20)],
        "item_value": [round(random.uniform(0, 1), 4) for _ in range(20)],
        "knapsack_capacity": 5
    }

    # Create solver pool
    kp_solve_pool = kpSolverPool()

    print("Testing KP Solver Pool")
    print("=" * 40)

    # Test single instance
    print("Single instance solution:")
    solution = kp_solve_pool.solve(data, solver_name="dp", problem_type="knapsack")

    print(f"Selected items: {solution['selected_items']}")
    print(f"Total weight: {solution['total_weight']:.4f}")
    print(f"Total value: {solution['total_value']:.4f}")
    print(f"Capacity utilization: {solution['capacity_utilization']:.4f}")
    print(f"Solve time: {solution['solve_time']:.6f} seconds")
    print(f"Algorithm: {solution['algorithm']}")
    print(f"Optimal: {solution['optimal']}")

    # Test batch solving
    print(f"\n{'=' * 40}")
    print("Batch solving test:")

    # Create multiple test instances
    batch_data = []
    for i in range(3):
        instance = {
            "item_weight": [round(random.uniform(0.1, 1.0), 2) for _ in range(10)],
            "item_value": [round(random.uniform(0.1, 1.0), 2) for _ in range(10)],
            "knapsack_capacity": 3.0
        }
        batch_data.append(instance)

    batch_solutions = kp_solve_pool.solve_batch(batch_data, solver_name="dp")

    for i, sol in enumerate(batch_solutions):
        print(f"\nInstance {i + 1}:")
        print(f"  Selected items: {sol['selected_items']}")
        print(f"  Total value: {sol['total_value']:.4f}")
        print(f"  Solve time: {sol['solve_time']:.6f} seconds")


if __name__ == '__main__':
    test_dp_solver()
