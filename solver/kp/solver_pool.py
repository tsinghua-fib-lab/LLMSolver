from dp.dp_solver import DPSolver
from pomo.pomo_solver import POMOSolver
import random
from typing import Dict, List


class kpSolverPool:
    def __init__(self):
        """Initialize the Knapsack Problem Solver Pool"""
        self.dp_solver = DPSolver(problem_type="knapsack")
        self.pomo_solver = None  # Will be initialized when needed
        
    def _get_pomo_solver(self, model_path: str = None):
        """Lazy initialization of POMO solver"""
        if self.pomo_solver is None:
            self.pomo_solver = POMOSolver(problem_type="knapsack", model_path=model_path)
        return self.pomo_solver

    def solve(self, inst: Dict, solver_name: str = "dp", problem_type: str = "knapsack", **kwargs) -> Dict:
        """
        Solve knapsack problem using specified solver
        
        Args:
            inst: Problem instance dictionary
            solver_name: Name of solver to use ("dp", "pomo", "genetic", etc.)
            problem_type: Type of problem (currently only "knapsack")
            **kwargs: Additional solver parameters
            
        Returns:
            Dict: Solution with detailed information
        """
        if solver_name == "dp":
            # Use the DP solver directly for single instance
            return self.dp_solver.solve(inst, **kwargs)
        elif solver_name == "pomo":
            # Use the POMO solver
            model_path = kwargs.get('model_path', None)
            pomo_solver = self._get_pomo_solver(model_path)
            return pomo_solver.solve(inst, **kwargs)
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
        elif solver_name == "pomo":
            # Use the POMO solver's batch method
            model_path = kwargs.get('model_path', None)
            pomo_solver = self._get_pomo_solver(model_path)
            return pomo_solver.solve_batch(batch_inst, **kwargs)
        elif solver_name == "genetic":
            # Placeholder for genetic algorithm solver
            raise NotImplementedError("Genetic solver not yet implemented")
        else:
            raise ValueError(f"Unknown solver: {solver_name}")


def test_solver_pool():
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

    # Test DP solver
    print("DP solver solution:")
    dp_solution = kp_solve_pool.solve(data, solver_name="dp", problem_type="knapsack")
    print(f"  Selected items: {dp_solution['selected_items']}")
    print(f"  Total weight: {dp_solution['total_weight']:.4f}")
    print(f"  Total value: {dp_solution['total_value']:.4f}")
    print(f"  Capacity utilization: {dp_solution['capacity_utilization']:.4f}")
    print(f"  Solve time: {dp_solution['solve_time']:.6f} seconds")
    print(f"  Algorithm: {dp_solution['algorithm']}")
    print(f"  Optimal: {dp_solution['optimal']}")

    # Test POMO solver
    print("\nPOMO solver solution:")
    try:
        pomo_solution = kp_solve_pool.solve(data, solver_name="pomo", problem_type="knapsack")
        print(f"  Selected items: {pomo_solution['selected_items']}")
        print(f"  Total weight: {pomo_solution['total_weight']:.4f}")
        print(f"  Total value: {pomo_solution['total_value']:.4f}")
        print(f"  Capacity utilization: {pomo_solution['capacity_utilization']:.4f}")
        print(f"  Solve time: {pomo_solution['solve_time']:.6f} seconds")
        print(f"  Algorithm: {pomo_solution['algorithm']}")
        print(f"  Optimal: {pomo_solution['optimal']}")
        print(f"  Trajectory type: {pomo_solution['trajectory_type']}")
    except Exception as e:
        print(f"  POMO solver failed: {e}")

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

    # Test DP batch solving
    print("\nDP batch solutions:")
    dp_batch_solutions = kp_solve_pool.solve_batch(batch_data, solver_name="dp")
    for i, sol in enumerate(dp_batch_solutions):
        print(f"  Instance {i + 1}: Value={sol['total_value']:.4f}, Time={sol['solve_time']:.6f}s")

    # Test POMO batch solving
    print("\nPOMO batch solutions:")
    try:
        pomo_batch_solutions = kp_solve_pool.solve_batch(batch_data, solver_name="pomo")
        for i, sol in enumerate(pomo_batch_solutions):
            print(f"  Instance {i + 1}: Value={sol['total_value']:.4f}, Time={sol['solve_time']:.6f}s")
    except Exception as e:
        print(f"  POMO batch solving failed: {e}")


if __name__ == '__main__':
    test_solver_pool()
