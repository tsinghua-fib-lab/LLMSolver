from copy import deepcopy
import numpy as np
from typing import Dict, List, Any, Optional


class BppSolverPool:
    """Solver pool for Bin Packing Problem (BPP)"""

    # Mapping of solvers to supported problem types
    SOLVER_PROBLEM_MAPPING = {
        'GA': ['2DOFBPP', '2DOFBPPR', '3DOFBPP', '3DOFBPPR'],
        'PCT': ['2DOFBPPR', '3DOFBPPR', '2DONBPPR', '3DONBPPR'],
        # Add more solvers here as they become available
    }

    def __init__(self, device: str = "cpu", **kwargs):
        """
        Initialize BPP Solver Pool.
        
        Args:
            device: Device to run solvers on (default: "cpu")
            **kwargs: Additional configuration parameters
        """
        self.problem_to_solver = {}
        self.solvers = {}
        self.device = device

        self._initialize_solvers()

    def _initialize_solvers(self):
        """Initialize all available solvers."""
        for solver_name, supported_problems in self.SOLVER_PROBLEM_MAPPING.items():
            try:
                self._initialize_solver(solver_name, supported_problems)
            except ImportError:
                print(f"⚠️  WARNING: {solver_name} solver not available")
            except Exception as e:
                print(f"❌ Error initializing {solver_name}: {e}")

        self._print_available_solvers()

    def _initialize_solver(self, solver_name: str, supported_problems: List[str]):
        """Initialize a specific solver for all supported problem types."""
        self.solvers[solver_name] = {}

        if solver_name in ['GA', 'genetic']:
            from solver.bpp.ga.genetic_solver import GASolver
            for problem_type in supported_problems:
                self.solvers[solver_name][problem_type] = GASolver(
                    problem_type=problem_type,
                    solver_name=solver_name
                )
        elif solver_name == "PCT":
            from solver.bpp.pct.pct_solver import PCTSolver
            for problem_type in supported_problems:
                self.solvers[solver_name][problem_type] = PCTSolver(
                    problem_type=problem_type,
                    solver_name=solver_name,
                    setting=1,
                    internal_node_holder=80,
                    leaf_node_holder=50
                )

        # Build problem to solver mapping
        for problem_type in supported_problems:
            if problem_type not in self.problem_to_solver:
                self.problem_to_solver[problem_type] = []
            self.problem_to_solver[problem_type].append(solver_name)

    def _get_pct_setting(self, problem_type: str) -> int:
        """Get appropriate PCT setting based on problem type."""
        if '2D' in problem_type:
            return 2
        elif '3D' in problem_type:
            return 2  # Use setting 2 for 3D as well
        return 2  # Default setting

    def _print_available_solvers(self):
        """Print available solvers for each problem type."""
        print("📦 Available BPP Solvers:")
        for problem_type, solvers in self.problem_to_solver.items():
            print(f"  {problem_type}: {solvers}")

    def get_problem_list(self) -> List[str]:
        """Get list of available problem types."""
        return list(self.problem_to_solver.keys())

    def get_solver_list(self, problem_type: str) -> List[str]:
        """Get list of available solvers for a specific problem type."""
        return self.problem_to_solver.get(problem_type, [])

    def _create_fallback_solution(self, instances: Dict) -> Dict:
        """Create a simple fallback solution when solver fails."""
        bin_size = instances['bin_size']
        items_size = instances['items_size']

        solution = {'bins': []}
        current_bin = {'bin_size': bin_size, 'items': []}

        for i, item_size in enumerate(items_size):
            position = [0, 0] if len(bin_size) == 2 else [0, 0, 0]
            current_bin['items'].append({
                'item_id': i,
                'position': position,
                'size': item_size
            })

        if current_bin['items']:
            solution['bins'].append(current_bin)

        return solution

    def _solve(
            self,
            instances: Dict,
            solver_name: str = "GA",
            problem_type: str = "2DOFBPP",
            max_runtime: float = 30,
            num_procs: int = 1,
            **kwargs
    ) -> Dict:
        """
        Solve BPP instances with specified solver.

        Args:
            instances: Dictionary containing the BPP instance data
            solver_name: The solver to use
            problem_type: Type of BPP problem
            max_runtime: Maximum runtime for the solver
            num_procs: Number of processors to use
            **kwargs: Additional solver-specific parameters

        Returns:
            Solution in dict format

        Raises:
            ValueError: If solver or problem type is not supported
        """
        if solver_name not in self.solvers:
            raise ValueError(f"❌ Unknown solver: {solver_name}")

        if problem_type not in self.solvers[solver_name]:
            raise ValueError(f"❌ Solver {solver_name} does not support problem type {problem_type}")

        solver = self.solvers[solver_name][problem_type]

        # Update solver parameters if provided
        for key, value in kwargs.items():
            if hasattr(solver, key):
                setattr(solver, key, value)

        # Solve the problem
        solution = solver.solve(instances=instances)

        # Handle case where solver returns None
        if solution is None:
            print(f"⚠️  Solver {solver_name} failed, using fallback solution")
            solution = self._create_fallback_solution(instances)

        return solution

    def solve(
            self,
            instances: Dict,
            solver_name: str = "GA",
            problem_type: str = "2DOFBPP",
            timeout: int = 30,
            num_procs: int = 1,
            **kwargs
    ) -> Dict:
        """
        Solve BPP problem with specified parameters.

        Args:
            instances: Problem instance data
            solver_name: Name of the solver to use
            problem_type: Type of BPP problem
            timeout: Maximum runtime in seconds
            num_procs: Number of processors to use
            **kwargs: Additional solver parameters

        Returns:
            Solution in dict format
        """
        return self._solve(
            instances=instances,
            solver_name=solver_name,
            problem_type=problem_type,
            max_runtime=timeout,
            num_procs=num_procs,
            **kwargs
        )

    def solve_batch(
            self,
            instances_list: List[Dict],
            solver_name: str = "GA",
            problem_type: str = "2DOFBPP",
            timeout: int = 30,
            num_procs: int = 1,
            **kwargs
    ) -> List[Dict]:
        """
        Solve multiple BPP problems.

        Args:
            instances_list: List of problem instances
            solver_name: Name of the solver to use
            problem_type: Type of BPP problem
            timeout: Maximum runtime in seconds
            num_procs: Number of processors to use
            **kwargs: Additional solver parameters

        Returns:
            List of solutions
        """
        solutions = []
        for i, instances in enumerate(instances_list):
            print(f"🔧 Solving instance {i + 1}/{len(instances_list)}")
            solution = self.solve(
                instances=instances,
                solver_name=solver_name,
                problem_type=problem_type,
                timeout=timeout,
                num_procs=num_procs,
                **kwargs
            )
            solutions.append(solution)
        return solutions


def test_bpp_solver_pool():
    """Test the BPP solver pool functionality."""
    import random

    # Create test instance
    instances = {
        "bin_size": [10, 10],
        "items_size": [[random.randint(1, 5) for _ in range(2)] for _ in range(10)],
        "bin_status": False,
        "can_rotate": True,
        "dimension": "2D",
        "label": "2DOFBPP"
    }

    print("🧪 Testing BPP Solver Pool")
    print("=" * 50)

    # Create solver pool
    solver_pool = BppSolverPool()

    # Test available problems and solvers
    print(f"\n📋 Available problems: {solver_pool.get_problem_list()}")
    print(f"🔧 Available solvers for 2DOFBPP: {solver_pool.get_solver_list('2DOFBPP')}")

    # Test solving with GA
    print(f"\n🔧 Testing GA solver...")
    ga_solution = solver_pool.solve(
        instances=instances,
        solver_name="GA",
        problem_type="2DOFBPP",
        population_size=20,
        generations=50
    )

    # Print GA results
    print(f"\n✅ GA Solution found with {len(ga_solution['bins'])} bins:")
    for i, bin_data in enumerate(ga_solution['bins']):
        print(f"  Bin {i + 1}: {len(bin_data['items'])} items")
        for item in bin_data['items']:
            print(f"    Item {item['item_id']}: pos{item['position']}, size{item['size']}")

    # Test solving with PCT
    print(f"\n🔧 Testing PCT solver...")
    pct_solution = solver_pool.solve(
        instances=instances,
        solver_name="PCT",
        problem_type="2DOFBPPR"
    )

    # Print PCT results
    print(f"\n✅ PCT Solution found with {len(pct_solution['bins'])} bins:")
    for i, bin_data in enumerate(pct_solution['bins']):
        print(f"  Bin {i + 1}: {len(bin_data['items'])} items")
        for item in bin_data['items']:
            print(f"    Item {item['item_id']}: pos{item['position']}, size{item['size']}")

    # Validate solutions
    from envs.bpp.env import BPPEnv
    env = BPPEnv(problem_type="2DOFBPPR")

    ga_valid = env.is_valid(instances, ga_solution)
    pct_valid = env.is_valid(instances, pct_solution)

    print(f"\n🔍 Solution validation:")
    print(f"  GA solver: {ga_valid}")
    print(f"  PCT solver: {pct_valid}")

    # Compare solutions
    if ga_valid and pct_valid:
        ga_bins = len(ga_solution['bins'])
        pct_bins = len(pct_solution['bins'])
        print(f"\n📊 Solution comparison:")
        print(f"  GA solver used {ga_bins} bins")
        print(f"  PCT solver used {pct_bins} bins")
        if ga_bins < pct_bins:
            print(f"  🏆 GA solver performed better (fewer bins)")
        elif pct_bins < ga_bins:
            print(f"  🏆 PCT solver performed better (fewer bins)")
        else:
            print(f"  🤝 Both solvers used the same number of bins")

    # Plot solutions if valid
    if ga_valid:
        env.plot_packing_2d(instances, ga_solution, "GA Solver Result")

    if pct_valid:
        env.plot_packing_2d(instances, pct_solution, "PCT Solver Result")

    return ga_solution, pct_solution


if __name__ == '__main__':
    test_bpp_solver_pool()
