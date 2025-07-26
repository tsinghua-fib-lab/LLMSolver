#!/usr/bin/env python3
"""
Genetic Algorithm Solver for BPP
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import random
import itertools
from typing import List, Tuple, Dict
from dataclasses import dataclass
from solver.bpp.genetic import Box, PlacedBox, Bin, genetic_algorithm


class GASolver:
    """Genetic Algorithm Solver for BPP"""

    def __init__(self, problem_type: str = "2DOFBPP", solver_name: str = "GA", **kwargs):
        """
        Initialize GA Solver
        
        Args:
            problem_type: Type of BPP problem (2DOFBPP, 2DOFBPPR, etc.)
            solver_name: Name of the solver
            **kwargs: Additional parameters for GA
        """
        self.problem_type = problem_type
        self.solver_name = solver_name
        self.population_size = kwargs.get('population_size', 30)
        self.generations = kwargs.get('generations', 100)
        self.mutation_rate = kwargs.get('mutation_rate', 0.2)

        # Extract problem characteristics from problem_type
        self.is_2d = problem_type.startswith('2D')
        self.is_online = 'ON' in problem_type
        self.can_rotate = 'R' in problem_type

    def solve(self, instances: Dict) -> Dict:
        """
        Solve BPP using genetic algorithm
        
        Args:
            instances: Problem instance data
            
        Returns:
            Solution in dict format
        """
        # Extract problem parameters
        bin_size = instances['bin_size']
        items_size = instances['items_size']

        # Handle 2D to 3D conversion
        if instances.get('dimension') == '2D':
            bin_size_3d = [bin_size[0], bin_size[1], 1]  # [width, height, 1]
            items_size_3d = [[item[0], item[1], 1] for item in items_size]  # [width, height, 1]
        else:
            bin_size_3d = bin_size
            items_size_3d = items_size

        # Convert boolean parameters
        is_online = 1 if instances.get('bin_status') == 'true' else 0
        can_rotate = 1 if instances.get('can_rotate') == 'true' else 0

        # Run genetic algorithm
        bins, best_individual = genetic_algorithm(
            box_sizes=items_size_3d,
            bin_size=bin_size_3d,
            is_online=is_online,
            can_rotate=can_rotate,
            population_size=self.population_size,
            generations=self.generations,
            mutation_rate=self.mutation_rate
        )

        # Convert to environment format
        solution = {
            "bins": []
        }

        is_2d = instances.get('dimension') == '2D'

        for bin_obj in bins:
            # Use appropriate format based on problem dimension
            if is_2d:
                bin_data = {
                    "bin_size": [bin_obj.width, bin_obj.height],  # [width, height] for 2D
                    "items": []
                }
            else:
                bin_data = {
                    "bin_size": [bin_obj.width, bin_obj.height, bin_obj.depth],
                    "items": []
                }

            for placed_box in bin_obj.placed_boxes:
                # Use appropriate format based on problem dimension
                if is_2d:
                    item_data = {
                        "item_id": placed_box.box.id,
                        "position": [placed_box.x, placed_box.y],  # [x, y] for 2D
                        "size": [placed_box.w, placed_box.h]  # [width, height] for 2D
                    }
                else:
                    item_data = {
                        "item_id": placed_box.box.id,
                        "position": [placed_box.x, placed_box.y, placed_box.z],
                        "size": [placed_box.w, placed_box.h, placed_box.d]
                    }
                bin_data["items"].append(item_data)

            solution["bins"].append(bin_data)

        return solution

    def get_solver_info(self) -> Dict:
        """Get solver information"""
        return {
            "solver_name": self.solver_name,
            "problem_type": self.problem_type,
            "population_size": self.population_size,
            "generations": self.generations,
            "mutation_rate": self.mutation_rate
        }


def test_ga_solver():
    """Test the GA solver"""
    # Test data
    instances = {
        "bin_size": [8, 8],
        "items_size": [
            [3, 4],
            [2, 5],
            [4, 2],
            [3, 3],
            [2, 2],
            [1, 6],
            [5, 1]
        ],
        "can_rotate": "true",
        "dimension": "2D",
        "bin_status": "false",
        "label": "2DOFBPP"
    }

    # Create solver
    solver = GASolver(
        problem_type="2DOFBPP",
        solver_name="GA",
        population_size=20,
        generations=50
    )

    # Solve
    solution = solver.solve(instances)

    print("Solution:")
    for i, bin_data in enumerate(solution['bins']):
        print(f"Bin {i + 1}: {len(bin_data['items'])} items")
        for item in bin_data['items']:
            print(f"  Item {item['item_id']}: pos{item['position']}, size{item['size']}")
    from envs.bpp.env import BPPEnv
    env = BPPEnv(problem_type="2DOFBPP")
    env.plot_packing_2d(instances, solution)
    print(env.is_valid(instances, solution))
    return solution


if __name__ == '__main__':
    test_ga_solver()
