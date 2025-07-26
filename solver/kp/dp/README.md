# Dynamic Programming Solver for Knapsack Problems

This directory contains a complete implementation of a Dynamic Programming (DP) solver for the 0-1 Knapsack Problem.

## Overview

The DP solver provides an optimal solution to the 0-1 Knapsack Problem using dynamic programming. It handles floating-point weights and values, making it suitable for real-world applications where exact integer weights are not required.

## Files

- `dp_solver.py` - Main DP solver implementation
- `DPsolver.py` - Legacy DP implementation (for reference)
- `__init__.py` - Package initialization
- `README.md` - This documentation file

## Features

### Core Algorithm (`knapsack_dp`)

The main DP algorithm with the following features:

- **Floating-point support**: Handles decimal weights and values
- **Precision control**: Configurable decimal precision for floating-point handling
- **Optimal solutions**: Guarantees optimal (best possible) solutions
- **Efficient implementation**: Uses dynamic programming with O(n*capacity) time complexity

### DPSolver Class

A comprehensive solver class that provides:

- **Single instance solving**: `solve_single()` method
- **Batch solving**: `solve()` method for multiple instances
- **Solution formatting**: Structured output with detailed information
- **Performance tracking**: Includes solve time measurements
- **Validation**: Input validation and error handling

### Key Methods

#### `knapsack_dp(weights, values, capacity, precision=4)`
Core DP algorithm that returns:
- `max_value`: Maximum achievable value
- `selected_items`: List of selected item indices
- `total_weight`: Total weight of selected items

#### `DPSolver.solve_single(instance, **params)`
Solves a single knapsack instance and returns a detailed solution dictionary.

#### `DPSolver.solve(instances, **params)`
Solves multiple instances and returns a list of solutions.

## Usage Examples

### Basic Usage

```python
from solver.kp.dp.dp_solver import DPSolver

# Create solver
solver = DPSolver(problem_type="knapsack")

# Define problem instance
instance = {
    'item_weight': [0.5, 0.3, 0.8, 0.2, 0.6],
    'item_value': [0.8, 0.5, 1.2, 0.3, 0.9],
    'knapsack_capacity': 1.0
}

# Solve
solution = solver.solve_single(instance)
print(f"Selected items: {solution['selected_items']}")
print(f"Total value: {solution['total_value']}")
print(f"Total weight: {solution['total_weight']}")
```

### Using the Solver Pool

```python
from solver.kp.solver_pool import kpSolverPool

# Create solver pool
pool = kpSolverPool()

# Solve using DP
solution = pool.solve(instance, solver_name="dp")
```

### Batch Solving

```python
# Multiple instances
instances = [
    {
        'item_weight': [0.5, 0.3, 0.8],
        'item_value': [0.8, 0.5, 1.2],
        'knapsack_capacity': 1.0
    },
    {
        'item_weight': [0.2, 0.4, 0.6],
        'item_value': [0.3, 0.7, 1.0],
        'knapsack_capacity': 0.8
    }
]

# Solve all instances
solutions = solver.solve_batch(instances)
```

## Algorithm Details

### Dynamic Programming Approach

The algorithm uses a 2D DP table where:
- `dp[i][j]` = maximum value achievable with first `i` items and capacity `j`
- `keep[i][j]` = whether item `i-1` is included in the optimal solution

### Floating-Point Handling

For floating-point weights, the algorithm:
1. Scales weights and capacity by a precision factor (default: 10^4)
2. Converts to integers for DP computation
3. Scales results back to original precision

### Time and Space Complexity

- **Time Complexity**: O(n × capacity × precision)
- **Space Complexity**: O(n × capacity × precision)

Where:
- `n` = number of items
- `capacity` = knapsack capacity
- `precision` = decimal precision factor

## Solution Format

The solver returns solutions in the following format:

```python
{
    'selected_items': [0, 2, 4],  # Indices of selected items
    'total_weight': 1.1,          # Total weight of selected items
    'total_value': 2.9,           # Total value of selected items
    'capacity_utilization': 0.73, # Fraction of capacity used
    'solve_time': 0.001234,       # Time taken to solve (seconds)
    'algorithm': 'dynamic_programming',
    'optimal': True               # Whether solution is optimal
}
```

## Integration with Environment

The DP solver is fully compatible with the KP environment:

```python
from envs.kp.env import KPEnv
from solver.kp.dp.dp_solver import DPSolver

env = KPEnv(problem_type="knapsack")
solver = DPSolver(problem_type="knapsack")

# Solve and validate
solution = solver.solve_single(instance)
is_valid = env.is_valid(instance, solution)
reward = env.get_reward(instance, solution)
```

## Testing

Run the comprehensive test suite:

```bash
python test_dp_solver.py
```

Or test individual components:

```bash
# Test DP solver directly
python solver/kp/dp/dp_solver.py

# Test solver pool
python solver/kp/solver_pool.py
```

## Performance Considerations

### Memory Usage
For large instances, consider:
- Reducing precision for floating-point handling
- Using approximate algorithms for very large problems
- Implementing memory-efficient DP variants

### Time Complexity
The DP algorithm is optimal but may be slow for:
- Very large capacities
- High precision requirements
- Many items (>1000)

For such cases, consider using:
- Greedy algorithms (faster, approximate)
- Genetic algorithms (heuristic, good quality)
- Other metaheuristics

## Limitations

1. **Memory intensive**: Requires O(n × capacity) memory
2. **Integer scaling**: Floating-point weights are scaled to integers
3. **Precision trade-off**: Higher precision increases memory and time usage
4. **Large instances**: May be slow for very large problems

## Future Enhancements

Potential improvements:
- Memory-efficient DP implementation
- Parallel processing for batch solving
- Integration with other optimization algorithms
- Support for additional knapsack variants (unbounded, multiple knapsacks) 