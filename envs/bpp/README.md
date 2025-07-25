# Bin Packing Problem (BPP) Environment

This module provides a comprehensive environment for Bin Packing Problems (BPP) with support for both 2D and 3D variants, online and offline modes, and rotatable/non-rotatable items.

## Problem Types Supported

The BPP environment supports 8 different problem types:

| Problem Type | Description |
|--------------|-------------|
| `2DOFBPP` | 2D Offline Bin Packing Problem |
| `2DOFBPPR` | 2D Offline Rotatable Bin Packing Problem |
| `2DONBPP` | 2D Online Bin Packing Problem |
| `2DONBPPR` | 2D Online Rotatable Bin Packing Problem |
| `3DOFBPP` | 3D Offline Bin Packing Problem |
| `3DOFBPPR` | 3D Offline Rotatable Bin Packing Problem |
| `3DONBPP` | 3D Online Bin Packing Problem |
| `3DONBPPR` | 3D Online Rotatable Bin Packing Problem |

## Components

### 1. BPPGenerator

Generates problem instances for different BPP variants.

```python
from envs.bpp import BPPGenerator

# Create generator for 2D offline rotatable BPP
generator = BPPGenerator(
    problem_type="2DOFBPPR",
    min_items=10,
    max_items=30,
    min_bin_size=8,
    max_bin_size=15,
    min_item_size=1,
    max_item_size=8
)

# Generate problem instances
instances = generator.generate(batch_size=5)
```

### 2. BPPEnv

Environment for evaluating BPP solutions.

```python
from envs.bpp import BPPEnv

# Create environment
env = BPPEnv(problem_type="2DOFBPPR")

# Calculate reward (negative number of bins used)
reward = env.get_reward(problem_data, solution)

# Check if solution is valid
is_valid = env.is_valid(problem_data, solution)

# Calculate utilization
utilization = env.calculate_utilization(problem_data, solution)
```

### 3. Validation and Visualization

```python
from envs.bpp import validate_packing

# Validate solution
is_valid, message = validate_packing(problem_data, solution)

# Visualize 2D packing
env.plot_packing_2d(problem_data, solution, "My Packing Solution")

# Visualize 3D packing
env.plot_packing_3d(problem_data, solution, "My 3D Packing Solution")
```

## Data Format

### Problem Data Format

```python
problem_data = {
    "bin_size": [10, 10],  # [width, height] for 2D, [width, height, depth] for 3D
    "items_size": [
        [3, 4],    # [width, height] for 2D
        [2, 5],    # [width, height] for 2D
        # ... more items
    ],
    "bin_status": False,    # True for online, False for offline
    "can_rotate": True,     # True if items can be rotated
    "dimension": "2D",      # "2D" or "3D"
    "label": "2DOFBPPR"     # Problem type label
}
```

### Solution Data Format

```python
solution = {
    "bins": [
        {
            "bin_size": [10, 10],  # Must match problem_data["bin_size"]
            "items": [
                {
                    "item_id": 0,                    # Index of item in items_size
                    "position": [0, 0, 0],           # [x, y, z] placement position
                    "size": [3, 4, 1]                # [width, height, depth] after rotation
                },
                # ... more items
            ]
        },
        # ... more bins
    ]
}
```

## Features

### 1. Problem Generation
- **Flexible Parameters**: Customize item counts, bin sizes, and item sizes
- **All Variants**: Support for all 8 BPP problem types
- **Realistic Constraints**: Ensures items can fit within bins

### 2. Solution Validation
- **Overlap Detection**: Checks for item overlaps within bins
- **Boundary Checking**: Ensures items fit within bin boundaries
- **Rotation Validation**: Validates item rotations for rotatable problems
- **Completeness**: Ensures all items are placed exactly once

### 3. Reward Calculation
- **Minimization Objective**: Returns negative number of bins used
- **Invalid Penalty**: Returns negative infinity for invalid solutions
- **Utilization Metrics**: Calculates space utilization percentage

### 4. Visualization
- **2D Visualization**: Clear 2D packing diagrams with item labels
- **3D Visualization**: Multi-layer 3D packing visualization
- **Color Coding**: Different colors for different items
- **Bin Layout**: Shows multiple bins in organized layout

## Usage Examples

### Basic Usage

```python
from envs.bpp import BPPGenerator, BPPEnv

# Generate problem
generator = BPPGenerator("2DOFBPPR")
problem_data = generator.generate(1)[0]

# Create environment
env = BPPEnv("2DOFBPPR")

# Your solver would generate a solution here
solution = your_solver(problem_data)

# Evaluate solution
reward = env.get_reward(problem_data, solution)
is_valid = env.is_valid(problem_data, solution)
utilization = env.calculate_utilization(problem_data, solution)

print(f"Number of bins used: {-reward}")
print(f"Solution is valid: {is_valid}")
print(f"Space utilization: {utilization:.2%}")
```

### Advanced Usage

```python
# Generate multiple instances with custom parameters
generator = BPPGenerator(
    problem_type="3DOFBPPR",
    min_items=15,
    max_items=25,
    min_bin_size=10,
    max_bin_size=20,
    min_item_size=2,
    max_item_size=6
)

instances = generator.generate(batch_size=10)

# Evaluate multiple solutions
env = BPPEnv("3DOFBPPR")
results = []

for problem_data in instances:
    solution = your_solver(problem_data)
    reward = env.get_reward(problem_data, solution)
    is_valid = env.is_valid(problem_data, solution)
    utilization = env.calculate_utilization(problem_data, solution)
    results.append({
        'reward': reward,
        'is_valid': is_valid,
        'utilization': utilization,
        'num_bins': -reward
    })
```

## Testing

Run the test script to verify functionality:

```bash
python envs/bpp/test_bpp.py
```

This will test:
- Problem generation for all 8 variants
- Solution validation
- Solution validity checking
- Reward calculation
- Utilization metrics
- 3D problem handling

## Integration with Solvers

The BPP environment is designed to work seamlessly with the existing solver infrastructure:

```python
# Example integration with solver pool
from solver.bpp.GA.bpp_solver import solve_bpp
from envs.bpp import BPPGenerator, BPPEnv

# Generate problem
generator = BPPGenerator("2DOFBPPR")
problem_data = generator.generate(1)[0]

# Solve using existing solver
solution = solve_bpp(problem_data)

# Evaluate using environment
env = BPPEnv("2DOFBPPR")
reward = env.get_reward(problem_data, solution)
is_valid = env.is_valid(problem_data, solution)
```

## Performance Considerations

- **Validation**: O(n²) complexity for overlap checking
- **Visualization**: Efficient matplotlib-based rendering
- **Memory**: Minimal memory footprint for large problems
- **Scalability**: Supports problems with hundreds of items

## Future Enhancements

- **Gravity Constraints**: Support for gravity-based placement
- **Weight Constraints**: Support for weighted items
- **Fragile Items**: Support for fragile item constraints
- **Loading Patterns**: Support for specific loading patterns
- **Real-time Visualization**: Interactive 3D visualization