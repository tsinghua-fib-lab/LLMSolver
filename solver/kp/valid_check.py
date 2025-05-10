from solver_pool import kpSolverPool
import random
def validate_solution(instances: dict, solve_output: list) -> bool:
    """
    Verify the validity of the solution from the solve function.
    - The total weight of the selected items must be less than or equal to the knapsack capacity.
    - The indices of the selected items must be within the valid range.
    Parameters:
    - instances (dict): Contains problem instance data, including "item_weight", "item_value", "knapsack_capacity".
    - solve_output (list): The output of the solve function, contains "Maximum value", "Selected item indices", "Total weight".
    Returns:
    - bool: Returns True if the solution is valid, otherwise returns False.
    """
    selected_items = list(map(int, solve_output[1].split(":")[1].strip()[1:-1].split(", ")))  
    item_weights = instances["item_weight"]
    knapsack_capacity = instances["knapsack_capacity"]
    total_weight_calculated = sum(item_weights[i] for i in selected_items)
    # Capacity constraint 
    if total_weight_calculated > knapsack_capacity:
#        print("Invalid solution: total weight exceeds knapsack capacity.")
        return False
    # Validity of the index
    if any(i < 0 or i >= len(item_weights) for i in selected_items):
#        print("Invalid solution: invalid item indices.")
        return False
    return True

def test():
   data = {
    "item_weight": [round(random.uniform(0, 1), 4) for _ in range(50)],
    "item_value": [round(random.uniform(0, 1), 4) for _ in range(50)],
    "knapsack_capacity": 5
   }
   solver_pool = kpSolverPool()
   solution = solver_pool.solve(data)
   is_valid = validate_solution(data, solution)
   print("Is the solution valid?:", is_valid)
if __name__ == '__main__':
    test()