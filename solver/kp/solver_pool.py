from DPsolver import knapsack_Dynamic_Programming
import os
import random
#random.seed(20)
class kpSolverPool:
 def solve(self,instances: dict, solver_name: str = "genetic", problem_type: str = "hfssp", **kwargs):
   value, selected, total_weight = knapsack_Dynamic_Programming(instances["item_weight"], instances["item_value"], instances["knapsack_capacity"])
   return [f"Maximum value: {value}", 
        f"Selected item indices: {selected}", 
        f"Total weight: {total_weight}"]

def test():
#Test example
    data= {
                "item_weight":[round(random.uniform(0, 1), 4) for _ in range(20)],
                "item_value": [round(random.uniform(0, 1), 4) for _ in range(20)],
                "knapsack_capacity": 5
                        }   
#solve
    kp_solve_pool = kpSolverPool()
    solution=kp_solve_pool.solve(data, solver_name="dp", problem_type="kp")
    print(solution)

    
if __name__ == '__main__':
    test()