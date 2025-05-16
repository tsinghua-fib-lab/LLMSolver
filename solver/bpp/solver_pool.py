from .genetic import genetic_algorithm
import os
import random

class bppSolverPool:
 def solve(self,instances: dict, solver_name: str = "genetic", problem_type: str = "hfssp", **kwargs):
     if instances["dimension"]=="2D":
              instances["bin_size"]=[10,1,10]
              instances["items_size"]=[[row[0], 1, row[1]] for row in instances["items_size"]]
     elif instances["dimension"] not in ["2D","3D"]:
                raise ValueError("Dimension error")
     if instances["bin_status"]=="false":
            is_online=0
     elif instances["bin_status"]=="true": 
            is_online=1
     else:
            raise ValueError("Cannot determine whether the mode is online or offline")
     if instances["can_rotate"]=="false":
            can_rotate=0
     elif instances["can_rotate"]=="true": 
            can_rotate=1
     else:
            raise ValueError("Cannot determine whether the item is rotatable")
     bins, best_individual = genetic_algorithm(instances["items_size"], instances["bin_size"], is_online, can_rotate)
     result = []
     for i, b in enumerate(bins):
        box_ids = [pb.box.id for pb in b.placed_boxes]
        result.append(f"Bin {i+1} contains: {box_ids}")
     return result
     
def test():
#Test example
    data = {
            "bin_size":[10,10],
            "items_size": [[random.randint(1, 10) for _ in range(3)] for _ in range(20)],
            "bin_status":"true",
            "can_rotate":"false",
            "dimension":"2D"
        }    
    bpp_solve_pool = bppSolverPool()
    solution=bpp_solve_pool.solve(instances=data, solver_name="genetic", problem_type="2dofbpp")
    print(solution)

    
if __name__ == '__main__':
    test()
'''
try:
        bins, best_individual = genetic_algorithm(box_sizes, bin_size, is_online, can_rotate)  #implement
        print("Problem solved successfully!Number of bins used:",len(bins))
        print("bins:",bins)
        print("best_individual:",best_individual)
except Exception:
        print(False) 
print(box_sizes)
print("使用的 bin 数量：", len(bins))
for i, b in enumerate(bins):
    print(f"Bin {i+1} 放了 {len(b.placed_boxes)} 个 box")
    for pb in b.placed_boxes:
        print(f"  Box{pb.box.id} at ({pb.x}, {pb.y}, {pb.z}) size ({pb.w}, {pb.h}, {pb.d})")
print("最佳放置顺序：", best_individual)
print(type(data))
print(data)



  dimension="2D"       # input:"2D" or "3D"
  is_online=0          # input:0 or 1
  can_rotate=0         # inut:1 or 0
  bin_size =[10,10]           # input:bin_size--[10,10] or [10, 10,10]
  box_sizes= [[random.randint(1, 10) for _ in range(3)] for _ in range(20)]          # input:item_size--n×2  or  n×3
'''