from solver_pool import bppSolverPool
import random
from typing import List
from genetic import Box, Bin
def validate_solution(instances: dict, result: List[str]) -> bool:
    from copy import deepcopy

    bin_size = instances["bin_size"]
    box_sizes = instances["items_size"]
    can_rotate = instances["can_rotate"] == "true"

    # 构造所有 Box 对象
    boxes = [Box(i, w, h, d, can_rotate=can_rotate) for i, (w, h, d) in enumerate(box_sizes)]
    id_to_box = {box.id: box for box in boxes}

    bins = []
    for line in result:
        if not line.startswith("Bin"):
            continue
        try:
            box_ids = eval(line.split(":")[1].strip())
        except Exception as e:
            print(f"解析失败: '{line}' -> {e}")
            return False

        b = Bin(*bin_size)
        for box_id in box_ids:
            original_box = id_to_box.get(box_id)
            if not original_box:
                print(f"无效的 box id: {box_id}")
                return False

            # 注意：要 deepcopy 否则后续 rotate 会互相影响
            box = deepcopy(original_box)
            if not b.place_box(box):
                print(f"Box {box_id} 放置失败")
                return False
        bins.append(b)

    # 对每个 bin 做检查
    for b in bins:
        for i, pb1 in enumerate(b.placed_boxes):
            # 边界检查
            if pb1.x + pb1.w > b.width or pb1.y + pb1.h > b.height or pb1.z + pb1.d > b.depth:
                print(f"Box {pb1.box.id} 超出边界")
                return False
            # 重叠检查
            for j, pb2 in enumerate(b.placed_boxes):
                if i == j:
                    continue
                if not (
                    pb1.x + pb1.w <= pb2.x or pb1.x >= pb2.x + pb2.w or
                    pb1.y + pb1.h <= pb2.y or pb1.y >= pb2.y + pb2.h or
                    pb1.z + pb1.d <= pb2.z or pb1.z >= pb2.z + pb2.d
                ):
                    print(f"Box {pb1.box.id} 与 Box {pb2.box.id} 重叠")
                    return False
            # 重力支撑检查
            if pb1.z != 0:
                supported = False
                for pb2 in b.placed_boxes:
                    if (
                        pb2.x < pb1.x + pb1.w and pb2.x + pb2.w > pb1.x and
                        pb2.y < pb1.y + pb1.h and pb2.y + pb2.h > pb1.y and
                        pb2.z + pb2.d == pb1.z
                    ):
                        supported = True
                        break
                if not supported:
                    print(f"Box {pb1.box.id} 缺乏支撑")
                    return False

    return True
  
def test():
    data = {
            "bin_size":[10,10],
            "items_size": [[random.randint(1, 10) for _ in range(3)] for _ in range(20)],
            "bin_status":"false",
            "can_rotate":"false",
            "dimension":"2D"
        } 
    solver = bppSolverPool() 
    result = solver.solve(instances=data, solver_name="genetic", problem_type="000")  # 调用你提供的 solve 函数
    is_valid = validate_solution(data, result)
    print("是否为可行解:", is_valid)

if __name__ == '__main__':
    test()