import random
from typing import List
from genetic import Box, Bin


def validate_solution(instances: dict, result: List[str]) -> bool:
    from copy import deepcopy

    bin_size = instances["bin_size"]
    box_sizes = instances["items_size"]
    can_rotate = instances["can_rotate"] == "true"
    boxes = [Box(i, w, h, d, can_rotate=can_rotate) for i, (w, h, d) in enumerate(box_sizes)]
    id_to_box = {box.id: box for box in boxes}

    bins = []
    for line in result:
        if not line.startswith("Bin"):
            continue
        try:
            box_ids = eval(line.split(":")[1].strip())
        except Exception as e:
            return False

        b = Bin(*bin_size)
        for box_id in box_ids:
            original_box = id_to_box.get(box_id)
            if not original_box:
                return False

            box = deepcopy(original_box)
            if not b.place_box(box):
                return False
        bins.append(b)

    # check
    for b in bins:
        for i, pb1 in enumerate(b.placed_boxes):
            # Boundary check
            if pb1.x + pb1.w > b.width or pb1.y + pb1.h > b.height or pb1.z + pb1.d > b.depth:
                print(f"Box {pb1.box.id} 超出边界")
                return False
            # Overlap check
            for j, pb2 in enumerate(b.placed_boxes):
                if i == j:
                    continue
                if not (
                        pb1.x + pb1.w <= pb2.x or pb1.x >= pb2.x + pb2.w or
                        pb1.y + pb1.h <= pb2.y or pb1.y >= pb2.y + pb2.h or
                        pb1.z + pb1.d <= pb2.z or pb1.z >= pb2.z + pb2.d
                ):
                    return False
            # Gravitational support check
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
                    return False

    return True
