import random
from enum import Enum
from typing import Dict, List, Tuple, Union


class BPPProblemType(Enum):
    """Bin Packing Problem types"""
    BPP_2D_OFFLINE = '2DOFBPP'  # 2D Offline Bin Packing Problem
    BPP_2D_OFFLINE_ROTATABLE = '2DOFBPPR'  # 2D Offline Rotatable Bin Packing Problem
    BPP_2D_ONLINE = '2DONBPP'  # 2D Online Bin Packing Problem
    BPP_2D_ONLINE_ROTATABLE = '2DONBPPR'  # 2D Online Rotatable Bin Packing Problem
    BPP_3D_OFFLINE = '3DOFBPP'  # 3D Offline Bin Packing Problem
    BPP_3D_OFFLINE_ROTATABLE = '3DOFBPPR'  # 3D Offline Rotatable Bin Packing Problem
    BPP_3D_ONLINE = '3DONBPP'  # 3D Online Bin Packing Problem
    BPP_3D_ONLINE_ROTATABLE = '3DONBPPR'  # 3D Online Rotatable Bin Packing Problem


class BPPGenerator:
    def __init__(self,
                 problem_type: str,
                 min_items: int = 10,
                 max_items: int = 30,
                 min_bin_size: int = 8,
                 max_bin_size: int = 15,
                 min_item_size: int = 1,
                 max_item_size: int = 8):
        """
        Initialize BPP Generator
        
        Args:
            problem_type: One of the BPP problem types
            min_items: Minimum number of items to generate
            max_items: Maximum number of items to generate
            min_bin_size: Minimum bin dimension
            max_bin_size: Maximum bin dimension
            min_item_size: Minimum item dimension
            max_item_size: Maximum item dimension
        """
        if problem_type == '2DOFBPP':
            self.problem_type = BPPProblemType.BPP_2D_OFFLINE
        elif problem_type == '2DOFBPPR':
            self.problem_type = BPPProblemType.BPP_2D_OFFLINE_ROTATABLE
        elif problem_type == '2DONBPP':
            self.problem_type = BPPProblemType.BPP_2D_ONLINE
        elif problem_type == '2DONBPPR':
            self.problem_type = BPPProblemType.BPP_2D_ONLINE_ROTATABLE
        elif problem_type == '3DOFBPP':
            self.problem_type = BPPProblemType.BPP_3D_OFFLINE
        elif problem_type == '3DOFBPPR':
            self.problem_type = BPPProblemType.BPP_3D_OFFLINE_ROTATABLE
        elif problem_type == '3DONBPP':
            self.problem_type = BPPProblemType.BPP_3D_ONLINE
        elif problem_type == '3DONBPPR':
            self.problem_type = BPPProblemType.BPP_3D_ONLINE_ROTATABLE
        else:
            raise NotImplementedError(f'{problem_type} is not implemented')

        self.min_items = min_items
        self.max_items = max_items
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size
        self.min_item_size = min_item_size
        self.max_item_size = max_item_size

    def _is_2d(self) -> bool:
        """Check if problem is 2D"""
        return self.problem_type in [
            BPPProblemType.BPP_2D_OFFLINE,
            BPPProblemType.BPP_2D_OFFLINE_ROTATABLE,
            BPPProblemType.BPP_2D_ONLINE,
            BPPProblemType.BPP_2D_ONLINE_ROTATABLE
        ]

    def _is_online(self) -> bool:
        """Check if problem is online"""
        return self.problem_type in [
            BPPProblemType.BPP_2D_ONLINE,
            BPPProblemType.BPP_2D_ONLINE_ROTATABLE,
            BPPProblemType.BPP_3D_ONLINE,
            BPPProblemType.BPP_3D_ONLINE_ROTATABLE
        ]

    def _is_rotatable(self) -> bool:
        """Check if items can be rotated"""
        return self.problem_type in [
            BPPProblemType.BPP_2D_OFFLINE_ROTATABLE,
            BPPProblemType.BPP_2D_ONLINE_ROTATABLE,
            BPPProblemType.BPP_3D_OFFLINE_ROTATABLE,
            BPPProblemType.BPP_3D_ONLINE_ROTATABLE
        ]

    def generate_bin_size(self) -> List[int]:
        """Generate bin dimensions"""
        if self._is_2d():
            return [
                random.randint(self.min_bin_size, self.max_bin_size),
                random.randint(self.min_bin_size, self.max_bin_size)
            ]
        else:
            return [
                random.randint(self.min_bin_size, self.max_bin_size),
                random.randint(self.min_bin_size, self.max_bin_size),
                random.randint(self.min_bin_size, self.max_bin_size)
            ]

    def generate_item_sizes(self, num_items: int) -> List[List[int]]:
        """Generate item dimensions"""
        items = []
        for _ in range(num_items):
            if self._is_2d():
                item = [
                    random.randint(self.min_item_size, self.max_item_size),
                    random.randint(self.min_item_size, self.max_item_size)
                ]
            else:
                item = [
                    random.randint(self.min_item_size, self.max_item_size),
                    random.randint(self.min_item_size, self.max_item_size),
                    random.randint(self.min_item_size, self.max_item_size)
                ]
            items.append(item)
        return items

    def generate_problem_instance(self, **params) -> Dict:
        """Generate a complete BPP problem instance"""
        num_items = params.get('num_items', random.randint(self.min_items, self.max_items))
        bin_size = params.get('bin_size', self.generate_bin_size())
        items_size = params.get('items_size', self.generate_item_sizes(num_items))

        # Ensure items can fit in bin (at least one dimension should be smaller than bin)
        if self._is_2d():
            items_size = [
                [min(item[0], bin_size[0]), min(item[1], bin_size[1])]
                for item in items_size
            ]
        else:
            items_size = [
                [min(item[0], bin_size[0]), min(item[1], bin_size[1]), min(item[2], bin_size[2])]
                for item in items_size
            ]

        data = {
            "bin_size": bin_size,
            "items_size": items_size,
            "bin_status": self._is_online(),
            "can_rotate": self._is_rotatable(),
            "dimension": "2D" if self._is_2d() else "3D",
            "label": self.problem_type.value
        }

        return data

    def generate(self, batch_size: int, **params) -> List[Dict]:
        """Generate multiple BPP problem instances"""
        instances = []
        for _ in range(batch_size):
            instance = self.generate_problem_instance(**params)
            instances.append(instance)
        return instances


def test_generate_bpp():
    """Test function for BPP generator"""
    # Test all problem types
    problem_types = [
        '2DOFBPP', '2DOFBPPR', '2DONBPP', '2DONBPPR',
        '3DOFBPP', '3DOFBPPR', '3DONBPP', '3DONBPPR'
    ]
    
    for problem_type in problem_types:
        print(f"\nTesting {problem_type}:")
        generator = BPPGenerator(problem_type=problem_type)
        data_list = generator.generate(1)
        print(f"Generated instance: {data_list[0]}")


if __name__ == '__main__':
    test_generate_bpp()
