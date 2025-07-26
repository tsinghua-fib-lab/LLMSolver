import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
import itertools
from dataclasses import dataclass


@dataclass
class Box:
    id: int
    width: int
    height: int
    depth: int
    can_rotate: bool = True

    def get_orientations(self) -> List[Tuple[int, int, int]]:
        dims = [self.width, self.height, self.depth]
        if not self.can_rotate:
            return [tuple(dims)]
        orientations = list(set(itertools.permutations(dims, 3)))
        return orientations


@dataclass
class PlacedBox:
    box: Box
    x: int
    y: int
    z: int
    w: int
    h: int
    d: int


class Bin:
    """Represents a bin with placed items"""

    def __init__(self, width: int, height: int, depth: int = 1):
        self.width = width
        self.height = height
        self.depth = depth
        self.placed_items = []  # List of (x, y, z, w, h, d, item_id)
        self.placed_boxes: List[PlacedBox] = []  # For solver compatibility

    def can_place_item(self, x: int, y: int, z: int, w: int, h: int, d: int) -> bool:
        """Check if an item can be placed at the given position"""
        # Check boundaries
        if x + w > self.width or y + h > self.height or z + d > self.depth:
            return False

        # Check overlap with existing items
        for px, py, pz, pw, ph, pd, _ in self.placed_items:
            if not (x + w <= px or x >= px + pw or
                    y + h <= py or y >= py + ph or
                    z + d <= pz or z >= pz + pd):
                return False

        return True

    def place_item(self, x: int, y: int, z: int, w: int, h: int, d: int, item_id: int) -> bool:
        """Place an item in the bin"""
        if self.can_place_item(x, y, z, w, h, d):
            self.placed_items.append((x, y, z, w, h, d, item_id))
            return True
        return False

    def get_utilization(self) -> float:
        """Calculate space utilization"""
        total_volume = self.width * self.height * self.depth
        used_volume = sum(w * h * d for _, _, _, w, h, d, _ in self.placed_items)
        return used_volume / total_volume if total_volume > 0 else 0

    def fits(self, w, h, d, x, y, z) -> bool:
        """Check if a box can be placed at the given position (for solver compatibility)"""
        # Boundary checking
        if x + w > self.width or y + h > self.height or z + d > self.depth:
            return False

        # Overlap checking
        for pb in self.placed_boxes:
            if not (x + w <= pb.x or x >= pb.x + pb.w or
                    y + h <= pb.y or y >= pb.y + pb.h or
                    z + d <= pb.z or z >= pb.z + pb.d):
                return False

        # Gravitational support checking
        if z == 0:
            supported = True  # Placed directly on the ground
        else:
            supported = False
            for pb in self.placed_boxes:
                # Supported from below
                if (pb.x < x + w and pb.x + pb.w > x and
                        pb.y < y + h and pb.y + pb.h > y and
                        pb.z + pb.d == z):
                    supported = True
                    break
            if not supported:
                return False

        return True

    def get_candidate_positions(self) -> List[Tuple[int, int, int]]:
        """Get candidate positions for placing new boxes (for solver compatibility)"""
        positions = []

        sorted_placed_boxes = self.placed_boxes

        for pb in sorted_placed_boxes:
            if pb.z == 0:
                positions.append((pb.x + pb.w, pb.y, pb.z))
                positions.append((pb.x, pb.y + pb.h, pb.z))
                positions.append((pb.x, pb.y, pb.z + pb.d))
            else:
                for potential_support in sorted_placed_boxes:
                    if (potential_support.x < pb.x + pb.w and potential_support.x + potential_support.w > pb.x and
                            potential_support.y < pb.y + pb.h and potential_support.y + potential_support.h > pb.y and
                            potential_support.z + potential_support.d == pb.z):
                        positions.append((pb.x + pb.w, pb.y, pb.z))
                        positions.append((pb.x, pb.y + pb.h, pb.z))
                        positions.append((pb.x, pb.y, pb.z + pb.d))
                        break
        if not positions:
            positions.append((0, 0, 0))
        positions.sort(key=lambda pos: (pos[2], pos[
            0]))  # First sort by ascending z-value; if z-values are equal, then sort by ascending x-value.
        return positions

    def place_box(self, box: Box) -> bool:
        """Try placing the box in all possible rotation orientations (for solver compatibility)"""
        for orientation in box.get_orientations():
            w, h, d = orientation
            for x, y, z in self.get_candidate_positions():
                if self.fits(w, h, d, x, y, z):
                    self.placed_boxes.append(PlacedBox(box, x, y, z, w, h, d))
                    # Also update the placed_items for environment compatibility
                    self.place_item(x, y, z, w, h, d, box.id)
                    return True
        return False


def validate_packing(problem_data: Dict, solution: Dict) -> Tuple[bool, str]:
    """Validate the packing solution"""
    bin_size = problem_data['bin_size']
    items_size = problem_data['items_size']
    can_rotate = problem_data.get('can_rotate', False)
    is_2d = problem_data.get('dimension') == '2D'

    # Check if all items are placed
    placed_items = set()
    bins = []

    for bin_data in solution.get('bins', []):
        # Handle 2D vs 3D bin size
        if is_2d:
            if len(bin_data['bin_size']) != 2:
                return False, f"2D problem expects 2D bin size, got {bin_data['bin_size']}"
            bin_width, bin_height = bin_data['bin_size']
            bin_depth = 1
        else:
            if len(bin_data['bin_size']) != 3:
                return False, f"3D problem expects 3D bin size, got {bin_data['bin_size']}"
            bin_width, bin_height, bin_depth = bin_data['bin_size']

        # Validate bin size matches problem
        if bin_width != bin_size[0] or bin_height != bin_size[1]:
            return False, f"Bin size mismatch: expected {bin_size}, got {bin_data['bin_size']}"
        if not is_2d and len(bin_size) > 2 and bin_depth != bin_size[2]:
            return False, f"Bin depth mismatch: expected {bin_size[2]}, got {bin_depth}"

        bin_obj = Bin(bin_width, bin_height, bin_depth)

        for item_data in bin_data.get('items', []):
            item_id = item_data['item_id']

            # Handle 2D vs 3D position and size
            if is_2d:
                if len(item_data['position']) != 2 or len(item_data['size']) != 2:
                    return False, f"2D problem expects 2D position and size for item {item_id}"
                x, y = item_data['position']
                z = 0  # 2D items are placed at z=0
                w, h = item_data['size']
                d = 1  # 2D items have depth=1
            else:
                if len(item_data['position']) != 3 or len(item_data['size']) != 3:
                    return False, f"3D problem expects 3D position and size for item {item_id}"
                x, y, z = item_data['position']
                w, h, d = item_data['size']

            # Check if item exists in problem
            if item_id >= len(items_size):
                return False, f"Invalid item_id: {item_id}"

            # Check if item already placed
            if item_id in placed_items:
                return False, f"Item {item_id} placed multiple times"

            # Validate item size
            original_size = items_size[item_id]
            if is_2d and len(original_size) == 2:
                # For 2D problems, compare only width and height
                if not can_rotate:
                    # Check if size matches exactly
                    if not (w == original_size[0] and h == original_size[1]):
                        return False, f"Item {item_id} size mismatch: expected {original_size}, got {[w, h]}"
                else:
                    # Check if size is a valid rotation (2D)
                    valid_rotations = [
                        original_size,
                        [original_size[1], original_size[0]]
                    ]
                    if [w, h] not in valid_rotations:
                        return False, f"Item {item_id} invalid rotation: {[w, h]}"
            else:
                # For 3D problems, handle 3D rotations
                if len(original_size) == 2:
                    original_size = original_size + [1]  # Add depth for 2D items in 3D problem

                if not can_rotate:
                    # Check if size matches exactly
                    if not (w == original_size[0] and h == original_size[1] and d == original_size[2]):
                        return False, f"Item {item_id} size mismatch: expected {original_size}, got {[w, h, d]}"
                else:
                    # Check if size is a valid rotation (3D)
                    valid_rotations = [
                        original_size,
                        [original_size[1], original_size[0], original_size[2]],
                        [original_size[0], original_size[2], original_size[1]],
                        [original_size[1], original_size[2], original_size[0]],
                        [original_size[2], original_size[0], original_size[1]],
                        [original_size[2], original_size[1], original_size[0]]
                    ]
                    if [w, h, d] not in valid_rotations:
                        return False, f"Item {item_id} invalid rotation: {[w, h, d]}"

            # Try to place item
            if not bin_obj.place_item(x, y, z, w, h, d, item_id):
                return False, f"Item {item_id} cannot be placed at position ({x}, {y}, {z})"

            placed_items.add(item_id)
        bins.append(bin_obj)

    # Check if all items are placed
    if len(placed_items) != len(items_size):
        return False, f"Not all items placed: {len(placed_items)}/{len(items_size)}"

    return True, "Packing is valid"


class BPPEnv:
    """Bin Packing Problem Environment"""

    def __init__(self, problem_type: str = "2DOFBPP"):
        """
        Initialize BPP Environment
        
        Args:
            problem_type: Type of BPP problem (2DOFBPP, 2DOFBPPR, etc.)
        """
        self.problem_type = problem_type
        self.is_2d = problem_type.startswith('2D')
        self.is_online = 'ON' in problem_type
        self.can_rotate = 'R' in problem_type

    def is_valid(self, problem_data: Dict, solution: Dict) -> bool:
        """
        Check if the given solution is valid
        
        Args:
            problem_data: Problem instance data
            solution: Solution data (Dict format)
            
        Returns:
            True if solution is valid, False otherwise
        """
        is_valid, _ = validate_packing(problem_data, solution)
        print(_)
        return is_valid

    def get_reward(self, problem_data: Dict, solution: Dict) -> float:
        """
        Calculate reward based on number of bins used (negative for minimization)
        
        Args:
            problem_data: Problem instance data
            solution: Solution data (Dict format)
            
        Returns:
            Reward value (negative number of bins used)
        """
        is_valid, message = validate_packing(problem_data, solution)
        if not is_valid:
            return float('-inf')  # Invalid solution gets worst reward

        num_bins = len(solution.get('bins', []))
        return -num_bins  # Negative because we want to minimize bins

    def calculate_utilization(self, problem_data: Dict, solution: Dict) -> float:
        """Calculate average bin utilization"""
        is_valid, _ = validate_packing(problem_data, solution)
        if not is_valid:
            return 0.0

        total_utilization = 0.0
        num_bins = len(solution.get('bins', []))
        is_2d = problem_data.get('dimension') == '2D'

        for bin_data in solution.get('bins', []):
            # Handle 2D vs 3D bin size
            if is_2d and len(bin_data['bin_size']) == 2:
                bin_width, bin_height = bin_data['bin_size']
                bin_depth = 1
            else:
                bin_width, bin_height = bin_data['bin_size'][:2]
                bin_depth = bin_data['bin_size'][2] if len(bin_data['bin_size']) > 2 else 1

            bin_obj = Bin(bin_width, bin_height, bin_depth)

            for item_data in bin_data.get('items', []):
                # Handle 2D vs 3D position and size
                if is_2d and len(item_data['position']) == 2:
                    x, y = item_data['position']
                    z = 0
                    w, h = item_data['size']
                    d = 1
                else:
                    x, y, z = item_data['position'][:3]
                    w, h, d = item_data['size'][:3]

                bin_obj.place_item(x, y, z, w, h, d, item_data['item_id'])

            total_utilization += bin_obj.get_utilization()

        return total_utilization / num_bins if num_bins > 0 else 0.0

    def plot_packing_2d(self, problem_data: Dict, solution: Dict, title: str = "2D Bin Packing"):
        """Plot 2D packing solution"""
        if not self.is_2d:
            raise ValueError("This method is only for 2D problems")

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.Set3(np.linspace(0, 1, len(problem_data['items_size'])))

        for bin_idx, bin_data in enumerate(solution.get('bins', [])):
            # Handle 2D bin size
            if len(bin_data['bin_size']) == 2:
                bin_width, bin_height = bin_data['bin_size']
            else:
                # Fallback for 3D format
                bin_width, bin_height = bin_data['bin_size'][:2]

            # Draw bin outline
            bin_rect = mpatches.Rectangle((0, bin_idx * (bin_height + 2)),
                                          bin_width, bin_height,
                                          linewidth=2, edgecolor='black',
                                          facecolor='none', alpha=0.7)
            ax.add_patch(bin_rect)

            # Draw items
            for item_data in bin_data.get('items', []):
                item_id = item_data['item_id']

                # Handle 2D vs 3D position and size
                if len(item_data['position']) == 2:
                    # 2D format: [x, y]
                    x, y = item_data['position']
                else:
                    # 3D format: [x, y, z] - use x, y
                    x, y = item_data['position'][:2]

                if len(item_data['size']) == 2:
                    # 2D format: [w, h]
                    w, h = item_data['size']
                else:
                    # 3D format: [w, h, d] - use w, h
                    w, h = item_data['size'][:2]

                # Adjust y position for bin offset
                y_adjusted = y + bin_idx * (bin_height + 2)

                item_rect = mpatches.Rectangle((x, y_adjusted), w, h,
                                               facecolor=colors[item_id % len(colors)],
                                               edgecolor='black', alpha=0.8)
                ax.add_patch(item_rect)

                # Add item label
                ax.text(x + w / 2, y_adjusted + h / 2, f'I{item_id}',
                        ha='center', va='center', fontsize=8, fontweight='bold')

            # Add bin label
            ax.text(bin_width / 2, bin_idx * (bin_height + 2) + bin_height / 2, f'Bin {bin_idx}',
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # Set plot limits
        max_width = max(bin_data['bin_size'][0] if len(bin_data['bin_size']) == 2
                        else bin_data['bin_size'][0] for bin_data in solution.get('bins', []))
        num_bins = len(solution.get('bins', []))
        bin_height = (solution['bins'][0]['bin_size'][1] if len(solution['bins'][0]['bin_size']) == 2
                      else solution['bins'][0]['bin_size'][1]) if solution.get('bins') else 0

        ax.set_xlim(-1, max_width + 1)
        ax.set_ylim(-1, num_bins * (bin_height + 2) + 1)
        ax.set_aspect('equal')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Add legend
        legend_elements = [mpatches.Patch(color=colors[i % len(colors)],
                                          label=f'Item {i}')
                           for i in range(len(problem_data['items_size']))]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

        plt.tight_layout()
        plt.show()

    def plot_packing_3d(self, problem_data: Dict, solution: Dict, title: str = "3D Bin Packing"):
        """Plot 3D packing solution (showing multiple layers)"""
        if self.is_2d:
            raise ValueError("This method is only for 3D problems")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        colors = plt.cm.Set3(np.linspace(0, 1, len(problem_data['items_size'])))

        bins_data = solution.get('bins', [])

        for bin_idx, bin_data in enumerate(bins_data):
            if bin_idx >= 4:  # Limit to 4 bins for visualization
                break

            bin_width, bin_height, bin_depth = bin_data['bin_size']
            items = bin_data.get('items', [])

            ax = axes[bin_idx]

            # Group items by z-layer
            layers = {}
            for item_data in items:
                item_id = item_data['item_id']
                x, y, z = item_data['position']
                w, h, d = item_data['size']

                if z not in layers:
                    layers[z] = []
                layers[z].append((x, y, w, h, item_id))

            # Draw each layer
            for layer_z, layer_items in sorted(layers.items()):
                for x, y, w, h, item_id in layer_items:
                    item_rect = mpatches.Rectangle((x, y), w, h,
                                                   facecolor=colors[item_id % len(colors)],
                                                   edgecolor='black', alpha=0.8)
                    ax.add_patch(item_rect)

                    # Add item label
                    ax.text(x + w / 2, y + h / 2, f'I{item_id}',
                            ha='center', va='center', fontsize=8, fontweight='bold')

            # Draw bin outline
            bin_rect = mpatches.Rectangle((0, 0), bin_width, bin_height,
                                          linewidth=2, edgecolor='black',
                                          facecolor='none', alpha=0.7)
            ax.add_patch(bin_rect)

            ax.set_xlim(-1, bin_width + 1)
            ax.set_ylim(-1, bin_height + 1)
            ax.set_aspect('equal')
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
            ax.set_title(f'Bin {bin_idx} - Layers: {len(layers)}')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(bins_data), 4):
            axes[i].set_visible(False)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()


def test_bpp_env():
    """Test the BPP environment"""
    # Test data
    problem_data = {
        "bin_size": [10, 10],
        "items_size": [
            [3, 4],
            [2, 5],
            [4, 2],
            [3, 3],
            [2, 2]
        ],
        "bin_status": False,
        "can_rotate": True,
        "dimension": "2D",
        "label": "2DOFBPP"
    }

    # Test solution
    solution = {
        "bins": [
            {
                "bin_size": [10, 10],
                "items": [
                    {"item_id": 0, "position": [0, 0, 0], "size": [3, 4, 1]},
                    {"item_id": 1, "position": [3, 0, 0], "size": [2, 5, 1]},
                    {"item_id": 2, "position": [5, 0, 0], "size": [4, 2, 1]},
                ]
            },
            {
                "bin_size": [10, 10],
                "items": [
                    {"item_id": 3, "position": [0, 4, 0], "size": [3, 3, 1]},
                    {"item_id": 4, "position": [3, 5, 0], "size": [2, 2, 1]}
                ]
            }
        ]
    }

    # Create environment
    env = BPPEnv("2DOFBPP")

    # Test validation with legacy format
    is_valid, message = validate_packing(problem_data, solution)
    print(f"Legacy format validation: {is_valid}, Message: {message}")

    # Test reward calculation with legacy format
    reward = env.get_reward(problem_data, solution)
    print(f"Legacy format reward: {reward}")

    # Test utilization with legacy format
    utilization = env.calculate_utilization(problem_data, solution)
    print(f"Legacy format utilization: {utilization:.2%}")

    # Test visualization with legacy format
    env.plot_packing_2d(problem_data, solution, "Test 2D Packing - Legacy Format")


if __name__ == '__main__':
    test_bpp_env()
