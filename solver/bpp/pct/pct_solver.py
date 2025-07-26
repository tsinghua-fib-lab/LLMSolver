#!/usr/bin/env python3
"""
PCT Solver for BPP using Packing Completion Transformer
"""

import sys
import os

import gym
import numpy as np

from solver.bpp.pct.model import DRL_GAT

import torch
from typing import Dict, List, Tuple, Optional
from solver.bpp.pct.tools import get_args, load_policy, registration_envs, get_leaf_nodes_with_factor

registration_envs()


class PCTSolver:
    """PCT (Packing Completion Transformer) Solver for BPP"""

    def __init__(self, problem_type: str = "2DOFBPP", solver_name: str = "PCT",
                 setting: int = 2, internal_node_holder: int = 80, leaf_node_holder: int = 50,
                 device: Optional[torch.device] = None, **kwargs):
        """
        Initialize PCT Solver

        Args:
            problem_type: Type of BPP problem (2DOFBPP, 2DOFBPPR, etc.)
            solver_name: Name of the solver
            setting: PCT setting (1, 2, or 3)
            internal_node_holder: Maximum number of internal nodes
            leaf_node_holder: Maximum number of leaf nodes
            device: Device to run the model on
            **kwargs: Additional parameters for PCT
        """
        self.problem_type = problem_type
        self.solver_name = solver_name
        self.setting = setting
        self.internal_node_holder = internal_node_holder
        self.leaf_node_holder = leaf_node_holder
        self.lnes = kwargs.get('lnes', 'EMS')
        self.shuffle = kwargs.get('shuffle', True)

        # Extract problem characteristics
        self.is_2d = problem_type.startswith('2D')
        self.is_online = 'ON' in problem_type
        self.can_rotate = 'R' in problem_type

        # Set up device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Initialize model
        self.model = None
        self.env = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the PCT model and load pre-trained weights"""
        try:
            # Use get_args from tools to avoid code duplication
            self.args = get_args()
            self.args.setting = self.setting
            self.args.lnes = self.lnes
            self.args.internal_node_holder = self.internal_node_holder
            self.args.leaf_node_holder = self.leaf_node_holder
            self.args.shuffle = self.shuffle
            self.args.num_processes = 1

            # Create the PCT model
            self.model = DRL_GAT(self.args)
            self.model = self.model.to(self.device)

            # Load pre-trained model
            checkpoint_path = os.path.join(
                os.path.dirname(__file__),
                'checkpoint',
                f'setting{self.setting}_discrete.pt'
            )

            if os.path.exists(checkpoint_path):
                self.model = load_policy(checkpoint_path, self.model)
                print(f"✅ Pre-trained PCT model loaded from {checkpoint_path}")
                self.model.eval()
            else:
                print(f"⚠️  No pre-trained model found at {checkpoint_path}")
                self.model = None

        except Exception as e:
            print(f"❌ Error initializing PCT model: {e}")
            self.model = None

    def _create_environment(self, instances: Dict):
        """Create PCT environment for solving"""
        try:
            # Convert BPP format to PCT format
            pct_instances = self._convert_bpp_to_pct_format(instances)

            # Create environment with proper parameters
            env = gym.make('PctDiscrete-v0',
                           setting=self.setting,
                           container_size=pct_instances['container_size'],
                           data=[pct_instances['item_list']],  # Use new instances parameter
                           internal_node_holder=self.internal_node_holder,
                           leaf_node_holder=self.leaf_node_holder,
                           next_holder=1,
                           LNES=self.lnes,
                           shuffle=False,
                           load_test_data=False,
                           sample_from_distribution=False)

            return env

        except Exception as e:
            print(f"❌ Error creating PCT environment: {e}")
            print(f"   Container size: {pct_instances.get('container_size', 'N/A')}")
            print(f"   Item set size: {len(pct_instances.get('item_list', []))}")
            print(f"   Setting: {self.setting}")
            return None

    def _convert_bpp_to_pct_format(self, instances: Dict) -> Dict:
        """
        Convert BPPEnv format to PCT format

        Args:
            instances: BPPEnv format instances

        Returns:
            PCT format instances
        """
        # Extract BPP data
        bin_size = instances['bin_size']
        items_size = instances['items_size']

        # Handle 2D to 3D conversion for PCT
        if self.is_2d:
            if len(bin_size) == 2:
                container_size_pct = [bin_size[0], bin_size[1], 1]
            else:
                container_size_pct = bin_size

            if len(items_size[0]) == 2:
                items_size_pct = [[item[0], item[1], 1] for item in items_size]
            else:
                items_size_pct = items_size
        else:
            container_size_pct = bin_size
            items_size_pct = items_size

        # Convert to PCT format
        pct_instances = {
            'container_size': container_size_pct,
            'item_list': items_size_pct,
            'setting': self.setting,
            'lnes': self.lnes,
            'internal_node_holder': self.internal_node_holder,
            'leaf_node_holder': self.leaf_node_holder,
            'shuffle': self.shuffle,
            'can_rotate': instances.get('can_rotate', False),
            'dimension': instances.get('dimension', '3D'),
            'bin_status': instances.get('bin_status', 'false')
        }
        return pct_instances

    def _create_environment_for_bin(self, pct_instances: Dict) -> Optional[gym.Env]:
        """Create PCT environment for packing a single bin"""
        try:
            # Create environment with remaining items
            env = gym.make('PctDiscrete-v0',
                           setting=self.setting,
                           container_size=pct_instances['container_size'],
                           data=[pct_instances['item_list']],  # Use remaining items
                           internal_node_holder=self.internal_node_holder,
                           leaf_node_holder=self.leaf_node_holder,
                           next_holder=1,
                           LNES=self.lnes,
                           shuffle=False,
                           load_test_data=False,
                           sample_from_distribution=False)
            return env
        except Exception as e:
            print(f"❌ Error creating environment for bin: {e}")
            return None

    def _solve_bin(self, PCT_policy, eval_envs, timeStr, args, device, eval_freq=100, factor=1):
        PCT_policy.eval()
        obs = eval_envs.reset()
        obs = torch.FloatTensor(obs).to(device).unsqueeze(dim=0)
        all_nodes, leaf_nodes = get_leaf_nodes_with_factor(obs, args.num_processes,
                                                           args.internal_node_holder, args.leaf_node_holder)
        batchX = torch.arange(args.num_processes)
        step_counter = 0
        episode_ratio = []
        episode_length = []

        while step_counter < eval_freq:
            with torch.no_grad():
                selectedlogProb, selectedIdx, policy_dist_entropy, value = PCT_policy(all_nodes, True,
                                                                                      normFactor=factor)
            selected_leaf_node = leaf_nodes[batchX, selectedIdx.squeeze()]
            items = eval_envs.packed
            obs, reward, done, infos = eval_envs.step(selected_leaf_node.cpu().numpy()[0][0:6])

            if done:
                print('Episode {} ends.'.format(step_counter))
                if 'ratio' in infos.keys():
                    episode_ratio.append(infos['ratio'])
                if 'counter' in infos.keys():
                    episode_length.append(infos['counter'])

                return items

            obs = torch.FloatTensor(obs).to(device).unsqueeze(dim=0)
            all_nodes, leaf_nodes = get_leaf_nodes_with_factor(obs, args.num_processes,
                                                               args.internal_node_holder,
                                                               args.leaf_node_holder)
            all_nodes, leaf_nodes = all_nodes.to(device), leaf_nodes.to(device)

    def solve(self, instances: Dict) -> Optional[Dict]:
        """
        Solve BPP using PCT algorithm with multiple bins strategy to minimize bin count.

        Args:
            instances: BPPEnv format instances

        Returns:
            Solution in BPPEnv format or None if cannot solve
        """
        if self.model is None:
            print("❌ PCT model not loaded, cannot solve")
            return None

        try:
            bin_size = instances['bin_size']
            items_size = instances['items_size']
            can_rotate = instances.get('can_rotate', False)
            is_2d = len(bin_size) == 2

            print(f"📦 Multi-bin packing: {len(items_size)} items, Bin size: {bin_size}, Rotation: {can_rotate}")

            # Track remaining items with their original indices
            remaining_items = [{"item_id": i, "size": size} for i, size in enumerate(items_size)]
            solution_bins = []
            bin_count = 0

            while remaining_items:
                bin_count += 1
                print(f"\n🔧 Packing bin {bin_count} ({len(remaining_items)} items left)")

                current_items = [item["size"] for item in remaining_items]
                pct_instances = self._convert_bpp_to_pct_format({
                    "bin_size": bin_size,
                    "items_size": current_items,
                    "can_rotate": can_rotate,
                    "dimension": "2D" if is_2d else "3D",
                    "bin_status": instances.get('bin_status', False)
                })

                env = self._create_environment_for_bin(pct_instances)
                if env is None:
                    print(f"❌ Failed to create environment for bin {bin_count}")
                    break

                packed_result = self._solve_bin(
                    self.model, env, "timeStr", self.args, self.device,
                    eval_freq=self.args.evaluation_episodes, factor=self.args.normFactor
                )

                if not packed_result:
                    print(f"❌ No items packed in bin {bin_count}")
                    break

                bin_items = []
                packed_indices = []

                for idx, pack in enumerate(packed_result):
                    if idx >= len(remaining_items):
                        break
                    original_item = remaining_items[idx]
                    item_id = original_item["item_id"]
                    original_size = original_item["size"]
                    if is_2d:
                        position = [int(pack[3]), int(pack[4])]
                        size = [int(pack[0]), int(pack[1])]
                    else:
                        position = [int(pack[3]), int(pack[4]), int(pack[5])]
                        size = [int(pack[0]), int(pack[1]), int(pack[2])]
                    if not self._validate_item_size(original_size, size, can_rotate, is_2d):
                        print(f"⚠️  Size mismatch for item {item_id}: expected {original_size}, got {size}")
                        continue
                    bin_items.append({
                        "item_id": item_id,
                        "position": position,
                        "size": size
                    })
                    packed_indices.append(idx)

                if not bin_items:
                    print(f"❌ No valid items packed in bin {bin_count}")
                    break

                solution_bins.append({
                    "bin_size": bin_size,
                    "items": bin_items
                })
                print(f"✅ Bin {bin_count}: packed {len(bin_items)} items")

                for idx in sorted(packed_indices, reverse=True):
                    del remaining_items[idx]

            if remaining_items:
                print(f"❌ Failed to pack all items: {len(remaining_items)} items remaining")
                return None

            solution = {"bins": solution_bins}
            print(f"🎉 All items packed in {len(solution_bins)} bins.")
            return solution

        except Exception as e:
            print(f"❌ Error in solve: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _validate_item_size(self, original_size: List[int], packed_size: List[int], 
                           can_rotate: bool, is_2d: bool) -> bool:
        """
        Validate that packed size is a valid rotation of original size.
        """
        if not can_rotate:
            return original_size == packed_size
        if is_2d:
            return packed_size in [original_size, [original_size[1], original_size[0]]]
        # 3D
        perms = [
            original_size,
            [original_size[1], original_size[0], original_size[2]],
            [original_size[0], original_size[2], original_size[1]],
            [original_size[1], original_size[2], original_size[0]],
            [original_size[2], original_size[0], original_size[1]],
            [original_size[2], original_size[1], original_size[0]]
        ]
        return packed_size in perms

# --- TESTING ---
def test_pct_solver():
    """Test the PCT solver with both single and multiple instances"""
    print("🧪 Testing PCT Solver with Instances Support")
    print("=" * 60)

    # Test 1: Single instance
    print("\n📦 Test 1: Single Instance")
    print("-" * 40)

    single_instance = {
        "bin_size": [8, 8],
        "items_size": [
            [3, 4],
            [2, 5],
            [4, 2],
            [3, 3],
            [4, 3],
            [7, 3],
            [2, 3],
            [8, 3],
        ],
        "can_rotate": True,
        "dimension": "2D",
        "bin_status": False,
        "label": "2DOFBPP"
    }

    print(f"📦 Problem: {len(single_instance['items_size'])} items to pack")
    print(f"📏 Bin size: {single_instance['bin_size']}")
    print(f"🔄 Rotation: {single_instance['can_rotate']}")

    # Test single instance with different settings
    for setting in [1]:
        print(f"\n🔧 Testing Setting {setting}")
        print("-" * 20)
        solver = PCTSolver(
            problem_type="2DOFBPP",
            solver_name="PCT",
            setting=setting,
            internal_node_holder=80,
            leaf_node_holder=50
        )
        solution = solver.solve(single_instance)
        if solution and 'bins' in solution:
            print(f"Solution found: {len(solution['bins'])} bins used")
            for i, bin_data in enumerate(solution['bins']):
                print(f"  Bin {i + 1}: {len(bin_data['items'])} items")
                for item in bin_data['items']:
                    print(f"    Item {item['item_id']}: pos{item['position']}, size{item['size']}")
        else:
            print("❌ No solution found")

    print("-" * 40)


if __name__ == '__main__':
    test_pct_solver()
