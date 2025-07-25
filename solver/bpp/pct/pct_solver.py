#!/usr/bin/env python3
"""
PCT Solver for BPP using Packing Completion Transformer
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
from typing import Dict, List, Tuple
import argparse
from solver.bpp.pct.attention_model import AttentionModel


class PCTSolver:
    """PCT (Packing Completion Transformer) Solver for BPP"""
    
    def __init__(self, problem_type: str = "2DOFBPP", solver_name: str = "PCT", **kwargs):
        """
        Initialize PCT Solver
        
        Args:
            problem_type: Type of BPP problem (2DOFBPP, 2DOFBPPR, etc.)
            solver_name: Name of the solver
            **kwargs: Additional parameters for PCT
        """
        self.problem_type = problem_type
        self.solver_name = solver_name
        
        # Extract problem characteristics
        self.is_2d = problem_type.startswith('2D')
        self.is_online = 'ON' in problem_type
        self.can_rotate = 'R' in problem_type
        
        # PCT parameters
        self.setting = kwargs.get('setting', 2)
        self.internal_node_holder = kwargs.get('internal_node_holder', 80)
        self.leaf_node_holder = kwargs.get('leaf_node_holder', 50)
        self.lnes = kwargs.get('lnes', 'EMS')
        self.shuffle = kwargs.get('shuffle', True)
        
        # Model parameters
        self.embedding_dim = kwargs.get('embedding_dim', 64)
        self.hidden_dim = kwargs.get('hidden_dim', 128)
        self.n_encode_layers = kwargs.get('n_encode_layers', 1)
        self.n_heads = kwargs.get('n_heads', 1)
        
        # Set internal node length based on setting
        if self.setting == 1:
            self.internal_node_length = 6
        elif self.setting == 2:
            self.internal_node_length = 6
        elif self.setting == 3:
            self.internal_node_length = 7
        else:
            self.internal_node_length = 6
        
        # Initialize model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model if available
        model_path = kwargs.get('model_path', None)
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load pre-trained PCT model"""
        try:
            self.model = AttentionModel(
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                n_encode_layers=self.n_encode_layers,
                n_heads=self.n_heads,
                internal_node_holder=self.internal_node_holder,
                internal_node_length=self.internal_node_length,
                leaf_node_holder=self.leaf_node_holder
            )
            
            # Load pre-trained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded PCT model from {model_path}")
            
        except Exception as e:
            print(f"Failed to load PCT model: {e}")
            self.model = None

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
            'item_size_set': items_size_pct,
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

    def _create_pct_observation(self, pct_instances: Dict) -> np.ndarray:
        """
        Create PCT observation format
        
        Args:
            pct_instances: PCT format instances
            
        Returns:
            PCT observation array
        """
        container_size = pct_instances['container_size']
        items = pct_instances['item_size_set']
        
        # Initialize observation array
        total_nodes = self.internal_node_holder + self.leaf_node_holder + 1
        obs = np.zeros((1, total_nodes, 9))  # 9 features per node
        
        # Set container size as normalization factor
        norm_factor = 1.0 / max(container_size)
        
        # Fill internal nodes (packed items) - initially empty
        for i in range(self.internal_node_holder):
            obs[0, i, :6] = 0  # No packed items initially
            obs[0, i, 6] = 0   # Density (optional)
            obs[0, i, 7] = 0   # Valid flag
            obs[0, i, 8] = 1   # Full mask (should be encoded)
        
        # Fill leaf nodes (placement candidates) - initially empty
        for i in range(self.leaf_node_holder):
            obs[0, self.internal_node_holder + i, :6] = 0  # No placement candidates initially
            obs[0, self.internal_node_holder + i, 6] = 0   # Additional feature
            obs[0, self.internal_node_holder + i, 7] = 0   # Valid flag (invalid initially)
            obs[0, self.internal_node_holder + i, 8] = 0   # Full mask (should not be encoded initially)
        
        # Fill next item
        if items:
            next_item = items[0]  # First item to pack
            obs[0, -1, :3] = 0    # Density and padding
            obs[0, -1, 3:6] = np.array(next_item) * norm_factor  # Normalized item size
            obs[0, -1, 6:8] = 0   # Additional features
            obs[0, -1, 8] = 1     # Full mask (should be encoded)
        
        return obs

    def _pct_heuristic_solve(self, pct_instances: Dict) -> List[Dict]:
        """
        Use heuristic approach to solve BPP when PCT model is not available
        
        Args:
            pct_instances: PCT format instances
            
        Returns:
            Solution in BPPEnv format
        """
        container_size = pct_instances['container_size']
        items = pct_instances['item_size_set']
        
        # Simple first-fit decreasing heuristic
        bins = []
        current_bin = {
            'bin_size': container_size,
            'items': []
        }
        
        # Sort items by volume (largest first)
        sorted_items = sorted(enumerate(items), 
                            key=lambda x: x[1][0] * x[1][1] * x[1][2], 
                            reverse=True)
        
        for item_id, item_size in sorted_items:
            # Try to place item in current bin
            placed = False
            
            # Simple placement logic (can be improved)
            if (current_bin['items'] == [] or 
                (item_size[0] <= container_size[0] and 
                 item_size[1] <= container_size[1] and 
                 item_size[2] <= container_size[2])):
                
                # Place at origin
                position = [0, 0, 0]
                current_bin['items'].append({
                    'item_id': item_id,
                    'position': position,
                    'size': item_size
                })
                placed = True
            
            if not placed:
                # Start new bin
                bins.append(current_bin)
                current_bin = {
                    'bin_size': container_size,
                    'items': [{
                        'item_id': item_id,
                        'position': [0, 0, 0],
                        'size': item_size
                    }]
                }
        
        # Add last bin
        if current_bin['items']:
            bins.append(current_bin)
        
        return bins

    def solve(self, instances: Dict) -> Dict:
        """
        Solve BPP using PCT algorithm
        
        Args:
            instances: BPPEnv format instances
            
        Returns:
            Solution in BPPEnv format
        """
        # Convert BPP format to PCT format
        pct_instances = self._convert_bpp_to_pct_format(instances)
        
        # Check if PCT model is available
        if self.model is not None:
            try:
                # Use PCT model for solving
                solution = self._pct_model_solve(pct_instances)
            except Exception as e:
                print(f"PCT model solving failed: {e}, falling back to heuristic")
                solution = self._pct_heuristic_solve(pct_instances)
        else:
            # Use heuristic approach
            solution = self._pct_heuristic_solve(pct_instances)
        
        # Convert solution back to BPPEnv format
        bpp_solution = self._convert_pct_to_bpp_format(solution, instances)
        
        return bpp_solution

    def _pct_model_solve(self, pct_instances: Dict) -> List[Dict]:
        """
        Solve using PCT model (placeholder for actual implementation)
        
        Args:
            pct_instances: PCT format instances
            
        Returns:
            Solution in PCT format
        """
        # This is a placeholder - actual PCT model solving would be more complex
        # For now, fall back to heuristic
        return self._pct_heuristic_solve(pct_instances)

    def _convert_pct_to_bpp_format(self, pct_solution: List[Dict], original_instances: Dict) -> Dict:
        """
        Convert PCT solution format back to BPPEnv format
        
        Args:
            pct_solution: PCT format solution
            original_instances: Original BPPEnv instances
            
        Returns:
            BPPEnv format solution
        """
        bpp_solution = {'bins': []}
        
        for pct_bin in pct_solution:
            # Convert bin size
            if self.is_2d and len(original_instances['bin_size']) == 2:
                bin_size = pct_bin['bin_size'][:2]  # Remove depth for 2D
            else:
                bin_size = pct_bin['bin_size']
            
            bpp_bin = {
                'bin_size': bin_size,
                'items': []
            }
            
            for pct_item in pct_bin['items']:
                # Convert item data
                if self.is_2d and len(original_instances['items_size'][0]) == 2:
                    position = pct_item['position'][:2]  # Remove z for 2D
                    size = pct_item['size'][:2]          # Remove depth for 2D
                else:
                    position = pct_item['position']
                    size = pct_item['size']
                
                bpp_item = {
                    'item_id': pct_item['item_id'],
                    'position': position,
                    'size': size
                }
                bpp_bin['items'].append(bpp_item)
            
            bpp_solution['bins'].append(bpp_bin)
        
        return bpp_solution

    def get_solver_info(self) -> Dict:
        """Get solver information"""
        return {
            "solver_name": self.solver_name,
            "problem_type": self.problem_type,
            "setting": self.setting,
            "internal_node_holder": self.internal_node_holder,
            "leaf_node_holder": self.leaf_node_holder,
            "lnes": self.lnes,
            "model_loaded": self.model is not None
        }


def test_pct_solver():
    """Test the PCT solver"""
    # Test data
    instances = {
        "bin_size": [8, 8],
        "items_size": [
            [3, 4],
            [2, 5],
            [4, 2],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
        ],
        "can_rotate": "true",
        "dimension": "2D",
        "bin_status": "false",
        "label": "2DOFBPP"
    }
    
    # Create solver
    solver = PCTSolver(
        problem_type="2DOFBPP",
        solver_name="PCT",
        setting=2,
        internal_node_holder=20,
        leaf_node_holder=10
    )
    
    # Solve
    solution = solver.solve(instances)


    print("Solution:")
    for i, bin_data in enumerate(solution['bins']):
        print(f"Bin {i+1}: {len(bin_data['items'])} items")
        for item in bin_data['items']:
            print(f"  Item {item['item_id']}: pos{item['position']}, size{item['size']}")


if __name__ == '__main__':
    test_pct_solver()
