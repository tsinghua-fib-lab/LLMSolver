import os

import numpy as np
import torch
import vrplib
from tensordict import TensorDict

from data.vrplib_wrapper.vrplib_wrapper import read_instance

problem_type_suffix_dict = {
    "CVRP": ".vrp",
    "CVRPTW": ".vrptw",
    "DCVRP": ".vrp",
    "OVRP": ".vrp",
    "VRPB": ".vrpb",
}

problem_type_dataset_dict = {
    "CVRP": ["Augerat_A", "Augerat_B", "Augerat_P"],
    "CVRPTW": ["Homberger_200", "Homberger_400", "Homberger_600", "Homberger_800", "Homberger_1000"],
    "DCVRP": ["GWKC"],
    "OVRP": ["A", "B", "C"],
    "VRPB": ["G", "T"],
}


# attribute_dict = {
#     "num_depots": torch.int32,
#     "locs": torch.narray[np.ndarray[np.float32, Any]],
#     "demand_backhaul": np.ndarray[np.float32, Any],  # (C)
#     "demand_linehaul": np.ndarray[np.float32, Any],  # (B)
#     "backhaul_class": np.int32,  # (B)
#     "distance_limit": np.float32,  # (L)
#     "time_windows": np.ndarray[np.ndarray[np.float32, Any]],  # (TW)
#     "service_time": np.ndarray[np.float32, Any],  # (TW)
#     "vehicle_capacity": np.float32,  # (C)
#     "capacity_original": np.float32,  # unscaled capacity (C)
#     "open_route": np.bool,  # (O)
#     "speed": np.float32,  # common
# },


def move_indices_to_begin(indices_list: list, begin_indices: list) -> list:
    """
    Sorts the given indices first, then moves the corresponding rows
    to the end of the NumPy array.

    Parameters:
    - coords: np.ndarray, shape [n, 2], representing 2D coordinates.
    - indices: list, indices of rows to move to the begin.

    Returns:
    - A reordered NumPy array.
    """
    # Sort the indices first
    sorted_indices = sorted(begin_indices)

    # Get the remaining indices (those not in sorted_indices)
    remaining_indices = [i for i in indices_list if i not in sorted_indices]

    # Create a new order: first the remaining indices, then the sorted indices
    new_order = sorted_indices + remaining_indices

    return new_order


def move_indices_to_end(indices_list: list, end_indices: list) -> list:
    """
    Sorts the given indices first, then moves the corresponding rows
    to the end of the NumPy array.

    Parameters:
    - coords: np.ndarray, shape [n, 2], representing 2D coordinates.
    - indices: list, indices of rows to move to the begin.

    Returns:
    - A reordered NumPy array.
    """
    # Sort the indices first
    sorted_indices = sorted(end_indices)

    # Get the remaining indices (those not in sorted_indices)
    remaining_indices = [i for i in indices_list if i not in sorted_indices]

    # Create a new order: first the remaining indices, then the sorted indices
    new_order = remaining_indices + sorted_indices

    return new_order


def normalize_coord(coord: torch.Tensor) -> torch.Tensor:
    x, y = coord[:, 0], coord[:, 1]

    # Find the global minimum and maximum values
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Compute a unified scaling factor (largest range)
    scale = max(x_max - x_min, y_max - y_min)

    # Avoid division by zero
    if scale == 0:
        return coord - torch.tensor([x_min, y_min], device=coord.device)

    # Apply uniform scaling
    coord_scaled = (coord - torch.tensor([x_min, y_min], device=coord.device)) / scale
    return coord_scaled


def convert_dict2tensordict(inst: dict) -> TensorDict:
    indices_list = list(range(inst["node_coord"].shape[0]))
    indices_list = move_indices_to_begin(indices_list, inst["depot"])

    inst["node_coord"] = inst["node_coord"][indices_list]

    # coords = torch.tensor(inst["node_coord"]).float()
    # coords_norm = normalize_coord(coords)
    instance_convert = {
        "num_depots": len(inst["depot"]),
        "locs": inst["node_coord"].astype(np.float32),
        "demand_linehaul": inst["demand"].astype(np.float32) / float(inst["capacity"]),
        "capacity_original": float(inst["capacity"]),
    }
    if "backhaul" in inst and 'B' in inst['type']:
        backhaul_idx_list = inst["backhaul"]
        demand_backhaul = np.zeros_like(instance_convert["demand_linehaul"])
        demand_backhaul[backhaul_idx_list] = instance_convert["demand_linehaul"][backhaul_idx_list]
        instance_convert["demand_linehaul"][backhaul_idx_list] = 0
        instance_convert["demand_backhaul"] = demand_backhaul

        if "MB" in inst['type']:
            instance_convert["backhaul_class"] = [2]
        elif "B" in inst['type']:
            instance_convert["backhaul_class"] = [1]

    if "OVRP" in inst['type']:
        instance_convert["open_route"] = [True]
    else:
        instance_convert["open_route"] = [False]

    if "time_window" in inst:
        inst["time_windows"] = inst["time_window"][indices_list]
        instance_convert["service_time"] = np.full(instance_convert["locs"].shape[0], inst["service_time"]).astype(
            np.float32)
        instance_convert["time_windows"] = inst["time_windows"].astype(np.float32)
    else:
        instance_convert["service_time"] = np.full(instance_convert["locs"].shape[0], 0).astype(np.float32)
        instance_convert["time_windows"] = np.zeros_like(instance_convert["locs"]).astype(np.float32)
        instance_convert["time_windows"][:, 1] = np.inf

    if "distance" in inst:
        instance_convert["distance_limit"] = np.array([inst["distance"]]).astype(np.float32)
    else:
        instance_convert["distance_limit"] = np.array([np.inf]).astype(np.float32)

    instance_convert["demand_linehaul"] = instance_convert["demand_linehaul"][len(inst["depot"]):]
    if "demand_backhaul" in instance_convert:
        instance_convert["demand_backhaul"] = instance_convert["demand_backhaul"][len(inst["depot"]):]
    else:
        instance_convert["demand_backhaul"] = np.zeros_like(instance_convert["demand_linehaul"])

    instance_td = TensorDict(instance_convert, batch_size=[])[None]
    return instance_td


def load_problem_type_instances_path(problem_type="CVRP", root_dir="/data1/zsf/Workplace/LLMSolver_Poject/LLMSolver"):
    instances = {}
    data_dir = os.path.join(root_dir, f"dataset/cvrp_variant/{problem_type}/INSTANCES")
    tour_dir = os.path.join(root_dir, f"dataset/cvrp_variant/{problem_type}/TOURS")
    dataset_name = problem_type_dataset_dict[problem_type][0]
    # Walk through the vrplib directory recursively
    for root, dirs, files in sorted(os.walk(os.path.join(data_dir, dataset_name))):
        for file in files:
            if file.endswith(problem_type_suffix_dict[problem_type]):
                # Initialize the dictionary for this instance
                instance_name = file[:-len(problem_type_suffix_dict[problem_type])]  # Remove the '.vrp' extension
                instances[instance_name] = {}
                # Print the file for verification
                instances[instance_name]["data"] = os.path.join(root, file)  # Save the VRP file path

    tour_suffix = '.tour'
    for root, dirs, files in sorted(os.walk(os.path.join(tour_dir, dataset_name))):
        for file in files:
            if file.endswith(tour_suffix):
                instance_name = '.'.join(file.split('.')[:-2])
                if instance_name not in instances:
                    raise FileNotFoundError(f"No data file found for {instance_name}")
                instances[instance_name]["tours"] = os.path.join(root, file)

    valid_instance_list = []
    for instance_name in instances:
        if instances[instance_name]["data"] is None:
            print(f"No data file found for {instance_name}")
            continue
        elif instances[instance_name].get("tours") is None:
            print(f"No tour data file found for {instance_name}")
            continue
        valid_instance_list.append(instance_name)
        print(f"Found VRP: {instance_name}")
    instances = {key: instances[key] for key in valid_instance_list}

    return instances
