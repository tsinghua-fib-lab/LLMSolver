VRP_VARIANTS = [
    "CVRP", "VRPB", "VRPBL", "VRPBLTW", "VRPBTW", "VRPL", "VRPLTW",
    "VRPMB", "VRPMBL", "VRPMBLTW", "VRPMBTW", "VRPTW",
    "OVRP", "OVRPB", "OVRPBL", "OVRPBLTW", "OVRPBTW", "OVRPL",
    "OVRPLTW", "OVRPMB", "OVRPMBL", "OVRPMBLTW", "OVRPMBTW", "OVRPTW"
]

VRP_DESCRIPTION = {
    'C': 'capacity, ignored when other features are present',
    'B': 'backhaul demand which can be satisfied only when all the linehual demand is satisfied',
    'L': 'the vichiles have distance limits',
    'O': 'open route, which means that the vichiles do not have to return to the depot',
    'MB': 'mixed backhaul, which means that both linehaul and backhaul demand can be satisfied simultaneously',
    'TW': 'time window constraints that specify the time frame for service',
}

VRP_ADDITIONAL_DESCRIPTION = "If the description contains any attribute in 'O', 'B', 'L' or 'TW', the most likely corresponding classification is given, otherwise is 'C'."




SP_VARIANTS = [
    "JSSP", "FJSSP", "FSSP", "HFSSP", "OSSP", "ASP",
]

SP_DESCRIPTION = {
    'JSSP': 'Job Shop Scheduling Problem - fixed machine per operation, predefined operation order',
    'FJSSP': 'Flexible Job Shop Scheduling Problem - flexible machine choice per operation, predefined operation order',
    'FSSP': 'Flow-Shop Scheduling Problem - all jobs follow the same machine sequence',
    'HFSSP': 'Hybrid Flow-shop Scheduling Problem - multiple stages with parallel machines in at least one stage',
    'OSSP': 'Open Shop Scheduling Problem - no predefined operation order',
    'ASP': 'Assembly Scheduling Problem - operations have precedence constraints (DAG)',
}

SP_ADDITIONAL_DESCRIPTION = ""




BP_VARIANTS = [
    "2DOFBPP", "2DOFBPPR", "2DONBPP", "2DONBPPR", "3DOFBPP", "3DOFBPPR", "3DONBPP", "3DONBPPR"
]

BP_DESCRIPTION = {
    '2DOFBPP': '2D Offline Bin Packing Problem: Place 2D items, each with only length and width, into 2D bins that also have only length and width.',
    '2DOFBPPR': '2D Offline Rotatable Bin Packing Problem: Place 2D items, each with only length and width, into 2D bins that also have only length and width, allowing items to be rotated during placement.',
    '2DONBPP': '2D Online Bin Packing Problem: Place 2D items, each with only length and width, into 2D bins that also have only length and width. Items arrive sequentially, and the dimensions of items not yet placed are unknown when placing the current item.',
    '2DONBPPR': '2D Online Rotatable Bin Packing Problem: Place 2D items, each with only length and width, into 2D bins that also have only length and width, allowing items to be rotated during placement. Items arrive sequentially, and the dimensions of items not yet placed are unknown when placing the current item.',
    '3DOFBPP': '3D Offline Bin Packing Problem: Place 3D items, each with length, width, and height, into 3D bins that also have length, width, and height.',
    '3DOFBPPR': '3D Offline Rotatable Bin Packing Problem: Place 3D items, each with length, width, and height, into 3D bins that also have length, width, and height, allowing items to be rotated.',
    '3DONBPP': '3D Online Bin Packing Problem: Place 3D items, each with length, width, and height, into 3D bins that also have length, width, and height. Items arrive sequentially, and the dimensions of items not yet placed are unknown when placing the current item.',
    '3DONBPPR': '3D Online Rotatable Bin Packing Problem: Place 3D items, each with length, width, and height, into 3D bins that also have length, width, and height, allowing items to be rotated during placement. Items arrive sequentially, and the dimensions of items not yet placed are unknown when placing the current item.',
}

BP_ADDITIONAL_DESCRIPTION = ""


GRAPH_VARIANTS = [
    "MAXCUT", "MIS", "MVC", "MAXCLIQUE"
]

GRAPH_DESCRIPTION = {
    "MAXCUT": 'Max-Cut Problem – partition the nodes of a given graph into two disjoint subsets such that the number (or total weight) of edges crossing between them is maximized.',
    "MIS": 'Maximal Independent Set problem – select a subset of nodes from a given graph such that no two selected nodes are adjacent and no additional node can be added without violating this condition; the goal is to find a set that is maximal, ensuring that every unselected node is adjacent to at least one selected node.',
    "MVC": 'Minimum Vertex Cover problem – find the smallest possible subset of nodes in a graph such that every edge is incident to at least one selected node, minimizing the number of chosen nodes while ensuring that all edges are covered.',
    "MAXCLIQUE": "Maximum Clique problem - find the largest subset of nodes such that every pair of nodes in the subset is connected by an edge.",
}

GRAPH_ADDITIONAL_DESCRIPTION = ""


KP_VARIANTS = [
    "KNAPSACK"
]

KP_DESCRIPTION = {
    "KNAPSACK": 'Knapsack problem - given a set of items, each with a weight and a value, and a knapsack with a maximum weight capacity, the objective is to determine the most valuable subset of items that can fit within the knapsack's weight limit.',
}

KP_ADDITIONAL_DESCRIPTION = ""

PROBLEM_INFO = {
    "VRP": {'variants': VRP_VARIANTS, 'description': VRP_DESCRIPTION, 'additional_description': VRP_ADDITIONAL_DESCRIPTION},
    "SP": {'variants': SP_VARIANTS, 'description': SP_DESCRIPTION, 'additional_description': SP_ADDITIONAL_DESCRIPTION},
    "BP": {'variants': BP_VARIANTS, 'description': BP_DESCRIPTION, 'additional_description': BP_ADDITIONAL_DESCRIPTION},
    "GRAPH": {'variants': GRAPH_VARIANTS, 'description': GRAPH_DESCRIPTION, 'additional_description': GRAPH_ADDITIONAL_DESCRIPTION},
    "KP": {'variants': KP_VARIANTS, 'description': KP_DESCRIPTION, 'additional_description': KP_ADDITIONAL_DESCRIPTION},
}

PROBLEM_DESCRIPTION = {
    "VRP": "Vehicle Routing Problem - a set of customers with demands and a set of vehicles with limited capacity, the goal is to minimize the total distance traveled by the vehicles while satisfying the demands of the customers.",
    "SP": "Scheduling Problem - a set of jobs with operations and a set of machines, the goal is to minimize the total makespan or total tardiness.",
    "BP": "Bin Packing Problem - a set of items and a set of fixed bins, the goal is to minimize the total number of bins used while placing all the items into the bins.",
    "GRAPH": "Graph Problem - a class of problems involving selection or partitioning of nodes and edges in a graph, aiming to optimize structures like cuts, covers, independent sets, or cliques.",
    "KP": 'Knapsack problem - given a set of items, each with a weight and a value, and a knapsack with a maximum weight capacity, the objective is to determine the most valuable subset of items that can fit within the knapsack's weight limit.',

}

