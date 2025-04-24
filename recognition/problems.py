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



PROBLEM_INFO = {
    "VRP": {'variants': VRP_VARIANTS, 'description': VRP_DESCRIPTION, 'additional_description': VRP_ADDITIONAL_DESCRIPTION},
    "SP": {'variants': SP_VARIANTS, 'description': SP_DESCRIPTION, 'additional_description': SP_ADDITIONAL_DESCRIPTION},
}

PROBLEM_DESCRIPTION = {
    "VRP": "Vehicle Routing Problem - a set of customers with demands and a set of vehicles with limited capacity, the goal is to minimize the total distance traveled by the vehicles while satisfying the demands of the customers.",
    "SP": "Scheduling Problem - a set of jobs with operations and a set of machines, the goal is to minimize the total makespan or total tardiness.",
}

