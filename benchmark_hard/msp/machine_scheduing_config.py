problem_type_dict = {
    "jssp": "Job Shop Scheduling Problem (JSSP)",
    "fjssp": "Flexible Job Shop Scheduling Problem (FJSSP)",
    "fssp": "Flow-Shop Scheduling Problem (FSSP)",
    "hfssp": "Hybrid Flow-shop Scheduling Problem (HFSSP)",
    "ossp": "Open Shop Scheduling Problem (OSSP)",
    "asp": "Assembly Scheduling Problem (ASP)",
}

problem_type_param = {
    "hfssp": {
        'machines_per_stage': {
            f'stage_{0}': 4,
            f'stage_{1}': 4,
            f'stage_{2}': 4,
        }
    }
}
