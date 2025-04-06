from envs import MTVRPGenerator, MTVRPEnv

vrptw_data_seed = [
    {
        "title": "Unmanned Aerial Vehicle Delivery Problem",
        "desc_split": (
                """
                A company needs to use drones for parcel delivery. Specifically, given a set of delivery locations relative to the launch site, 
                the coordinates of the start point is <loc_depot> and the delivery points are provided as <locs>. Each drone has a speed of 2 km/h and a maximum endurance time 
                of 1.2676 hours. Before the battery is depleted, each drone must return to the launch site to recharge.
                Please plan the flight paths of the drones such that:
                1. Each customer receives their parcel.
                2. Each drone can safely return to the launch site.
                3. The total flight distance of the drones is minimized.
                """
        ),
        "data_template": {
            "locs": [[0.8377, 0.8782],
                    [0.1830, 0.1320],
                    [0.0445, 0.0418],
                    [0.1386, 0.1537],
                    [0.5456, 0.8554],
                    [0.4450, 0.7329],
                    [0.6567, 0.8046],
                    [0.7094, 0.3489],
                    [0.6698, 0.4157],
                    [0.6090, 0.7695],
                    [0.8741, 0.2506],
                    [0.5968, 0.3619],
                    [0.1669, 0.0233],
                    [0.6307, 0.9386],
                    [0.5203, 0.8845],
                    [0.9748, 0.9545],
                    [0.7378, 0.5791],
                    [0.1089, 0.4648],
                    [0.6210, 0.4947],
                    [0.9399, 0.8094],
                    [0.7146, 0.5329]],
            "speed": [1],
            "distance_limit": [2.5351],
            "num_depot": [1],
        },
        "user_template": {
            "loc_depot": [[0.8377, 0.8782],],
            "loc_customer": [[0.1830, 0.1320],
                    [0.0445, 0.0418],
                    [0.1386, 0.1537],
                    [0.5456, 0.8554],
                    [0.4450, 0.7329],
                    [0.6567, 0.8046],
                    [0.7094, 0.3489],
                    [0.6698, 0.4157],
                    [0.6090, 0.7695],
                    [0.8741, 0.2506],
                    [0.5968, 0.3619],
                    [0.1669, 0.0233],
                    [0.6307, 0.9386],
                    [0.5203, 0.8845],
                    [0.9748, 0.9545],
                    [0.7378, 0.5791],
                    [0.1089, 0.4648],
                    [0.6210, 0.4947],
                    [0.9399, 0.8094],
                    [0.7146, 0.5329]],
            "speed": [2],
            "max_flight_time": [1.2676],
            "num_depot": [1],
        },
        "label": "vrpl",
    },

    
    {
        "title": "Optimizing Research Team Routes for Polar Biological Sampling in Antarctica with Distance and Time Constraints",
        "desc_split": (
                "Our Antarctic scientific research team needs to conduct polar biological sampling tasks at multiple locations. The coordinates of base camp and the sampling task locations are <locs>." +\
                "The research team can dispatch multiple research teams, each advancing at a speed of 5 kilometers per day, but they only have enough food to support them for 3 days. " +\
                "They must return to the base camp before the food runs out. " +\
                "Please plan the number of research teams and routes to ensure that each team can safely return to the base camp, " +\
                "that each sampling location is sampled by at least one team, and to minimize the total travel distance for all teams."
        ),
        "data_template": {
            "locs": [[0.5056, 0.3433],
                    [0.0334, 0.6224],
                    [0.9954, 0.8627],
                    [0.1140, 0.0617],
                    [0.1329, 0.9319],
                    [0.8296, 0.1476],
                    [0.1184, 0.6619],
                    [0.9143, 0.3657],
                    [0.8786, 0.9539],
                    [0.2329, 0.5766],
                    [0.7101, 0.2987],
                    [0.4786, 0.3366],
                    [0.6947, 0.7070],
                    [0.3539, 0.2064],
                    [0.2585, 0.2720],
                    [0.3676, 0.8714],
                    [0.6844, 0.9874],
                    [0.2704, 0.6790],
                    [0.1420, 0.4048],
                    [0.4274, 0.1739],
                    [0.3531, 0.6503]],
            "speed": [1],
            "distance_limit": [15.0],
            "num_depot": [1],
        },
        "user_template": {
            "corrdinates": [[505.6, 343.3],
                    [33.4, 622.4],
                    [995.4, 862.7],
                    [114.0, 61.7],
                    [132.9, 931.9],
                    [829.6, 147.6],
                    [118.4, 661.9],
                    [914.3, 365.7],
                    [878.6, 953.9],
                    [232.9, 576.6],
                    [710.1, 298.7],
                    [478.6, 336.6],
                    [694.7, 707.0],
                    [353.9, 206.4],
                    [258.5, 272.0],
                    [367.6, 871.4],
                    [684.4, 987.4],
                    [270.4, 679.0],
                    [142.0, 404.8],
                    [427.4, 173.9],
                    [353.1, 650.3]],
            "speed": [5.0],
            "max_explore": [3.0],
            "num_depot": [1],
        },
        "label": "vrpl",
    },
    {
        "title": "Designing an Irrigation System for a Large Farm",
        "desc_split": (
            "You need to design an irrigation system for a large farm. The system draws water from a well and distributes it to various fields through pipes.\n" +
            "For maintenance purposes, each pipe must form a loop, starting from the well and returning to it, with a maximum length of <max_length> per loop.\n" +
            "The well is located at <loc_well>, and the fields requiring irrigation are located at <loc_lands>.\n" +
            "Please design the irrigation system to ensure that each field has access to water while minimizing the total length of all pipes."
        ),
        "data_template": {
            "locs": [[0.4395, 0.2900],
                    [0.3850, 0.7146],
                    [0.8491, 0.9639],
                    [0.8616, 0.2484],
                    [0.4244, 0.3050],
                    [0.9191, 0.5160],
                    [0.7415, 0.8086],
                    [0.8831, 0.6517],
                    [0.5524, 0.1347],
                    [0.8555, 0.1541],
                    [0.1219, 0.1892],
                    [0.2643, 0.6324],
                    [0.3643, 0.9438],
                    [0.7493, 0.1549],
                    [0.6434, 0.0524],
                    [0.7780, 0.3403],
                    [0.3521, 0.6932],
                    [0.5879, 0.8368],
                    [0.1261, 0.7342],
                    [0.8222, 0.0669],
                    [0.9104, 0.4225]],
            "distance_limit": [1.8319],
            "num_depot": [1],
        },
        "user_template": {
            "loc_well": [[439.5, 290.0],],
            "loc_lands": [[38.50, 71.46],
                    [84.91, 96.39],
                    [86.16, 24.84],
                    [42.44, 30.50],
                    [91.91, 51.60],
                    [74.15, 80.86],
                    [88.31, 65.17],
                    [55.24, 13.47],
                    [85.55, 15.41],
                    [12.19, 18.92],
                    [26.43, 63.24],
                    [36.43, 94.38],
                    [74.93, 15.49],
                    [64.34, 5.24],
                    [77.80, 34.03],
                    [35.21, 69.32],
                    [58.79, 83.68],
                    [12.61, 73.42],
                    [82.22, 6.69],
                    [91.04, 42.25]],
            "max_length": [1.8319],
        },
        "label": "vrpl",
    },
    {
        "title": "Power Company Signal Tower Inspection",
        "desc_split": (
                "Due to rain and snow, a power company must inspect several signal towers potentially impacted by the weather. Workers will drive from the depot <loc_depot> to inspect the towers, located at <locs>.\n" +
                "Since the towers are in mountainous areas with no gas stations along the route, workers must return to the depot before running out of fuel.\n" +
                "A fully fueled service vehicle can travel <max_distance> km.\n" +
                "Please plan a route for the power company to ensure all signal towers are inspected and the workers can safely return to the depot."
        ),
        "data_template": {
            "locs": [[0.6948, 0.7903],
                    [0.7892, 0.6657],
                    [0.2843, 0.7519],
                    [0.6807, 0.6699],
                    [0.5470, 0.5642],
                    [0.2516, 0.2800],
                    [0.3870, 0.4634],
                    [0.6323, 0.7678],
                    [0.9159, 0.9478],
                    [0.3654, 0.2344],
                    [0.2839, 0.2107],
                    [0.4521, 0.5311],
                    [0.1325, 0.3887],
                    [0.6672, 0.9279],
                    [0.2763, 0.6119],
                    [0.3304, 0.8566],
                    [0.4241, 0.7897],
                    [0.8184, 0.7376],
                    [0.9975, 0.8508],
                    [0.4626, 0.5961],
                    [0.3569, 0.3570]],
            "distance_limit": [2.1349],
            "num_depot": [1],
        },
        "user_template": {
            "loc_depot": [[694.8, 790.3]],
            "loc_tower": [[789.2, 665.7],
                    [284.3, 751.9],
                    [680.7, 669.9],
                    [547.0, 564.2],
                    [251.6, 280.0],
                    [387.0, 463.4],
                    [632.3, 767.8],
                    [915.9, 947.8],
                    [365.4, 234.4],
                    [283.9, 210.7],
                    [452.1, 531.1],
                    [132.5, 388.7],
                    [667.2, 927.9],
                    [276.3, 611.9],
                    [330.4, 856.6],
                    [424.1, 789.7],
                    [818.4, 737.6],
                    [997.5, 850.8],
                    [462.6, 596.1],
                    [356.9, 357.0]],
            "max_distance": [213.49],
        },
        "label": "vrpl",
    },
    {
        "title": "School Bus Route Optimization",
        "desc_split": (    
            "You need to plan a route for a school bus to drop off all students at their homes and return to the school without refueling.\n" +
            "The school is located at <school>, and the students' homes are located at <homes>.\n" +
            "The school bus has a maximum range of 200 km when fully fueled.\n" +
            "Please design a route to ensure all students are delivered home and the bus returns to the school within the fuel limit."
        ),
        "data_template": {
            "locs": [[0.6407, 0.1983],
                    [0.9578, 0.2955],
                    [0.4054, 0.3158],
                    [0.2715, 0.7087],
                    [0.3629, 0.0925],
                    [0.0966, 0.8107],
                    [0.7319, 0.6801],
                    [0.7745, 0.7688],
                    [0.5018, 0.6560],
                    [0.5592, 0.4259],
                    [0.0475, 0.4659],
                    [0.6600, 0.3493],
                    [0.5077, 0.9083],
                    [0.1145, 0.0942],
                    [0.5544, 0.9959],
                    [0.7356, 0.8210],
                    [0.8376, 0.6715],
                    [0.8774, 0.0322],
                    [0.2165, 0.8597],
                    [0.6959, 0.8090],
                    [0.3618, 0.6036]],
            "distance_limit": [2.3625],
        },
        "user_template": {
            "num_students": [20],
            "school": [[640.7, 198.3]],
            "homes": [[957.8, 295.5],
                    [405.4, 315.8],
                    [271.5, 708.7],
                    [362.9, 92.5],
                    [96.6, 810.7],
                    [731.9, 680.1],
                    [774.5, 768.8],
                    [501.8, 656.0],
                    [559.2, 425.9],
                    [475.0, 465.9],
                    [660.0, 349.3],
                    [507.7, 908.3],
                    [114.5, 94.2],
                    [554.4, 995.9],
                    [735.6, 821.0],
                    [837.6, 671.5],
                    [877.4, 32.2],
                    [216.5, 859.7],
                    [695.9, 809.0],
                    [361.8, 603.6]],
            "max_distance": [236.25],
            "max_speed": [120],
        },
        "label": "vrpl"
    }
]

if __name__ == '__main__':
    problem_type = "vrpl"
    generator = MTVRPGenerator(num_loc=1, variant_preset=problem_type)

    env = MTVRPEnv(generator, check_solution=False)
    td_data = env.generator(1)
    td_data_dict = td_data.to_dict()
    for key in td_data_dict.keys():
        print(key)
        print((td_data_dict[key]))
    print(len([]))
