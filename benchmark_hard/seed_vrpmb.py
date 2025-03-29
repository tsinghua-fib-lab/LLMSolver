from envs.mtdvrp import MTVRPGenerator, MTVRPEnv

vrpmb_data_seed = [
    {
        "title": "Route Planning for Fleet Logistics",
        "desc_split": (
                "A company has a fleet of trucks that need to deliver goods to some customers and pick up goods from others.\n" +
                "Each truck in its route can either choose to deliver or to pick up in any order.\n" +
                "The challenge is figuring out the best routes for trucks so that they cover all the customers while keeping travel costs as low as possible.\n" +
                "Each truck with capacity <capacity> starts at a central warehouse, makes deliveries to certain locations, and then picks up goods from other locations before returning to the warehouse.\n" +
                "The locations of the central warehouse and the 20 customers is <locs>.\n" +
                "The capacity for each truck is <capacity>.\n" +
                "The delivery demand to all customers is <demand_linehaul>.\n" +
                "The pickup demand to all customers is <demand_linehaul>.\n" +
                "I need to decide which customers each truck should visit and in what order, while ensuring that no truck exceeds its capacity and that all customers are served efficiently.\n" +
                "I’m looking for a way to optimize this whole process so I can save fuel, reduce travel time, and make sure everything runs smoothly.\n"
        ),
        "data_template": {
            "locs": [[0.19151945, 0.62210876],
                     [0.43772775, 0.78535861],
                     [0.77997583, 0.2725926],
                     [0.27646425, 0.80187219],
                     [0.95813936, 0.87593263],
                     [0.35781726, 0.5009951],
                     [0.68346292, 0.71270204],
                     [0.37025076, 0.56119621],
                     [0.50308317, 0.01376845],
                     [0.77282661, 0.8826412],
                     [0.36488599, 0.6153962],
                     [0.07538124, 0.36882401],
                     [0.9331401, 0.65137815],
                     [0.39720258, 0.78873014],
                     [0.31683612, 0.56809866],
                     [0.86912739, 0.43617341],
                     [0.80214763, 0.14376682],
                     [0.70426095, 0.70458132],
                     [0.21879211, 0.92486763],
                     [0.44214076, 0.90931594],
                     [0.05980922, 0.18428709]
                     ],
            "demand_linehaul": [
                0.0, 0.0, 0.26666668, 0.0, 0.16666667, 0.1, 0.13333334, 0.2, 0.30000001,
                0.16666667, 0.30000001, 0.30000001, 0.13333334, 0.13333334, 0.1, 0.0, 0.1,
                0.30000001, 0.26666668, 0.0
            ],
            "demand_backhaul": [
                0.2, 0.2, 0.0, 0.30000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.16666667, 0.0, 0.0, 0.0, 0.26666668
            ],
            "capacity": [1],
            "num_depot": [1],
            "backhaul_class": [2],
        },
        "label": "vrpmb"
    },
    {
        "title": "Route Optimization for Logistics Fleets",
        "desc_split": (
                "I run a logistics company, and every day we send out trucks to handle both deliveries and pickups.\n" +
                "Deliveries and pickups can be mixed along the routs.\n" +
                "Some customers are waiting for shipments, while others have items that need to be returned to our warehouse.\n" +
                "The locations of the central warehouse where trucks with capacity <capacity> start and the 20 customers is <locs>.\n" +
                "The delivery demand to all customers is <demand_linehaul>.\n" +
                "The pickup demand to all customers is <demand_linehaul>.\n" +
                "I need to figure out the best way to assign trucks to customers and determine the most efficient routes so that deliveries happen first, pickups happen afterward, and everything gets done with minimal cost and travel time.\n"
        ),
        "data_template": {
            "locs": [[0.7949, 0.5138],
                     [0.0748, 0.5360],
                     [0.4732, 0.3048],
                     [0.7463, 0.9480],
                     [0.3961, 0.5864],
                     [0.1447, 0.3196],
                     [0.5676, 0.9172],
                     [0.1293, 0.1230],
                     [0.1592, 0.0266],
                     [0.0643, 0.1583],
                     [0.4790, 0.3431]],
            "demand_linehaul": [0.0333, 0.1667, 0.2000, 0.2333, 0.2333, 0.1333, 0.0000, 0.0667, 0.0000, 0.0000],
            "demand_backhaul": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2000, 0.0000, 0.1000, 0.0333],
            "capacity": [1],
            "num_depot": [1],
            "backhaul_class": [2],
        },
        "label": "vrpmb"
    },
    {
        "title": "Grocery Distribution & Returns Logistics Optimization",
        "desc_split": (
                "I manage a distribution network for a grocery supplier, and every day we send trucks to restock stores and collect unsold or expired goods for disposal.\n" +
                "Each store has specific delivery needs, and some stores also have returns that must be brought back.\n" +
                "The locactions of my storage and all these stores is <locs>, the restock demand is <demand_linehaul>, and the collect demand is <demand_backhaul>.\n" +
                "The truck is of capacity <capacity>.\n" +
                "I need to figure out an optimized plan that ensures all stores get their supplies on time while also collecting returns efficiently, keeping transportation costs low, and ensuring trucks don’t exceed their capacity.\n"
        ),
        "data_template": {
            "locs": [[0.8473, 0.6327],
                     [0.1675, 0.4364],
                     [0.5611, 0.8788],
                     [0.8859, 0.6644],
                     [0.4538, 0.3096],
                     [0.5276, 0.8419],
                     [0.3764, 0.5898],
                     [0.7676, 0.2699],
                     [0.1130, 0.5286],
                     [0.4558, 0.4626],
                     [0.9267, 0.3047]],
            "demand_linehaul": [0.2333, 0.3000, 0.3000, 0.0667, 0.1333, 0.0000, 0.1333, 0.3000, 0.0333, 0.1000],
            "demand_backhaul": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2667, 0.0000, 0.0000, 0.0000, 0.0000],
            "capacity": [1],
            "num_depot": [1],
            "backhaul_class": [2],
        },
        "label": "vrpmb"
    },
    {
        "title": "Medical Supply Delivery & Medical Waste Reverse Logistics Optimization",
        "desc_split": (
                "I run a medical supply distribution for hospitals and clinics.\n" +
                "Every day, our trucks transport essential medical equipment, medications, and supplies to healthcare facilities.\n" +
                "At the same time, they need to collect expired drugs for proper disposal.\n" +
                "The locactions of the warehouse and all these target spots is <locs>, the restock demand is <demand_linehaul>, and the collect demand is <demand_backhaul>.\n" +
                "The truck is of capacity <capacity>.\n" +
                "I need an optimized routing plan that ensures all hospitals receive their supplies on time while also handling pickups efficiently, minimizing travel costs, and complying with medical transportation regulations.\n"
        ),
        "data_template": {
            "locs": [],
            "demand_linehaul": [],
            "demand_backhaul": [],
            "capacity": [1],
            "num_depot": [1],
            "backhaul_class": [2],
        },
        "label": "vrpmb"
    },
    {
        "title": "Beverage Distribution & Empty Container Return Logistics Optimization",
        "desc_split": (
                "I manage a beverage distribution business that supplies restaurants and bars with drinks while also collecting empty kegs and bottles for reuse.\n" +
                "Each truck leaves with a full load of beverages, delivers them to customers, and also picks up empty containers on the way back.\n" +
                "The locactions of the warehouse and all these target spots is <locs>, the restock demand is <demand_linehaul>, and the collect demand is <demand_backhaul>.\n" +
                "The truck is of capacity <capacity>.\n" +
                "I need to figure out the best way to assign delivery and pickup stops to each truck, ensuring that every customer gets their order on time while also efficiently collecting returns—all while minimizing fuel costs and travel time.\n"
        ),
        "data_template": {
            "locs": [[0.6554, 0.2246],
                     [0.0541, 0.2184],
                     [0.6678, 0.6608],
                     [0.0982, 0.1884],
                     [0.2524, 0.6683],
                     [0.1731, 0.1134],
                     [0.2169, 0.4563],
                     [0.9667, 0.9699],
                     [0.0306, 0.4067],
                     [0.2354, 0.5070],
                     [0.1530, 0.5415]],
            "demand_linehaul": [0.1667, 0.0000, 0.1667, 0.3000, 0.3000, 0.0333, 0.0000, 0.0667, 0.1667,
                                0.0000],
            "demand_backhaul": [0.0000, 0.2667, 0.0000, 0.0000, 0.0000, 0.0000, 0.2000, 0.0000, 0.0000,
                                0.3000],
            "capacity": [1],
            "num_depot": [1],
            "backhaul_class": [2],
        },
        "label": "vrpmb"
    },

]

if __name__ == '__main__':
    problem_type = "vrpb"
    generator = MTVRPGenerator(num_loc=15, variant_preset=problem_type)

    env = MTVRPEnv(generator, check_solution=False)
    td_data = env.generator(1)
    td_data_dict = td_data.to_dict()
    for key in td_data_dict.keys():
        print(key)
        print((td_data_dict[key]))
    print()
