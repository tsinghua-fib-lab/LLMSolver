from envs.cvrp import MTVRPGenerator, MTVRPEnv

cvrp_data_seed = [
    {
        "title": "Self-Service Drink Machines Restocking Problem",
        "desc_split": (
                "A beverage company has set up many self-service drink machines across the city and needs to restock them regularly.\n" +
                "They use trucks (with a <capacity> capacity) from a central warehouse <loc_depot> to deliver drinks to these scattered machines.\n" +
                "Each machine has a specific location and a demand.\n" +
                "Knowing the exact <num_customer> locations (<loc_customer>) and demands (<demand>), the company wants to plan fuel-efficient routes without overloading any truck.\n" +
                "The goal is to minimize the total gas bill by reducing unnecessary driving while ensuring each machine is restocked exactly once.\n"
        ),
        "data_template": {
            "locs": [[9.5368, 0.0341],
                     [0.6264, 1.1813],
                     [1.9398, 4.9717],
                     [1.4026, 6.6282],
                     [0.3915, 7.0841],
                     [1.4657, 4.1706],
                     [3.9588, 2.4021],
                     [0.9185, 7.7860],
                     [7.3407, 1.8473],
                     [4.0074, 5.4404],
                     [0.0173, 9.4268]],
            "demand_linehaul": [1.6667, 1.3333, 3.0000, 0.3333, 1.6667, 2.6667, 2.3333, 1.0000, 3.0000, 1.6667],
            "capacity": [10],
            "num_depot": [1]
        },
        "user_template": {
            "loc_depot": [[9.5368, 0.0341]],
            "loc_customer": [[0.6264, 1.1813],
                             [1.9398, 4.9717],
                             [1.4026, 6.6282],
                             [0.3915, 7.0841],
                             [1.4657, 4.1706],
                             [3.9588, 2.4021],
                             [0.9185, 7.7860],
                             [7.3407, 1.8473],
                             [4.0074, 5.4404],
                             [0.0173, 9.4268]],
            "demand": [1.6667, 1.3333, 3.0000, 0.3333, 1.6667, 2.6667, 2.3333, 1.0000, 3.0000, 1.6667],
            "num_customer": [10],
            "capacity": [10],
        },
        "label": "cvrp",
    },
    {
        "title": "Book Distribution Center Delivery Problem",
        "desc_split": (
                "The book distribution center at <loc_depot> needs to deliver orders to <num_customer> bookstores across the city using vans with a <capacity> ton capacity.\n" +
                "Each store is at a unique location (<loc_customer>) and requires a different amount of books (<demand> ton).\n" +
                "The delivery team has to organize their vans schedules so every store gets their book shipments, but no truck ever carries more than it's supposed to.\n" +
                "They hope each store is visited just once and the total distance traveled by all vans is minimized.\n" +
                "Since all the bookstores operate 24/7, the delivery teams don't need to worry about timing their shipments.\n" +
                "They just need to focus on efficiently loading the trucks without exceeding weight limits and planning the shortest possible routes between stores.\n"
        ),
        "data_template": {
            "locs": [[6.5288, 11.4193],
                     [4.5300, 7.8018],
                     [9.9467, 6.2894],
                     [10.5737, 4.6932],
                     [8.4988, 6.2468],
                     [9.8155, 14.5357],
                     [14.3001, 2.0244],
                     [6.6016, 5.5725],
                     [8.5811, 1.6571]],
            "demand_linehaul": [0.2333, 0.1000, 0.2000, 0.0667, 0.2667, 0.2667, 0.2667, 0.1333],
            "capacity": [1],
            "num_depot": [1]
        },
        "user_template": {
            "loc_depot": [6.5288, 11.4193],
            "loc_customer": [[4.5300, 7.8018],
                             [9.9467, 6.2894],
                             [10.5737, 4.6932],
                             [8.4988, 6.2468],
                             [9.8155, 14.5357],
                             [14.3001, 2.0244],
                             [6.6016, 5.5725],
                             [8.5811, 1.6571]],
            "num_customer": [8],
            "demand": [0.2333, 0.1000, 0.2000, 0.0667, 0.2667, 0.2667, 0.2667, 0.1333],
            "capacity": [1],
        },
        "label": "cvrp",
    },
    {
        "title": "Emergency Supplies Delivery Problem",
        "desc_split": (
                "After a disaster strikes mountainous areas at locations <loc_customer>, each with urgent needs like medical kits, food packs, and blankets (<demand>).\n" +
                "Rescue teams start from their base at <loc_depot> with vehicles that can each carry up to <capacity> ton of supplies."
                "Rescue teams have to figure out how to share their limited supplies smartly.\n" +
                "They need to get help to every single spot, even the hard-to-reach ones, without wasting time so they can save as many lives as possible.\n" +
                "Every minute counts to reach everyone, especially in remote areas.\n"
        ),
        "data_template": {
            "locs": [[45.1628, 0.3389],
                     [4.4140, 45.1586],
                     [10.8706, 36.0827],
                     [9.0352, 46.3730],
                     [42.7484, 1.4545],
                     [46.2215, 39.2451],
                     [29.7302, 32.7303],
                     [45.5001, 21.1496],
                     [34.5360, 23.4735],
                     [46.9857, 28.8684],
                     [36.8961, 9.2650],
                     [16.1803, 8.5975],
                     [1.4455, 27.8106],
                     [31.2922, 11.6886],
                     [37.8205, 42.3249],
                     [35.4748, 38.4581],
                     [2.3681, 1.2359],
                     [3.0294, 5.7156],
                     [17.4340, 1.6115],
                     [8.7506, 42.5734],
                     [13.0304, 46.1875]],
            "demand_backhaul": [0.5000, 1.0000, 1.0000, 1.3333, 0.6667, 1.0000, 1.1667, 1.0000, 0.6667,
                                0.3333, 1.3333, 1.1667, 0.6667, 1.1667, 1.1667, 1.1667, 0.5000, 1.5000,
                                0.5000, 0.6667],
            "capacity": [5],
            "num_depot": [1]
        },
        "user_template": {
            "loc_depot": [[45.1628, 0.3389]],
            "loc_customer": [[4.4140, 45.1586],
                             [10.8706, 36.0827],
                             [9.0352, 46.3730],
                             [42.7484, 1.4545],
                             [46.2215, 39.2451],
                             [29.7302, 32.7303],
                             [45.5001, 21.1496],
                             [34.5360, 23.4735],
                             [46.9857, 28.8684],
                             [36.8961, 9.2650],
                             [16.1803, 8.5975],
                             [1.4455, 27.8106],
                             [31.2922, 11.6886],
                             [37.8205, 42.3249],
                             [35.4748, 38.4581],
                             [2.3681, 1.2359],
                             [3.0294, 5.7156],
                             [17.4340, 1.6115],
                             [8.7506, 42.5734],
                             [13.0304, 46.1875]],
            "demand": [0.5000, 1.0000, 1.0000, 1.3333, 0.6667, 1.0000, 1.1667, 1.0000, 0.6667,
                       0.3333, 1.3333, 1.1667, 0.6667, 1.1667, 1.1667, 1.1667, 0.5000, 1.5000,
                       0.5000, 0.6667],
            "capacity": [5],
        },
        "label": "cvrp",
    },
    {
        "title": "Restaurant Chain Delivery Problem",
        "desc_split": (
                "You run a restaurant chain with a central kitchen <loc_depot> supplying <num_customer> restaurants.\n" +
                "Your central kitchen needs to get fresh food out to all locations using trucks (<capacity>) that can only carry so much.\n" +
                "Each restaurant at <loc_customer> has its specific daily order (<demand>) - some need more ingredients, others less.\n" +
                "You've got to plan routes that:\n" +
                "\t✓ Deliver exactly what each spot needs.\n" +
                "\t✓ Never pack trucks beyond their limit.\n" +
                "\t✓ Cover all stops in the shortest distance.\n" +
                "\t✓ Keep everything fresh and on time.\n" +
                "The smarter your routes, the fresher the meals and the more you save on gas and time!\n"
        ),
        "data_template": {
            "locs": [[77.4377, 95.7173],
                     [95.7230, 56.1522],
                     [71.1036, 76.5780],
                     [28.9117, 57.0789],
                     [37.3356, 3.4924],
                     [7.0284, 13.0125],
                     [68.1048, 83.4480],
                     [49.2185, 23.4884],
                     [95.8750, 96.1848],
                     [5.5720, 98.2724],
                     [58.8317, 97.2547],
                     [99.5067, 42.8537],
                     [23.8334, 75.5441],
                     [97.1866, 43.3411]],
            "demand_linehaul": [0.2667, 1.6000, 0.5333, 0.2667, 0.2667, 2.4000, 1.8667, 1.3333, 0.2667,
                                1.0667, 0.2667, 1.3333, 2.4000],
            "capacity": [4],
            "num_depot": [1]
        },
        "user_template": {
            "loc_depot": [[77.4377, 95.7173], ],
            "loc_customer": [
                [95.7230, 56.1522],
                [71.1036, 76.5780],
                [28.9117, 57.0789],
                [37.3356, 3.4924],
                [7.0284, 13.0125],
                [68.1048, 83.4480],
                [49.2185, 23.4884],
                [95.8750, 96.1848],
                [5.5720, 98.2724],
                [58.8317, 97.2547],
                [99.5067, 42.8537],
                [23.8334, 75.5441],
                [97.1866, 43.3411]],
            "demand": [0.2667, 1.6000, 0.5333, 0.2667, 0.2667, 2.4000, 1.8667, 1.3333, 0.2667,
                       1.0667, 0.2667, 1.3333, 2.4000],
            "num_customer": [13],
            "capacity": [4],
        },
        "label": "cvrp",
    },
    {
        "title": "Shared Bikes Distribution Problem",
        "desc_split": (
                "When rolling out shared bikes across the city, you'll need to carefully distribute them to different neighborhoods (<loc_customer>, etc.), with each area getting just the right number of bikes (<demand>, etc.).\n" +
                "Starting from the base at <loc_depot>, delivery vehicles (each holding up to <capacity>) must transport supplies to multiple locations." +
                "The real challenge? Planning the most efficient delivery routes for your trucks to drop off all these bikes.\n" +
                "you'll want to figure out the smartest transportation sequence to minimize fuel costs and time spent on the road, while making sure every location gets exactly the bikes it needs without overloading any delivery vehicle.\n"
        ),
        "data_template": {
            "locs": [[39.9011, 74.1811],
                     [64.8425, 35.6192],
                     [23.6588, 96.6388],
                     [20.6195, 14.0243],
                     [45.2912, 6.5895],
                     [2.8569, 18.2382],
                     [82.4929, 18.1094],
                     [74.1189, 81.6149],
                     [15.5185, 76.1067],
                     [20.4280, 12.6284],
                     [92.4714, 72.7380],
                     [17.2763, 37.9009],
                     [23.7733, 50.5108],
                     [40.8286, 79.9127],
                     [0.2983, 65.3552],
                     [39.9923, 75.5846],
                     [18.3114, 17.0106],
                     [15.0964, 23.5309],
                     [35.0018, 98.8764]],
            "demand_linehaul": [1.6667, 3.0000, 2.0000, 2.0000, 0.6667, 2.0000, 2.3333, 1.0000, 3.0000,
                                2.3333, 2.3333, 1.0000, 1.3333, 3.0000, 1.3333, 2.3333, 2.6667, 0.3333],
            "capacity": [4],
            "num_depot": [1]
        },
        "user_template": {
            "loc_depot": [[39.9011, 74.1811], ],
            "loc_customer": [
                [64.8425, 35.6192],
                [23.6588, 96.6388],
                [20.6195, 14.0243],
                [45.2912, 6.5895],
                [2.8569, 18.2382],
                [82.4929, 18.1094],
                [74.1189, 81.6149],
                [15.5185, 76.1067],
                [20.4280, 12.6284],
                [92.4714, 72.7380],
                [17.2763, 37.9009],
                [23.7733, 50.5108],
                [40.8286, 79.9127],
                [0.2983, 65.3552],
                [39.9923, 75.5846],
                [18.3114, 17.0106],
                [15.0964, 23.5309],
                [35.0018, 98.8764]],
            "demand": [1.6667, 3.0000, 2.0000, 2.0000, 0.6667, 2.0000, 2.3333, 1.0000, 3.0000,
                       2.3333, 2.3333, 1.0000, 1.3333, 3.0000, 1.3333, 2.3333, 2.6667, 0.3333],
            "capacity": [4],
        },
        "label": "cvrp",
    }
]

if __name__ == '__main__':
    problem_type = "cvrp"
    generator = MTVRPGenerator(num_loc=18, variant_preset=problem_type)

    env = MTVRPEnv(generator, check_solution=False)
    td_data = env.generator(1)
    td_data_dict = td_data.to_dict()
    for key in td_data_dict.keys():
        print(key)
        print(td_data_dict[key])
    print()
