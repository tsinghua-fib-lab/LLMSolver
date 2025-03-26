from envs import MTVRPGenerator, MTVRPEnv

ovrp_data = [
    {
        "desc_split": (
                "Your after-school bus service needs to get kids home from school at <loc_school> efficiently every day.\n" +
                "Across the city, you've got designated pickup spots (<loc_spot>, etc.)."
                "Each pickup point needs to drop off a different number of child.\n" +
                "Your buses can each carry up to <capacity_buse> children safely.\n" +
                "The challenge? Create the smartest routes that:\n" +
                "\t✓ Never overload any bus.\n" +
                "\t✓ Minimize total driving distance.\n" +
                "\t✓ Let drivers finish at their last stop (no backtracking).\n"
        ),
        "data_template": {
            "locs": [[8.3144, 32.7492],
                     [1.3097, 35.0156],
                     [44.2244, 26.3332],
                     [49.2972, 49.0268],
                     [26.1826, 16.5157],
                     [44.8903, 41.3092],
                     [14.2201, 22.5614],
                     [29.1558, 40.7860],
                     [1.3097, 35.0156],
                     [44.2244, 26.3332],
                     [49.2972, 49.0268],
                     [26.1826, 16.5157],
                     [44.8903, 41.3092],
                     [1.3097, 35.0156],
                     [44.2244, 26.3332],
                     [49.2972, 49.0268],
                     [26.1826, 16.5157],
                     [44.8903, 41.3092],
                     [14.2201, 22.5614],
                     [29.1558, 40.7860],
                     [18.7545, 1.5324], ],
            "demand_linehaul": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "capacity": [20],
            "num_depot": [1],
            "open_route": [1]
        },
        "user_template": {
            "loc_school": [[8.3144, 32.7492], ],
            "loc_spot": [
                [1.3097, 35.0156],
                [44.2244, 26.3332],
                [49.2972, 49.0268],
                [26.1826, 16.5157],
                [44.8903, 41.3092],
                [14.2201, 22.5614],
                [29.1558, 40.7860],
                [1.3097, 35.0156],
                [44.2244, 26.3332],
                [49.2972, 49.0268],
                [26.1826, 16.5157],
                [44.8903, 41.3092],
                [1.3097, 35.0156],
                [44.2244, 26.3332],
                [49.2972, 49.0268],
                [26.1826, 16.5157],
                [44.8903, 41.3092],
                [14.2201, 22.5614],
                [29.1558, 40.7860],
                [18.7545, 1.5324], ],
            "demand_kid": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "capacity_buse": [20],
        },
        "label": "ovrp",
    },
    {
        "desc_split": (
                "Your furniture delivery service faces a daily logistics challenge: transporting sofas, dining sets, and other large items to customers' homes across the town.\n" +
                "The delivery starts at the factory (<loc_factory>), where trucks with strict capacity limits (<capacity_truck>).\n"
                "Each customers (<loc_customer>) has a different order weighing (<demand_order> kilograms).\n" +
                "Couriers pick up their assigned loads (never exceeding their vehicle's capacity) and follow optimized routes to drop off all packages.\n" +
                "After that final drop-off, drivers are done for the day - no wasting time or gas driving back to base.\n" +
                "The goal is to plan routes that deliver everything without overloading any truck, hit every stop efficiently so drivers aren't wasting miles.\n" +
                "it's all about saving time, gas money and keeping customers happy with prompt deliveries.\n"
        ),
        "data_template": {
            "locs": [[50.8367, 45.2838],
                     [27.7630, 45.8388],
                     [57.6376, 40.9865],
                     [7.6372, 66.9145],
                     [34.0745, 72.6432],
                     [62.2890, 55.1624],
                     [37.7084, 33.1914],
                     [37.5137, 19.0591],
                     [14.4246, 65.7853],
                     [14.3931, 40.4804],
                     [4.1217, 44.6961],
                     [18.6837, 6.4944],
                     [61.8608, 56.1496],
                     [58.5419, 76.6527],
                     [16.7603, 21.8249]],
            "demand_linehaul": [1.3333, 2.0000, 5.3333, 3.3333, 1.3333, 0.6667, 1.3333, 0.6667, 3.3333,
                                4.6667, 6.0000, 2.6667, 6.0000, 2.6667],
            "capacity": [10],
            "num_depot": [1],
            "open_route": [1]
        },
        "user_template": {
            "loc_factory": [[50.8367, 45.2838], ],
            "loc_customer": [
                [27.7630, 45.8388],
                [57.6376, 40.9865],
                [7.6372, 66.9145],
                [34.0745, 72.6432],
                [62.2890, 55.1624],
                [37.7084, 33.1914],
                [37.5137, 19.0591],
                [14.4246, 65.7853],
                [14.3931, 40.4804],
                [4.1217, 44.6961],
                [18.6837, 6.4944],
                [61.8608, 56.1496],
                [58.5419, 76.6527],
                [16.7603, 21.8249]],
            "demand_order": [1.3333, 2.0000, 5.3333, 3.3333, 1.3333, 0.6667, 1.3333, 0.6667, 3.3333,
                             4.6667, 6.0000, 2.6667, 6.0000, 2.6667],
            "capacity_truck": [10],
        },
        "label": "ovrp",
    },
    {
        "desc_split": (
            "As a postal company, you need to efficiently assign mail carriers to deliver letters from your central office to various addresses (<loc_address>, etc.) from the base <loc_base>.\n" +
            "Each address letter weighs <demand_letter>\n",
            "Each carrier can only carry a limited load <capacity_load> and gets paid based on their travel distance.\n" +
            "The goal is to minimize your total costs by planning optimal routes that ensure all mail gets delivered without overloading any carrier.\n" +
            "Let them finish their day after the last drop-off at their final destination, eliminating unnecessary return trips to the office.\n"
        ),
        "data_template": {
            "locs": [[0.0826, 53.4963],
                     [16.8892, 11.7286],
                     [2.1138, 41.2318],
                     [24.4406, 36.7905],
                     [28.3453, 38.3198],
                     [52.0136, 1.0894],
                     [32.2170, 57.7848],
                     [42.7649, 47.4547],
                     [7.0387, 9.3687],
                     [28.1731, 19.9955],
                     [11.2233, 38.6417],
                     [55.4536, 23.1370],
                     [48.0130, 27.1515],
                     [26.8617, 42.5743],
                     [46.7104, 19.0671]],
            "demand_linehaul": [0.8000, 1.8000, 1.8000, 0.4000, 1.8000, 0.2000, 1.2000, 1.8000, 0.2000,
                                0.2000, 1.8000, 0.8000, 1.2000, 1.2000],
            "capacity": [4],
            "num_depot": [1],
            "open_route": [1]
        },
        "user_template": {
            "loc_base": [[0.0826, 53.4963], ],
            "loc_address": [
                [16.8892, 11.7286],
                [2.1138, 41.2318],
                [24.4406, 36.7905],
                [28.3453, 38.3198],
                [52.0136, 1.0894],
                [32.2170, 57.7848],
                [42.7649, 47.4547],
                [7.0387, 9.3687],
                [28.1731, 19.9955],
                [11.2233, 38.6417],
                [55.4536, 23.1370],
                [48.0130, 27.1515],
                [26.8617, 42.5743],
                [46.7104, 19.0671]],
            "demand_letter": [0.8000, 1.8000, 1.8000, 0.4000, 1.8000, 0.2000, 1.2000, 1.8000, 0.2000,
                              0.2000, 1.8000, 0.8000, 1.2000, 1.2000],
            "capacity_load": [4],
        },
        "label": "ovrp",
    },
    {
        "desc_split": (
                "During emergencies, ambulances need to deliver critical medical supplies from the hospital <loc_hospital> to multiple temporary care stations (<loc_station>, etc.), each requiring different items (<demand_item>).\n" +
                "Ambulances can only carry limited supplies (<capacity_ambulance> kg of medical equipment/drugs).\n"
                "After their last delivery, ambulances stay at that final location to assist, rather than returning to base.\n" +
                "The challenge is to plan routes that get all supplies where they're needed fastest, without overloading any ambulance, while minimizing total travel time to save more lives.\n"
        ),
        "data_template": {
            "locs": [[17.4788, 38.9278],
                     [40.3526, 43.2221],
                     [54.7079, 17.5786],
                     [26.5853, 1.5871],
                     [35.7721, 30.8734],
                     [6.8062, 36.1273],
                     [1.7587, 53.8079],
                     [17.9931, 35.3549],
                     [48.9843, 42.0241],
                     [33.6374, 44.4953],
                     [58.6593, 35.8314],
                     [44.7212, 44.2194],
                     [38.1232, 40.8865],
                     [21.8435, 4.1336],
                     [40.3153, 30.1194],
                     [57.2047, 19.3143],
                     [52.8897, 32.4708]],
            "demand_linehaul": [0.6667, 0.3333, 2.0000, 3.0000, 1.6667, 2.3333, 1.0000, 1.3333, 1.0000,
                                0.3333, 3.0000, 1.0000, 1.6667, 1.3333, 0.3333, 1.3333],
            "capacity": [5],
            "num_depot": [1],
            "open_route": [1]
        },
        "user_template": {
            "loc_hospital": [[17.4788, 38.9278], ],
            "loc_station": [
                [40.3526, 43.2221],
                [54.7079, 17.5786],
                [26.5853, 1.5871],
                [35.7721, 30.8734],
                [6.8062, 36.1273],
                [1.7587, 53.8079],
                [17.9931, 35.3549],
                [48.9843, 42.0241],
                [33.6374, 44.4953],
                [58.6593, 35.8314],
                [44.7212, 44.2194],
                [38.1232, 40.8865],
                [21.8435, 4.1336],
                [40.3153, 30.1194],
                [57.2047, 19.3143],
                [52.8897, 32.4708]],
            "demand_item": [0.6667, 0.3333, 2.0000, 3.0000, 1.6667, 2.3333, 1.0000, 1.3333, 1.0000,
                            0.3333, 3.0000, 1.0000, 1.6667, 1.3333, 0.3333, 1.3333],
            "capacity_ambulance": [5],
        },
        "label": "ovrp",
    },
    {
        "desc_split": (
                "A repair company handles service calls where customers provide their location (<loc_company>) and describe the issue.\n" +
                "The company checks what tools/parts are needed for each job (<demand_job>), then assigns technicians to carry those supplies from the office to multiple homes (<loc_home>) in one trip.\n" +
                "Workers can head straight home after their last repair instead of returning to base.\n" +
                "Please figure out the smartest way to group nearby jobs and plan routes so that:\n" +
                "\t- No technician hauls more tools than their van can hold (<capacity_tool>).\n" +
                "\t- All repairs get done in a single outing.\n" +
                "\t- Total driving distance is as short as possible.\n" +
                "This means less time on the road, lower fuel costs, and faster service for customers.\n"
        ),
        "data_template": {
            "locs": [[46.3978, 74.7336],
                     [64.8050, 76.7873],
                     [13.1908, 11.6239],
                     [66.5339, 17.8907],
                     [9.7798, 73.5239],
                     [51.1981, 45.4647],
                     [32.0085, 58.0112],
                     [74.8517, 75.3714],
                     [10.1840, 57.5611],
                     [20.5903, 51.9892],
                     [5.6358, 57.6920],
                     [71.4825, 23.7918],
                     [34.6723, 22.9662],
                     [51.8234, 46.9302],
                     [44.4252, 3.3464],
                     [15.2707, 65.9410],
                     [65.3227, 56.3345]],
            "demand_linehaul": [0.6667, 0.3333, 2.3333, 1.3333, 3.0000, 2.0000, 2.6667, 2.0000, 2.0000,
                                3.0000, 2.0000, 2.6667, 2.0000, 1.6667, 0.6667, 3.0000],
            "capacity": [5],
            "num_depot": [1],
            "open_route": [1]
        },
        "user_template": {
            "loc_company": [[46.3978, 74.7336], ],
            "loc_home": [
                [64.8050, 76.7873],
                [13.1908, 11.6239],
                [66.5339, 17.8907],
                [9.7798, 73.5239],
                [51.1981, 45.4647],
                [32.0085, 58.0112],
                [74.8517, 75.3714],
                [10.1840, 57.5611],
                [20.5903, 51.9892],
                [5.6358, 57.6920],
                [71.4825, 23.7918],
                [34.6723, 22.9662],
                [51.8234, 46.9302],
                [44.4252, 3.3464],
                [15.2707, 65.9410],
                [65.3227, 56.3345]],
            "demand_job": [0.6667, 0.3333, 2.3333, 1.3333, 3.0000, 2.0000, 2.6667, 2.0000, 2.0000,
                           3.0000, 2.0000, 2.6667, 2.0000, 1.6667, 0.6667, 3.0000],
            "capacity_tool": [5],
        },
        "label": "ovrp",
    }
]

if __name__ == '__main__':
    problem_type = "ovrp"
    generator = MTVRPGenerator(num_loc=16, variant_preset=problem_type)

    env = MTVRPEnv(generator, check_solution=False)
    td_data = env.generator(1)
    td_data_dict = td_data.to_dict()
    for key in td_data_dict.keys():
        print(key)
        print((td_data_dict[key] * 80))
    print()
