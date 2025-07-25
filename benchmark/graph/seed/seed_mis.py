mis_description = """
### **Maximal Independent Set (MIS) Problem**
The Maximal Independent Set (MIS) problem asks for a selection of nodes from a given graph such that no two selected nodes are directly connected, and no additional nodes can be added without violating this property. 
The goal is to find a set that is as large as possible without being extendable, ensuring that every unselected node is adjacent to at least one selected node. 
This concept is fundamental in applications where conflict-free selections are needed, such as scheduling, network design, and resource allocation.
"""

mis_data_seed = [
    {
        "title": "Wireless Sensor Network Channel Allocation",
        "desc_split": "In a wireless sensor network represented as <graph>, a group of sensor nodes must be selected so that none interfere with each other during data transmission. This selection ensures that as many non-conflicting nodes as possible can operate simultaneously, maximizing communication efficiency.",
        "label": "mis",
    },
    {
        "title": "Advertising Campaigns in Social Networks",
        "desc_split": "Given a social network graph <graph>, a marketing team needs to identify a set of users who are not directly connected. Selecting such users allows for independent promotion efforts, reducing overlapping influence among closely linked communities.",
        "label": "mis",
    },
    {
        "title": "Urban Traffic Light Scheduling",
        "desc_split": "In a city road network modeled by <graph>, intersections that are too close may conflict when releasing traffic simultaneously. Choosing a set of intersections that are not adjacent enables multiple green lights at once without congestion risks, improving overall traffic flow.",
        "label": "mis",
    },
    {
        "title": "Robotic Task Assignment in a Factory",
        "desc_split": "Inside a factory system represented by <graph>, robots are deployed across different areas. To prevent operational collisions, a set of robots must be chosen such that no two work in closely interfering zones at the same time, ensuring smooth and efficient task execution.",
        "label": "mis",
    },
    {
        "title": "Examination Scheduling for Students",
        "desc_split": "A university manages exam sessions using a relationship graph <graph>, where edges represent potential collaboration risks. By grouping students who have no direct connection, exams can be scheduled fairly, minimizing the chances of coordinated misconduct.",
        "label": "mis",
    },

]
mis_data_tag_list = ['graph', 'num_nodes', 'num_edges']
