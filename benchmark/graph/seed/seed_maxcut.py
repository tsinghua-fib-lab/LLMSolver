maxcut_description = """
### **Max-Cut Problem**
The Max-Cut problem aims to partition the nodes of a given graph into two disjoint sets such that the number (or total weight) of edges between the two sets is maximized. 
In this setting, each edge is considered "cut" if its endpoints lie in different sets. 
The goal is to find a division that maximizes the total number of cut edges, which is important in applications like network design, clustering, and circuit layout optimization.
"""

maxcut_data_seed = [
    {
        "title": "Circuit Design: Minimizing Wire Crossings",
        "desc_split": "In a VLSI circuit represented as <graph>, components are nodes and wires are edges. By partitioning the components into two groups to maximize the number of wires running between them, designers can simplify layout and reduce congestion within each group.",
        "label": "maxcut",
    },
    {
        "title": "Political District Division",
        "desc_split": "In a political network modeled as <graph>, where nodes represent regions and edges indicate strong social or economic ties, planners can divide the regions into two groups in a way that maximizes the number of ties crossing between groups, promoting balance and reducing internal conflicts.",
        "label": "maxcut",
    },
    {
        "title": "Data Clustering for Load Balancing",
        "desc_split": "In a distributed system modeled as <graph>, servers are nodes and high-traffic connections are edges. Splitting the servers into two groups while maximizing cross-connections ensures that heavily interacting components are separated, leading to balanced load distribution across systems.",
        "label": "maxcut",
    },
    {
        "title": "Sports League Scheduling",
        "desc_split": "In a sports competition network <graph>, teams are nodes and edges represent rivalry intensity. Organizing teams into two divisions by maximizing rivalries across divisions enhances competitive balance and fan engagement throughout the season.",
        "label": "maxcut",
    },
    {
        "title": "Social Conflict Analysis",
        "desc_split": "In a social tension graph <graph>, individuals are nodes and edges represent conflicts. Separating individuals into two groups to maximize the number of conflicts between groups minimizes intra-group tension, leading to more harmonious internal group dynamics.",
        "label": "maxcut",
    },
]
maxcut_data_tag_list = ['graph', 'num_nodes', 'num_edges']
