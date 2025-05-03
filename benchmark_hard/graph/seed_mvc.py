mvc_description = """
### **Minimum Vertex Cover (MVC) Problem**
The Minimum Vertex Cover (MVC) problem seeks the smallest possible set of nodes in a graph such that every edge is incident to at least one selected node. 
In other words, the selected nodes "cover" all connections in the graph. 
The objective is to minimize the number of nodes chosen while ensuring complete coverage of edges, which is critical in areas like network security, resource allocation, and infrastructure monitoring.
"""

mvc_data_seed = [
    {
        "title": "Network Security: Monitoring Points",
        "desc_split": "In a computer network represented as <graph>, security devices must be placed to monitor all communication links. By selecting a minimal set of nodes where each link is connected to at least one monitored device, the system ensures complete coverage while minimizing hardware costs.",
        "label": "mvc",
    },
    {
        "title": "Power Grid Maintenance",
        "desc_split": "In an electricity grid modeled as <graph>, engineers must inspect every connection between substations. Choosing the smallest possible set of substations to station inspectors guarantees that all power lines are checked without redundant inspections.",
        "label": "mvc",
    },
    {
        "title": "Social Network Content Moderation",
        "desc_split": "In a social media graph <graph>, where nodes are users and edges are interactions, moderators need to monitor activity. Selecting a minimal group of key users ensures that all interactions are observed by at least one trusted party, reducing the effort while maintaining platform safety.",
        "label": "mvc",
    },
    {
        "title": "Urban Surveillance System",
        "desc_split": "A city street map can be modeled as <graph>, where intersections are nodes and streets are edges. To install surveillance cameras efficiently, the goal is to cover all streets by placing cameras at as few intersections as possible, guaranteeing that every street is under observation.",
        "label": "mvc",
    },
    {
        "title": "Task Dependency Management",
        "desc_split": "In a project management graph <graph>, tasks are nodes and dependencies are edges. To control workflow, managers must oversee critical tasks that impact others. Selecting a minimal set of tasks to monitor ensures that all dependencies are under supervision, enabling better project control.",
        "label": "mvc",
    },

]
mvc_data_tag_list = ['graph', 'num_nodes', 'num_edges']
