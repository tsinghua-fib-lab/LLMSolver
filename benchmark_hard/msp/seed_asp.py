asp_features = """
### **Assembly Scheduling Problem (ASP) – Key Features**

- There is a set of **operations** that need to be scheduled for processing.  
- These operations are linked by **precedence constraints** — meaning that **some operations cannot start until all of their predecessor operations are completed**.  
- The dependencies define a **partial order**, typically modeled as a **directed acyclic graph (DAG)** representing the structure of the assembly process.  
"""

asp_data_seed = [
    {
        "title": "Aircraft Fuselage Assembly",
        "desc_split": "An aerospace manufacturer assembles <num_operations> fuselage components for a commercial jet. Each component corresponds to a specific operation such as frame welding, panel fitting, and internal wiring. These operations follow strict dependencies — e.g., wiring cannot begin before structural elements are in place — as defined in <precedence_relations>. The duration of each operation is fixed and recorded in <processing_times>.",
        "label": "asp",
    },
    {
        "title": "Medical Device Final Assembly",
        "desc_split": "In a biomedical plant, <num_operations> operations are required to assemble a portable diagnostic device. Tasks like sensor calibration, casing installation, and software flashing must follow a partial order where software cannot be flashed before internal components are fully installed. These precedence relationships are represented in <precedence_relations>, and execution times for each task are given in <processing_times>.",
        "label": "asp",
    },
    {
        "title": "Construction Equipment Subassembly",
        "desc_split": "A heavy machinery factory builds engine modules for excavators, involving <num_operations> operations such as block casting, piston fitting, lubrication setup, and final testing. The tasks follow a defined dependency structure — for instance, testing must wait until all internal components are installed. The assembly logic is modeled as a DAG in <precedence_relations>, and task durations are provided in <processing_times>.",
        "label": "asp",
    },
    {
        "title": "Smart Home Device Packaging Line",
        "desc_split": "In a smart electronics factory, <num_operations> packaging operations are carried out to assemble, test, and box smart home devices. Tasks like battery installation must be completed before final sealing, and device diagnostics must precede labeling. These relationships are encoded in <precedence_relations>, and each operation's processing time is listed in <processing_times>.",
        "label": "asp",
    },
    {
        "title": "Tailored Suit Manufacturing",
        "desc_split": "A luxury tailoring workshop manages <num_operations> dependent tasks such as cutting, assembling, buttoning, steaming, and final inspection. Each step relies on the completion of prior ones — e.g., sewing cannot start until all fabric pieces are cut. These dependencies form a DAG defined in <precedence_relations>, and each step has a specific duration defined in <processing_times>.",
        "label": "asp",
    }
]

asp_data_tag = """
### ✅ Data Tags Summary

| Tag | Description |
|------|-------------|
| `<num_operations>` | Total number of operations to schedule |
| `<precedence_relations>` | A list or graph of dependencies between operations (as a DAG) |
| `<processing_times>` | A dictionary or list of processing durations per operation |

---

Would you like to see one of these converted into a sample instance (e.g., using a DAG and task durations), or exported to JSON/CSV?
"""
