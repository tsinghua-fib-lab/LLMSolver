fssp_features = """
### **Flow-Shop Scheduling Problem (FSSP) – Key Features**

- Each job consists of **exactly *m* operations**, corresponding to *m* machines.  
- The **i-th operation of a job must be processed on the i-th machine** (i.e., all jobs follow the same machine sequence).  
- At any time, **each machine can process at most one operation** — no overlaps.  
- The **processing time for each operation of each job is known and fixed**.
- Objective: Minimize the makespan.
"""

fssp_data_seed = [
    {
        "title": "Printed Circuit Board (PCB) Assembly Line",
        "desc_split": "A PCB production facility processes <num_jobs> circuit boards through <num_machines> fixed stations including solder paste printing, component mounting, reflow soldering, and inspection. Each board follows the exact same sequence, where the i-th operation is processed on the i-th machine. Only one board can be handled on each machine at a time. All processing durations are predetermined and specified in <processing_times>. The scheduling objective is to minimize the makespan—the total time required to complete all jobs.",
        "label": "fssp",
    },
    {
        "title": "Aircraft Wing Surface Finishing",
        "desc_split": "In an aerospace facility, <num_jobs> wing panels undergo <num_machines> surface treatment steps in the same order — cleaning, sanding, coating, and drying. Every job must pass through all stages in the same fixed sequence, with each i-th operation on the i-th machine. Machine usage is exclusive at any time, and all operation durations are detailed in <processing_times>. The scheduling goal is to minimize the makespan (total completion time) of all wing panel treatments.",
        "label": "fssp",
    },
    {
        "title": "Household Cleaner Bottling Line",
        "desc_split": "A bottling plant fills <num_jobs> bottles through <num_machines> consecutive machines: liquid filling, cap sealing, labeling, and boxing. The operation sequence is identical for all jobs, and each machine handles one bottle at a time. The processing time for each job-machine pair is defined in <processing_times>. The production schedule aims to minimize the makespan - the total time needed to complete all bottling operations.",
        "label": "fssp",
    },
    {
        "title": "DNA Test Kit Manufacturing",
        "desc_split": "A biomedical factory manufactures <num_jobs> DNA test kits through <num_machines> tightly controlled stages like reagent mixing, packaging, sealing, and labeling. Each test kit must go through the same machine sequence, and processing times for each step are listed in <processing_times>. Only one job can be processed per machine at any moment. The production scheduling objective is to minimize the makespan (total production time) for completing all DNA test kits.",
        "label": "fssp",
    },
    {
        "title": "Automotive Brake Pad Production",
        "desc_split": "In an automotive supplier's production line, <num_jobs> brake pad units move through <num_machines> machines performing pressing, heating, grinding, and inspection. All jobs follow this exact order. Each machine operates on one job at a time, and the operation durations are pre-specified in <processing_times>. The production line scheduling aims to minimize the makespan (total completion time) for processing all brake pad units through the manufacturing sequence.",
        "label": "fssp",
    },
]

fssp_data_tag_list = ['num_jobs', 'num_machines', 'processing_times']

data = {
    'num_jobs': 4,
    'num_machines': 2,
    'processing_times': {
        'job_id': {'operation_id': ['machine_id']}
    },
}

"""
### ✅ Data Tags Summary:

| Tag | Description |
|------|-------------|
| `<num_jobs>` | Total number of jobs (or products) |
| `<num_machines>` | Total number of machines/stages (equal to number of operations per job) |
| `<processing_times>` | A matrix or dict defining processing time for each `(job, machine)` pair |

---

Would you like me to generate a sample `<processing_times>` matrix or convert one of these into a JSON/Python object format?

"""
