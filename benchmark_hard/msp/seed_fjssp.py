fjssp_features = """
### **Flexible Job Shop Scheduling Problem (FJSSP) – Key Features**

- Each job consists of a sequence of **ordered operations**, which must be performed in a **predefined order**.  
- Each operation can be processed by **one of several compatible machines** (i.e., it has a machine choice set).  
- Objective: Minimize the makespan.
"""

fjssp_data_seed = [
    {
        "title": "High-Precision Medical Device Manufacturing",
        "desc_split": "A facility produces <num_jobs> batches of custom medical devices, each requiring a specific sequence of operations such as micromachining, assembly, testing, and packaging. Each operation has a predefined position in the job and can be executed by one of several compatible machines from a pool of <num_machines>. The processing times vary depending on the selected machine, as defined in <processing_times>. No two operations overlap on the same machine, and tasks are non-preemptive. The goal is to schedule all jobs to minimize the makespan—the total time to complete all batches.",
        "label": "fjssp",
    },
    {
        "title": "Aerospace Structural Parts Fabrication",
        "desc_split": "An aerospace manufacturer handles <num_jobs> structural parts, each passing through operations like CNC milling, ultrasonic inspection, and surface finishing. The operations must follow a strict order, but flexibility exists in machine selection — each operation can be processed by any compatible machine among <num_machines>. Operation durations depend on the chosen machine, as recorded in <processing_times>, and no operation may be interrupted once started. The production scheduling aims to minimize the overall makespan while meeting aerospace quality standards.",
        "label": "fjssp",
    },
    {
        "title": "Smart Logistics Robotics Assembly",
        "desc_split": "A robotics assembly center manages <num_jobs> robotic units, each assembled through an ordered set of operations such as circuit soldering, sensor installation, casing, and calibration. Each operation is eligible for execution on multiple stations selected from <num_machines>, with varying speeds and capabilities. Processing times are mapped in <processing_times>, and all operations are exclusive and sequential within a job.  The production scheduling aims to minimize the total makespan while ensuring all robotic units meet functional specifications.",
        "label": "fjssp",
    },
    {
        "title": "Flexible Cosmetic Production Line",
        "desc_split": "A cosmetics company produces <num_jobs> product lines, each with a fixed sequence of operations: mixing, pouring, capping, and labeling. Each task may be handled by a subset of <num_machines> available machines, depending on product type or capacity. The processing times are machine-dependent and defined in <processing_times>, and all operations must run to completion once started. The production scheduling aims to minimize the makespan - the total time required to complete all product lines while maintaining manufacturing efficiency.",
        "label": "fjssp",
    },
    {
        "title": "Luxury Garment Tailoring Studio",
        "desc_split": "A luxury fashion studio manages <num_jobs> high-end clothing items that go through cutting, stitching, embroidery, and pressing in that order. Each operation is handled by a group of skilled workers or machines selected from <num_machines>. While each job has a fixed operation order, the flexibility comes from choosing the best resource for each step. The machine-specific processing durations are recorded in <processing_times>. The production scheduling aims to minimize the makespan - the total time required to complete all garments while maintaining the studio's exacting quality standards.",
        "label": "fjssp",
    },
]

fjssp_data_tag_list = ['num_jobs', 'num_machines', 'processing_times']

data = {
    'num_jobs': 4,
    'num_machines': 2,
    'processing_times': {
        'job_id': {'operation_id': ['machine_id']}
    },
}

# fjssp_data_tag = """
# ### ✅ Data Tags Summary:
#
# | Tag | Description |
# |------|-------------|
# | `<num_jobs>` | Total number of jobs |
# | `<num_machines>` | Total number of available machines |
# | `<processing_times>` | A dictionary: `(job, operation, machine) → processing time` |
#
# execution_times operations_on_machines
# ---
#
# Would you like a function to generate randomized `<processing_times>` or `<operations>` structures for one of these scenarios? Or export them into JSON, CSV, or Python dict format?
# """
