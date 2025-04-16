ossp_features = """
### **Open Shop Scheduling Problem (OSSP) – Key Features**

- Each job must be processed on a set of machines, but **the order of operations is not predefined** — the operation sequence is **flexible**.  
- At any given time, **a job can be processed on only one machine**, and **each machine can process only one job at a time**.  
- Unlike the Job Shop model, **there are no fixed operation sequences for jobs** — the scheduler can freely decide the order in which each job visits the machines.

"""

ossp_data_seed = [
    {
        "title": "Medical Imaging Center Scheduling",
        "desc_split": "A hospital must schedule <num_jobs> patients to undergo several diagnostic procedures, including X-rays, CT scans, and MRIs, across <num_machines> available machines. While each patient must complete all imaging procedures, the order is flexible and can be chosen to reduce overall wait times. Each scan has a fixed duration, defined in <processing_times>, and no machine or patient can be used for more than one scan at a time.",
        "label": "ossp",
    },
    {
        "title": "University Laboratory Equipment Usage",
        "desc_split": "In a research lab, <num_jobs> student projects require access to <num_machines> specialized instruments such as spectrometers, centrifuges, and analyzers. Each project must use all machines for testing, but the order is not constrained. The goal is to minimize total lab time while respecting that each project and each machine can only handle one task at a time. Processing durations are defined in <processing_times>.",
        "label": "ossp",
    },
    {
        "title": "Aircraft Maintenance Hangar",
        "desc_split": "An aircraft maintenance team handles <num_jobs> airplanes, each requiring inspections across <num_machines> service zones like hydraulics, avionics, and airframe. The order in which inspections are performed is flexible, depending on availability and technician schedules. Each service task has a fixed duration from <processing_times>, and only one aircraft can be serviced per zone at any time.",
        "label": "ossp",
    },
    {
        "title": "Laundry and Dry-Cleaning Service",
        "desc_split": "A dry-cleaning business processes <num_jobs> customer orders that must be cleaned, pressed, and packaged using <num_machines> stations. While each order must go through all stations, the sequence can be freely chosen to optimize machine usage and turnaround time. The time each order spends at each station is recorded in <processing_times>.",
        "label": "ossp",
    },
    {
        "title": "Custom Jewelry Polishing Workshop",
        "desc_split": "A jewelry shop processes <num_jobs> luxury items through <num_machines> stations like engraving, polishing, cleaning, and inspection. The workshop does not require a strict order of operations, allowing artisans to choose the sequence based on resource availability. Each task takes a defined time (see <processing_times>), and only one item can be processed at a time on any machine.",
        "label": "ossp",
    },
]

ossp_data_tag = """

### ✅ Data Tags Summary:

| Tag | Description |
|------|-------------|
| `<num_jobs>` | Total number of jobs (e.g., patients, items, aircraft) |
| `<num_machines>` | Number of machines or stations |
| `<processing_times>` | A matrix or dictionary defining time required for each `(job, machine)` pair |

---

Would you like me to help generate a random `<processing_times>` table or structure this for a specific optimization framework (e.g., OR-Tools, Gurobi)?
"""
