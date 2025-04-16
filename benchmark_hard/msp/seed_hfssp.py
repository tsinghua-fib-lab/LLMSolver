hfssp_features = """
### **Hybrid Flow-Shop Scheduling Problem (HFSSP) – Key Features**

- There is a set of **n jobs** that must be processed across a series of **processing stages**.  
- The number of stages is **at least two**.  
- **Each stage contains at least one machine**, and **at least one stage has multiple machines operating in parallel**.  
- All jobs follow the **same stage sequence**: processing begins at stage 1 and continues sequentially through to the final stage.  
- A job can only begin processing in the **next stage after it finishes the previous one**.  
"""
hfssp_data_seed = [
    {
        "title": "Electronic Component Assembly Line",
        "desc_split": "A factory produces <num_jobs> batches of electronic components through <num_stages> stages, including surface mounting, soldering, quality control, and final packaging. Each stage contains a different number of parallel machines, represented by <machines_per_stage>. All jobs follow the same stage sequence, and can only move to the next stage once the previous one is complete. The processing times per job and stage are defined in <processing_times>.",
        "label": "hfssp",
    },
    {
        "title": "Cosmetic Production and Packaging",
        "desc_split": "In a cosmetics facility, <num_jobs> product batches undergo <num_stages> stages: formulation, filling, labeling, and boxing. Each stage has at least one machine, and some stages like labeling include multiple parallel machines for high throughput. All jobs proceed in the same order, and no job can advance until its current stage is finished. Machine-specific task durations are specified in <processing_times>, and machine availability per stage is listed in <machines_per_stage>.",
        "label": "hfssp",
    },
    {
        "title": "Aircraft Component Manufacturing",
        "desc_split": "An aerospace plant processes <num_jobs> aircraft components across <num_stages> stages such as forging, heat treatment, grinding, and inspection. At least one of these stages (e.g., grinding) has multiple machines operating in parallel, as defined in <machines_per_stage>. Jobs must complete all stages in a fixed sequence, and cannot skip or overlap stages. The operation durations are machine-dependent and recorded in <processing_times>.",
        "label": "hfssp",
    },
    {
        "title": "E-commerce Order Fulfillment",
        "desc_split": "A logistics center handles <num_jobs> online orders, each passing through <num_stages> stages such as picking, sorting, packing, and dispatch. While each stage has at least one workstation, stages like sorting and packing include several machines working in parallel, shown in <machines_per_stage>. The workflow is linear, and each job follows the exact same sequence. Time required for each job at each stage is detailed in <processing_times>.",
        "label": "hfssp",
    },
    {
        "title": "Chemical Batch Processing",
        "desc_split": "In a chemical plant, <num_jobs> batches undergo <num_stages> operations including mixing, heating, cooling, and bottling. Machines in each stage may differ in type and capacity, and some stages (e.g., bottling) feature parallel equipment to increase efficiency. All jobs are processed in the same order through each stage, with no overlap allowed. Parallel capacity per stage is captured in <machines_per_stage>, and processing durations are listed in <processing_times>.",
        "label": "hfssp",
    },
]

hfssp_data_tag = """
### ✅ Data Tags Summary:

| Tag | Description |
|------|-------------|
| `<num_jobs>` | Number of jobs to schedule |
| `<num_stages>` | Number of processing stages |
| `<machines_per_stage>` | A list (or dict) indicating how many machines are available per stage |
| `<processing_times>` | A dict: `{(job, stage, machine): time}` for machine-specific durations |

---

Let me know if you’d like help generating actual sample instances for one of these scenarios, or want this structure exported as JSON/CSV!
"""
