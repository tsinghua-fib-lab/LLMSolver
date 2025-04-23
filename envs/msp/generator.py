import random
from typing import Dict, List, Tuple, Union, OrderedDict
import json
from enum import Enum


class SchedulingProblemType(Enum):
    ASP = 'asp'  # Assembly Scheduling Problem
    FJSSP = 'fjssp'  # Flexible Job Shop Scheduling Problem
    FSSP = 'fssp'  # Flow Shop Scheduling Problem
    HFSSP = 'hfssp'  # Hybrid Flow Shop Scheduling Problem
    JSSP = 'jssp'  # Job Shop Scheduling Problem
    OSSP = 'ossp'  # Open Shop Scheduling Problem


class SchedulingProblemGenerator:
    def __init__(self,
                 problem_type: SchedulingProblemType,
                 min_jobs: int = 4,
                 max_jobs: int = 10,
                 min_machines: int = 2,
                 max_machines: int = 5,
                 min_operations: int = 2,
                 max_operations: int = 5,
                 min_processing_time: int = 1,
                 max_processing_time: int = 10):

        self.problem_type = problem_type
        self.min_jobs = min_jobs
        self.max_jobs = max_jobs
        self.min_machines = min_machines
        self.max_machines = max_machines
        self.min_operations = min_operations
        self.max_operations = max_operations
        self.min_processing_time = min_processing_time
        self.max_processing_time = max_processing_time

    def generate_processing_times(self, num_jobs: int, num_machines: int, num_operations: int) -> Dict:
        """Generate processing times based on problem type"""
        processing_times = {}

        for job_id in range(0, num_jobs):
            job_key = f'job_{job_id}'
            processing_times[job_key] = {}

            for op_id in range(0, num_operations):
                op_key = f'op_{op_id}'

                if self.problem_type in [SchedulingProblemType.FJSSP, SchedulingProblemType.HFSSP]:
                    # Flexible problems: multiple machine options per operation
                    processing_times[job_key][op_key] = {}
                    num_eligible = random.randint(0, num_machines - 1)
                    eligible_machines = random.sample(range(0, num_machines), num_eligible)

                    for machine_id in eligible_machines:
                        processing_times[job_key][op_key][f'machine_{machine_id}'] = (
                            random.randint(self.min_processing_time, self.max_processing_time),
                            machine_id
                        )
                else:
                    # Non-flexible problems: one machine per operation
                    if self.problem_type == SchedulingProblemType.JSSP:
                        # Job Shop: random machine assignment
                        machine_id = random.randint(0, num_machines - 1)
                    elif self.problem_type == SchedulingProblemType.OSSP:
                        # Open Shop: any order, but need all machines
                        machine_id = ((op_id - 1) % num_machines)
                    else:
                        # Flow Shop variants: same machine sequence
                        machine_id = min(op_id, num_machines)

                    processing_times[job_key][op_key] = {
                        f'machine_{machine_id}': (
                            random.randint(self.min_processing_time, self.max_processing_time),
                            machine_id
                        )
                    }

        return processing_times

    def generate_precedence_relations(self, num_jobs: int, num_operations: int) -> List:
        """Generate precedence relations based on problem type"""
        precedence = []
        relation_num = random.randint(0, num_jobs - 1)
        for _ in range(relation_num):  # Generate some random cross-job relations
            job2 = random.sample(range(1, num_jobs), 1)[0]
            job1 = random.sample(range(0, job2), 1)[0]

            op1 = random.randint(0, num_operations - 1)
            op2 = random.randint(0, num_operations - 1)
            precedence.append([
                (f'job_{job1}', f'op_{op1}'),
                (f'job_{job2}', f'op_{op2}')
            ])
        return precedence

    def generate_machines_per_stage(self, num_machines: int) -> Dict:
        """Generate machine distribution for HFSSP with guaranteed:
        - At least 2 stages (stage_num >= 2)
        - At least one stage with ≥ 2 machines
        - All machines allocated"""

        if self.problem_type != SchedulingProblemType.HFSSP:
            return {}

        # Enforce minimum requirements
        if num_machines < 3:
            num_machines = 3  # Minimum machines needed to satisfy both conditions

        # Calculate number of stages (at least 2)
        min_stages = 2
        max_stages = min(num_machines - 1, 5)  # Reasonable upper limit
        num_stages = random.randint(min_stages, max_stages)

        # Initialize with at least 1 machine per stage
        machines_per_stage = {f'stage_{i}': 1 for i in range(num_stages)}
        remaining_machines = num_machines - num_stages  # After assigning 1 to each stage

        # Ensure at least one stage gets an extra machine (making it ≥ 2)
        if remaining_machines > 0:
            lucky_stage = random.choice(list(machines_per_stage.keys()))
            machines_per_stage[lucky_stage] += 1
            remaining_machines -= 1

        # Distribute remaining machines randomly
        while remaining_machines > 0:
            stage = random.choice(list(machines_per_stage.keys()))
            machines_per_stage[stage] += 1
            remaining_machines -= 1

        # Shuffle stage order for randomness
        stages = list(machines_per_stage.items())
        random.shuffle(stages)

        return OrderedDict(stages)

    def generate_HFSSP_processing_times(self, num_jobs: int, machines_per_stage: Dict):
        """Generate processing times respecting the stage-machine configuration"""
        processing_times = {}
        stages = sorted(machines_per_stage.keys(), key=lambda x: int(x.split('_')[1]))

        for job_id in range(0, num_jobs):
            job_key = f'job_{job_id}'
            processing_times[job_key] = {}

            for stage_num, stage in enumerate(stages):
                op_key = f'stage_{stage_num}'  # Using stage as operation
                processing_times[job_key][op_key] = {}

                num_machines = machines_per_stage[stage]
                for machine_id in range(0, num_machines):
                    processing_times[job_key][op_key][f'machine_{machine_id}'] = (
                        random.randint(self.min_processing_time, self.max_processing_time),
                        machine_id
                    )

        return processing_times

    def generate_problem_instance(self) -> Dict:
        """Generate a complete problem instance"""
        num_jobs = random.randint(self.min_jobs, self.max_jobs)
        num_machines = random.randint(self.min_machines, self.max_machines)
        num_operations = random.randint(self.min_operations, self.max_operations)

        # Adjust for problem type constraints
        if self.problem_type in [SchedulingProblemType.FSSP, SchedulingProblemType.HFSSP]:
            num_operations = min(num_operations, num_machines)

        data = {
            'problem_type': self.problem_type.value,
            'num_jobs': num_jobs,
            'num_machines': num_machines,
            'machines_per_stage': {},
            'precedence_relations': [],
            'is_open': False,
        }

        if self.problem_type == SchedulingProblemType.HFSSP:
            num_operations = num_machines
            data['machines_per_stage'] = self.generate_machines_per_stage(num_machines)
            data['processing_times'] = self.generate_HFSSP_processing_times(num_jobs, data['machines_per_stage'])
        else:
            data['processing_times'] = self.generate_processing_times(num_jobs, num_machines, num_operations)

        if self.problem_type == SchedulingProblemType.ASP:
            data['precedence_relations'] = self.generate_precedence_relations(num_jobs, num_operations)

        if self.problem_type == SchedulingProblemType.OSSP:
            data['is_open'] = True

        if self.problem_type in [SchedulingProblemType.FSSP, SchedulingProblemType.HFSSP]:
            data['is_flow'] = True
        else:
            data['is_flow'] = False
        return data

    def generate_json(self, file_path: str = None) -> str:
        """Generate and optionally save problem instance as JSON"""
        instance = self.generate_problem_instance()
        json_data = json.dumps(instance, indent=2)

        if file_path:
            with open(file_path, 'w') as f:
                f.write(json_data)

        return json_data


# Example usage
if __name__ == "__main__":
    # Generate one of each problem type
    # for problem in ProblemType:
    #     generator = SchedulingProblemGenerator(problem_type=problem)
    #     print(f"\n{problem.name} Example:")
    #     print(generator.generate_json())
    problem = SchedulingProblemType.FSSP
    generator = SchedulingProblemGenerator(problem_type=problem)
    print(f"\n{problem.name} Example:")
    print(generator.generate_json())
# if __name__ == '__main__':
#     n_j = 5
#     n_m = 4
#     op_per_job = 10
#     op_per_mch_min = 1
#     op_per_mch_max = n_m
#     n_data = 1
#     instances = generate_instances(n_j=n_j, n_m=n_m, op_per_job=op_per_job, op_per_mch_min=op_per_mch_min,
#                                    op_per_mch_max=op_per_mch_max,
#                                    n_data=n_data)
#     print(instances)
