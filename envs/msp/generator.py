import random
from enum import Enum
from collections import defaultdict
from typing import Dict, List, Tuple, Union, OrderedDict

from benchmark_hard.msp.machine_scheduing_config import problem_type_param


class SchedulingProblemType(Enum):
    ASP = 'asp'  # Assembly Scheduling Problem
    FJSSP = 'fjssp'  # Flexible Job Shop Scheduling Problem
    FSSP = 'fssp'  # Flow Shop Scheduling Problem
    HFSSP = 'hfssp'  # Hybrid Flow Shop Scheduling Problem
    JSSP = 'jssp'  # Job Shop Scheduling Problem
    OSSP = 'ossp'  # Open Shop Scheduling Problem


class SchedulingProblemGenerator:
    def __init__(self,
                 problem_type: str,
                 min_jobs: int = 4,
                 max_jobs: int = 10,
                 min_machines: int = 2,
                 max_machines: int = 5,
                 min_operations: int = 2,
                 max_operations: int = 5,
                 min_processing_time: int = 1,
                 max_processing_time: int = 10):
        if problem_type == 'jssp':
            self.problem_type = SchedulingProblemType.JSSP
        elif problem_type == 'fjssp':
            self.problem_type = SchedulingProblemType.FJSSP
        elif problem_type == 'fssp':
            self.problem_type = SchedulingProblemType.FSSP
        elif problem_type == 'hfssp':
            self.problem_type = SchedulingProblemType.HFSSP
        elif problem_type == 'ossp':
            self.problem_type = SchedulingProblemType.OSSP
        elif problem_type == 'asp':
            self.problem_type = SchedulingProblemType.ASP
        else:
            raise NotImplementedError(f'{problem_type} is not implemented')

        self.min_jobs = min_jobs
        self.max_jobs = max_jobs
        self.min_machines = min_machines
        self.max_machines = max_machines
        self.min_operations = min_operations
        self.max_operations = max_operations
        self.min_processing_time = min_processing_time
        self.max_processing_time = max_processing_time

    def generate_jssp_processing_times(self, num_jobs: int, num_machines: int) -> dict:
        """
        Generate JSSP data to ensure that all machines are used as evenly as possible.

        Returns:
            Dict[int, Dict[Dict[int, int]]]
                {
                    job_id: {
                        op_id: {machine_id: processing_time},
                        op_id: {machine_id: processing_time},
                        ...
                    },
                    ...
                }
        """

        jobs: Dict[int, Dict[int, Dict[int, Tuple[int, int]]]] = {}
        op_id = 0
        for job_id in range(num_jobs):
            job_ops: Dict[int, Dict[int, Tuple[int, int]]] = {}
            machines = list(range(num_machines))
            random.shuffle(machines)
            for _ in range(num_machines):
                machine_id = machines[_]
                proc_time = random.randint(self.min_processing_time, self.max_processing_time)
                job_ops[op_id] = {machine_id: (proc_time, machine_id), }
                op_id += 1
            jobs[job_id] = job_ops
        return jobs

    def generate_fjssp_processing_times(self, num_jobs: int, num_machines: int, num_operations: int, ) -> dict:
        """
        Create random processing‑time data for a Flexible Job‑Shop Scheduling Problem.

        Parameters
        ----------
        num_jobs : int
            How many distinct jobs.
        num_machines : int
            How many parallel machines are in the shop.
        num_operations : int
            How many sequential operations each job contains.

        Returns
        -------
        Dict[int, Dict[Dict[int, int]]]
            {
                job_id: {
                    op_id: {machine_id: processing_time, …},
                    op_id: {machine_id: processing_time, …},
                    ...
                },
                ...
            }
        """

        jobs: Dict[int, Dict[int, Dict[int, Tuple[int, int]]]] = {}

        op_id = 0
        for job_id in range(num_jobs):
            job_ops: Dict[int, Dict[int, Tuple[int, int]]] = {}
            for _ in range(num_operations):
                op_times = {
                    machine_id: (random.randint(self.min_processing_time, self.max_processing_time), machine_id)
                    for machine_id in range(num_machines)
                }
                job_ops[op_id] = op_times
                op_id += 1
            jobs[job_id] = job_ops

        return jobs

    def generate_fssp_processing_times(self, num_jobs: int, num_machines: int, ) -> Dict:
        """
        Processing‑time data for an FSSP in a 3‑level dict:
            {
                job_id: {
                    op_id: { machine_id: processing_time },
                    ...
                },
                ...
            }

        Notes
        -----
        * In a pure flow shop, op_id == machine_id (the *k*‑th operation
          of every job is performed on machine *k*).  Keeping the extra
          dictionary layer preserves the same schema you used for FJSSP,
          which can make downstream code simpler when you support both
          problems with one loader.
        """
        jobs: Dict[int, Dict[int, Dict[int, Tuple[int, int]]]] = {}
        op_id = 0
        for job_id in range(num_jobs):
            job_ops: Dict[int, Dict[int, Tuple[int, int]]] = {}
            for machine_id in range(num_machines):
                proc_time = random.randint(self.min_processing_time, self.max_processing_time)
                job_ops[op_id] = {machine_id: (proc_time, machine_id)}  # only one machine per op in FSSP
                op_id += 1
            jobs[job_id] = job_ops

        return jobs

    def generate_hfssp_processing_times(self, num_jobs: int, machines_per_stage: Dict) -> Dict:
        """Generate processing times respecting the stage-machine configuration"""

        # stages = sorted(machines_per_stage.keys(), key=lambda x: int(x.split('_')[1]))
        stages = sorted(machines_per_stage.keys())
        jobs: Dict[int, Dict[int, Dict[int, Tuple[int, int]]]] = {}

        for job_id in range(num_jobs):
            job_stages: Dict[int, Dict[int, Tuple[int, int]]] = {}
            machine_id = 0
            for stage in stages:
                stage_times = {}
                num_machines = machines_per_stage[stage]
                for m_idx in range(num_machines):
                    stage_times[machine_id] = (
                        random.randint(self.min_processing_time, self.max_processing_time),
                        machine_id
                    )
                    machine_id += 1
                job_stages[stage] = stage_times
            jobs[job_id] = job_stages
        return jobs

    def generate_processing_times(self, num_jobs: int, num_machines: int, num_operations: int) -> dict:
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

    def generate_precedence_relations(
            self,
            num_jobs: int,
            jobs: Dict,

    ) -> List[List]:
        """
        把 data 按顺序划分成 n 份（每份≥2），
        然后从每份挑 2 个元素汇成列表并返回。

        返回值: (chunks, picked)
            chunks : List[List]   # n' 个区间 (n' ≤ n)
            picked : List         # len == 2 * n'
        """
        items = list(range(0, num_jobs))
        relation_num = max(int(0.3 * num_jobs), 1)
        # ---------- 1. 校正 n ----------
        base = num_jobs // relation_num  # 每段至少 base 个
        while relation_num > 0 and base < 2:
            relation_num = relation_num // 2
            base = num_jobs // relation_num

        # ---------- 2. 划分 ----------

        extra = num_jobs % relation_num  # 前 extra 段多 1 个
        chunks: List[List] = []
        idx = 0
        for i in range(relation_num):
            size = base + (1 if i < extra else 0)
            chunks.append(items[idx: idx + size])
            idx += size

        # ---------- 3. 挑选 ----------

        pairs = [random.sample(chunk, 2) for chunk in chunks]

        precedence = []
        for p in pairs:
            job1 = min(p)
            job2 = max(p)
            job1_opes = sorted(list(jobs[job1].keys()))
            job2_opes = sorted(list(jobs[job2].keys()))
            op1_idx = random.randint(0, len(job1_opes) - 1)
            op2_idx = random.randint(0, len(job2_opes) - 1)
            precedence.append([
                (job1, job1_opes[op1_idx]),
                (job2, job2_opes[op2_idx])
            ])
        return precedence

    # def generate_precedence_relations(self, num_jobs: int, num_operations: int, jobs: Dict) -> List:
    #     """Generate precedence relations based on problem type"""
    #     precedence = []
    #     relation_num = random.randint(1, max(2, num_jobs // 2))
    #     step = max(num_jobs // relation_num, 2)
    #     prev_job = -1
    #     try:
    #         for idx in range(relation_num):
    #             job1 = random.randint(prev_job + 1, min(prev_job + step - 1, num_jobs - 2))
    #             job2 = random.randint(job1 + 1, min(prev_job + step, num_jobs - 1))
    #             prev_job = job2
    #             job1_opes = sorted(list(jobs[job1].keys()))
    #             job2_opes = sorted(list(jobs[job2].keys()))
    #             op1_idx = random.randint(0, len(job1_opes) - 1)
    #             op2_idx = random.randint(0, len(job2_opes) - 1)
    #             precedence.append([
    #                 (job1, job1_opes[op1_idx]),
    #                 (job2, job2_opes[op2_idx])
    #             ])
    #     except Exception:
    #         print(precedence)
    #
    #     # prev_job1 = -1
    #     # prev_job2 = 0
    #     # check_jobs = np.zeros(num_jobs)
    #     # for idx in range(relation_num):  # Generate some random cross-job relations
    #     #     job1 = random.sample(range(prev_job1 + 1, num_jobs - relation_num + idx), 1)[0]
    #     #     job2 = random.sample(range(max(job1 + 1, prev_job2 + 1), num_jobs - relation_num + idx + 1), 1)[0]
    #     #     if check_jobs[job1] and check_jobs[job2]:
    #     #         continue
    #     #     job1_opes = sorted(list(jobs[job1].keys()))
    #     #     job2_opes = sorted(list(jobs[job2].keys()))
    #     #     op1_idx = random.randint(0, len(job1_opes) - 2)
    #     #     op2_idx = random.randint(op1_idx + 1, len(job2_opes) - 1)
    #     #     check_jobs[job1] = True
    #     #     check_jobs[job2] = True
    #     #     precedence.append([
    #     #         (job1, job1_opes[op1_idx]),
    #     #         (job2, job2_opes[op2_idx])
    #     #     ])
    #     return precedence

    def generate_machines_per_stage(self, num_machines: int) -> Dict:
        """Generate machine distribution for HFSSP with guaranteed:
        - At least 2 stages (stage_num >= 2)
        - At least one stage with ≥ 2 machines
        - All machines allocated"""

        if self.problem_type != SchedulingProblemType.HFSSP:
            return {}

        # Enforce minimum requirements
        if num_machines < 3:
            raise RuntimeError("Num machines must larger than 3.")  # Minimum machines needed to satisfy both conditions

        # Calculate number of stages (at least 2)
        min_stages = 2
        max_stages = min(num_machines - 1, 5)  # Reasonable upper limit
        num_stages = random.randint(min_stages, max_stages)

        # Initialize with at least 1 machine per stage
        machines_per_stage = {i: 1 for i in range(num_stages)}
        remaining_machines = num_machines - num_stages  # After assigning 1 to each stage

        # Ensure at least one stage gets an extra machine (making it ≥ 2)
        # Distribute remaining machines randomly
        while remaining_machines > 0:
            stage = random.choice(list(machines_per_stage.keys()))
            machines_per_stage[stage] += 1
            remaining_machines -= 1

        # Shuffle stage order for randomness
        stages = list(machines_per_stage.items())
        random.shuffle(stages)

        return OrderedDict(stages)

    def generate_data_tag(self, data):
        format_data = data.copy()
        format_jobs = {}
        for job_id in data['processing_times']:
            format_ops = {}
            for op_id in data['processing_times'][job_id]:
                op_times = {}
                for machine_id in data['processing_times'][job_id][op_id]:
                    process_time = data['processing_times'][job_id][op_id][machine_id][0]
                    op_times[f'machine_{machine_id}'] = (process_time, f'machine_{machine_id}')
                if self.problem_type == SchedulingProblemType.HFSSP:
                    format_ops[f'stage_{op_id}'] = op_times
                else:
                    format_ops[f'op_{op_id}'] = op_times
            format_jobs[f'job_{job_id}'] = format_ops
        format_data['processing_times'] = format_jobs

        if data.get('machines_per_stage', None):
            format_machines_per_stage = {}
            for stage_id, num_stage_machine in data['machines_per_stage'].items():
                format_machines_per_stage[f'stage_{stage_id}'] = num_stage_machine
            format_data['machines_per_stage'] = format_machines_per_stage

        if data.get('precedence_relations', None):
            format_precedence_relations = []
            for relation in data['precedence_relations']:
                format_precedence_relations.append([
                    (f'job_{relation[0][0]}', f'op_{relation[0][1]}'),
                    (f'job_{relation[1][0]}', f'op_{relation[1][1]}')
                ])
            format_data['precedence_relations'] = format_precedence_relations

        return format_data

    def generate_problem_instance(self, **params) -> Dict:
        """Generate a complete problem instance"""
        num_jobs = params.get('num_jobs', random.randint(self.min_jobs, self.max_jobs))
        num_machines = params.get('num_machines', random.randint(self.min_machines, self.max_machines))
        num_operations = params.get('num_operations', random.randint(self.min_operations, self.max_operations))

        data = {
            'num_jobs': num_jobs,
            'num_machines': num_machines,
            'machines_per_stage': {},
            'precedence_relations': [],
            'is_open': False,
            'is_flow': False,
        }

        if self.problem_type == SchedulingProblemType.JSSP:
            data['processing_times'] = self.generate_jssp_processing_times(num_jobs, num_machines)

        elif self.problem_type == SchedulingProblemType.FJSSP:
            data['processing_times'] = self.generate_fjssp_processing_times(num_jobs, num_machines, num_operations)

        elif self.problem_type == SchedulingProblemType.FSSP:
            data['processing_times'] = self.generate_fssp_processing_times(num_jobs, num_machines)
            data['is_flow'] = True
        elif self.problem_type == SchedulingProblemType.HFSSP:
            machines_per_stage = params.get('machines_per_stage', None)
            if not machines_per_stage:
                while num_machines < 3:
                    num_machines = random.randint(self.min_machines, self.max_machines)
                machines_per_stage = self.generate_machines_per_stage(num_machines)
            num_machines = sum(machines_per_stage.values())
            data['num_machines'] = num_machines
            data['machines_per_stage'] = machines_per_stage
            data['processing_times'] = self.generate_hfssp_processing_times(num_jobs, machines_per_stage)
            data['is_flow'] = True
        elif self.problem_type == SchedulingProblemType.OSSP:
            data['processing_times'] = self.generate_jssp_processing_times(num_jobs, num_machines)
            data['is_open'] = True
        elif self.problem_type == SchedulingProblemType.ASP:
            jobs = self.generate_fjssp_processing_times(num_jobs, num_machines, num_operations)
            data['processing_times'] = jobs
            data['precedence_relations'] = self.generate_precedence_relations(num_jobs, jobs)
        else:
            raise RuntimeError(f"Unknown problem type {self.problem_type}.")
        data = self.generate_data_tag(data)
        return data

    def generate(self, batch_size, **params) -> List[Dict]:
        """Generate and optionally save problem instance as JSON"""
        if not params:
            params = {}
            if self.problem_type.value in problem_type_param:
                params = problem_type_param[self.problem_type.value]
        instances = []
        for i in range(batch_size):
            instance = self.generate_problem_instance(**params)
            instances.append(instance)
        return instances


# Example usage
if __name__ == "__main__":
    # Generate one of each problem type
    # for problem in ProblemType:
    #     generator = SchedulingProblemGenerator(problem_type=problem)
    #     print(f"\n{problem.name} Example:")
    #     print(generator.generate_json())
    problem = SchedulingProblemType.HFSSP
    generator = SchedulingProblemGenerator(problem_type=problem)
    print(f"\n{problem.name} Example:")
    print(generator.generate(1))
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
