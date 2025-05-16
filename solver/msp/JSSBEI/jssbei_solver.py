import os
from copy import deepcopy
from typing import List, Dict

import numpy as np

from solver.msp.JSSBEI.data.data_parsers.custom_instance_parser import parse
from solver.msp.JSSBEI.solution_methods import load_parameters
from solver.msp.JSSBEI.solution_methods.GA.src.initialization import initialize_run
from solver.msp.JSSBEI.visualization import gantt_chart, precedence_chart

from envs.msp.generator import SchedulingProblemType, SchedulingProblemGenerator

##########################################################################################
# parameters

param_path_dict = {
    'milp': 'milp.toml',
    'cp_sat': 'cp_sat.toml',
    'dispatching_rules': 'dispatching_rules.toml',
    'GA': 'GA.toml',
    'l2d': 'L2D.toml',
    'fjsp_drl': 'FJSP_DRL.toml',
    'dannel': 'DANIEL.toml',
}

problem_type_dict = {
    'jssp': 'jsp',
    'fjssp': 'fjsp',
    'fssp': 'fsp',
    'asp': 'fajsp',
}


##########################################################################################
# format

def format_instance(data):
    """
    Convert the scheduling problem data to the specified format

    Args:
        data (dict): The original problem data with processing_times and precedence_relations

    Returns:
        dict: Reformatted data in the target structure
    """
    # Initialize the result structure

    result = {
        "nr_machines": data["num_machines"],
        "jobs": [],
        "sequence_dependent_setup_times": {},
    }

    sequence_dependent_setup_times = {}
    ope_num = sum(len(job_data) for job_data in data["processing_times"].values())
    for machine_id in range(data["num_machines"]):
        sequence_dependent_setup_times[f'machine_{machine_id}'] = np.zeros((ope_num, ope_num)).tolist()
    result["sequence_dependent_setup_times"] = sequence_dependent_setup_times
    # Create a mapping from job/operation to operation_id

    # First pass: assign operation IDs and build job structure
    for job_key in sorted(data["processing_times"].keys(), key=lambda x: int(x.split('_')[1])):
        job_id = int(job_key.split('_')[1])  # Extract job number from "job_X"
        operations = []

        for op_key in sorted(data["processing_times"][job_key].keys(), key=lambda x: int(x.split('_')[1])):
            # Create operation entry
            op_id = int(op_key.split('_')[1])
            operation = {
                "operation_id": op_id,
                "processing_times": {
                    f"machine_{int(machine.split('_')[1])}": time[0]
                    # Extract just the processing time (ignore machine_id)
                    for machine, time in data["processing_times"][job_key][op_key].items()
                },
                "predecessors": None  # Initialize as None, will update in next pass
            }
            operations.append(operation)

        # Add job to result
        result["jobs"].append({
            "job_id": job_id,
            "operations": operations
        })

    for job in result["jobs"]:
        sorted_operations = sorted(job["operations"], key=lambda x: x["operation_id"])
        prev_operation_id = -1
        for operation in sorted_operations:
            if prev_operation_id >= 0:
                if 'predecessors' not in operation or not operation['predecessors']:
                    operation['predecessors'] = []
                operation["predecessors"].append(prev_operation_id)
            prev_operation_id = operation["operation_id"]

    # Second pass: process precedence relations
    for relation in data["precedence_relations"]:
        from_job, from_op = relation[0]  # Predecessor
        to_job, to_op = relation[1]  # Successor

        # Find the operation IDs
        from_op_id = int(from_op.split('_')[1])
        to_op_id = int(to_op.split('_')[1])
        to_job_id = int(to_job.split('_')[1])

        # Find the target operation in the result structure
        for operation in result["jobs"][to_job_id]["operations"]:
            if operation["operation_id"] == to_op_id:
                # Initialize predecessors list if needed
                if operation["predecessors"] is None:
                    operation["predecessors"] = []
                # Add the predecessor
                operation["predecessors"].append(from_op_id)
                operation["predecessors"] = sorted(list(set(operation["predecessors"])))
                break

    return result


def format_parameters(problem_type: str, solver_name: str, parameters: Dict, instance: Dict):
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    num_operations = len(list(instance['processing_times'].values())[0])

    if solver_name == 'milp':
        parameters['instance']['instance_type'] = problem_type_dict[problem_type]
    elif solver_name == 'cp_sat':
        parameters['solver']['model'] = problem_type_dict[problem_type]
    elif solver_name == 'dispatching_rules':
        pass
    elif solver_name == 'GA':
        parameters['instance']['instance'] = problem_type_dict[problem_type]
    elif solver_name == 'l2d':
        pass
    elif solver_name == 'fjsp_drl':
        parameters['env_parameters']['num_jobs'] = num_jobs
        parameters['env_parameters']['num_mas'] = num_machines
    elif solver_name == 'dannel':
        parameters['env']['n_j'] = num_jobs
        parameters['env']['n_m'] = num_machines
        parameters['env']['n_op'] = num_operations
    else:
        raise NotImplementedError(f'No such solver: {solver_name}')

    return parameters


def format_result(schedule, instance):
    processing_times = instance['processing_times']
    # Get dimensions
    num_jobs = instance['num_jobs']
    machines_per_stage = instance['machines_per_stage']
    num_machines = sum(machines_per_stage.values())

    output_schedule = [{'job': job, 'tasks': []} for job in range(num_jobs)]

    for machine in schedule.machines:
        machine_id = machine.machine_id
        machine_operations = sorted(machine._processed_operations,
                                    key=lambda op: op.scheduling_information['start_time'])
        for task_id, operation in enumerate(machine_operations):
            job_id = operation.job_id
            op_id = operation.operation_id
            operation_start = round(operation.scheduling_information['start_time'])
            operation_end = round(operation.scheduling_information['end_time'])
            operation_duration = round(operation_end - operation_start)
            # operation_label = f"{operation.operation_id}"
            assert operation_duration == processing_times[f'job_{job_id}'][f'op_{op_id}'][f'machine_{machine_id}'][0]
            output_schedule[job_id]['tasks'].append({
                'duration':
                    processing_times[f'job_{job_id}'][f'op_{op_id}'][f'machine_{machine_id}'][0],
                'machine': machine_id,
                'start': operation_start,
                'task': op_id
            })

    result = {'schedule': output_schedule}
    return result


##########################################################################################
# main

class JSSBEISolver:
    def __init__(self, problem_type: str, solver_name: str):
        self.problem_type = problem_type
        assert problem_type in problem_type_dict
        param_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", param_path_dict[solver_name])
        self.parameters = load_parameters(param_path)
        self.solver_name = solver_name
        if solver_name == 'milp':
            from solver.msp.JSSBEI.solution_methods.MILP.run_MILP import run_MILP
            self.solver_func = run_MILP
        elif solver_name == 'cp_sat':
            from solver.msp.JSSBEI.solution_methods.CP_SAT.run_cp_sat import run_CP_SAT
            self.solver_func = run_CP_SAT
        elif solver_name == 'dispatching_rules':
            from solver.msp.JSSBEI.solution_methods.dispatching_rules.run_dispatching_rules import run_dispatching_rules
            self.solver_func = run_dispatching_rules
        elif solver_name == 'GA':
            from solver.msp.JSSBEI.solution_methods.GA.run_GA import run_GA
            self.solver_func = run_GA
        elif solver_name == 'l2d':
            from solver.msp.JSSBEI.solution_methods.L2D.run_L2D import run_L2D
            self.solver_func = run_L2D
        elif solver_name == 'fjsp_drl':
            from solver.msp.JSSBEI.solution_methods.FJSP_DRL.run_FJSP_DRL import run_FJSP_DRL
            self.solver_func = run_FJSP_DRL
        elif solver_name == 'dannel':
            from solver.msp.JSSBEI.solution_methods.DANIEL.run_DANIEL import run_DANIEL_FJSP
            self.solver_func = run_DANIEL_FJSP
        else:
            raise NotImplementedError(f'No such solver: {solver_name}')

    def solve(self, instances: List[Dict], **params) -> List[Dict]:
        result_list = []
        for instance in instances:
            parameters = format_parameters(self.problem_type, self.solver_name, deepcopy(self.parameters), instance)
            processing_info = format_instance(instance)
            # print(processing_info)
            jobShopEnv = parse(processing_info)
            if self.solver_name == 'GA':
                population, toolbox, stats, hof = initialize_run(jobShopEnv, **parameters)
                _, jobShopEnv = self.solver_func(jobShopEnv, population, toolbox, stats, hof, **parameters)
            else:
                _, jobShopEnv = self.solver_func(jobShopEnv, **parameters)
            # plt = gantt_chart.plot(jobShopEnv)
            # plt.show()
            result = format_result(jobShopEnv, instance)
            result_list.append(result)
        return result_list


def test_solver():
    from envs.msp.env import SchedulingProblemEnv

    problem_type = 'jssp'
    solver_name = 'cp_sat'
    generator = SchedulingProblemGenerator(problem_type)
    env = SchedulingProblemEnv(problem_type)

    instances = generator.generate(20)
    # instances=[
    #     {'is_flow': False, 'is_open': False, 'machines_per_stage': {}, 'num_jobs': 5, 'num_machines': 2,
    #      'precedence_relations': [[('job_1', 'op_3'), ('job_3', 'op_7')]], 'processing_times': {
    #         'job_0': {'op_0': {'machine_1': (3, 'machine_1')}, 'op_1': {'machine_0': (6, 'machine_0')}},
    #         'job_1': {'op_2': {'machine_1': (5, 'machine_1')}, 'op_3': {'machine_0': (10, 'machine_0')}},
    #         'job_2': {'op_4': {'machine_1': (1, 'machine_1')}, 'op_5': {'machine_0': (5, 'machine_0')}},
    #         'job_3': {'op_6': {'machine_0': (8, 'machine_0')}, 'op_7': {'machine_1': (10, 'machine_1')}},
    #         'job_4': {'op_8': {'machine_0': (7, 'machine_0')}, 'op_9': {'machine_1': (7, 'machine_1')}}}}
    # ]
    print(instances)
    jssbei_solver = JSSBEISolver(problem_type, solver_name)
    schedule_list = jssbei_solver.solve(instances)

    reward_list = []
    for instance, schedule in zip(instances, schedule_list):
        env.plot_schedule(schedule)
        print(env.check_valid(instance, schedule))

        reward = env.get_reward(instance, schedule)
        reward_list.append(reward)
    print(np.mean(np.array(reward_list)))

if __name__ == '__main__':
    test_solver()
