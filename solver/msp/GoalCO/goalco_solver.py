import argparse
import os
import time
from typing import Dict, List

import numpy as np

from solver.msp.GoalCO.args import add_common_args, add_common_training_args
from solver.msp.GoalCO.learning.data_iterators import DataIterator
from solver.msp.GoalCO.learning.tester import Tester
from solver.msp.GoalCO.utils.exp import setup_experiment, setup_test_environment

##########################################################################################
# parameters

model_path_dict = {
    'jssp': 'checkpoint/multi.best',
    'ossp': 'checkpoint/multi.best',
}


##########################################################################################
# format

def format_instance(instances: List[Dict]):
    datasets_size = len(instances)

    num_jobs = instances[0]['num_jobs']
    num_machines = instances[0]['num_machines']

    execution_times_list = []
    task_on_machines_list = []
    for instance in instances:
        inst_num_jobs = instance['num_jobs']
        inst_num_machines = instance['num_machines']
        assert (inst_num_jobs == num_jobs)
        assert (inst_num_machines == num_machines)

        processing_times = instance['processing_times']
        job_ids = sorted(processing_times.keys(), key=lambda x: int(x.split('_')[-1]))
        task_list = []

        for job_id in job_ids:
            ops = processing_times[job_id]
            op_ids = sorted(ops.keys(), key=lambda x: int(x.split('_')[-1]))
            for op_id in op_ids:
                assert len(list(ops[op_id].values())) == 1
                op = list(ops[op_id].values())[0]
                processing_time = op[0]
                machine_id = int(op[1].split('_')[-1])
                task_list.append((machine_id, processing_time))

        task_on_machines_list.append([m for m, _ in task_list])
        execution_times_list.append([p for _, p in task_list])

    # Convert to numpy arrays
    execution_times = np.array(execution_times_list)
    task_on_machines = np.array(task_on_machines_list)
    scales = np.ones((datasets_size, 1)) * np.max(execution_times)
    optimal_values = np.zeros(datasets_size)

    data = {
        "execution_times": execution_times,
        "task_on_machines": task_on_machines,
        "scales": scales,
        "optimal_values": optimal_values,
        "num_jobs": np.array(num_jobs),
        "num_machines": np.array(num_machines),
    }

    return data


def format_result(schedule, instance):
    num_machines = instance['num_machines']
    num_jobs = instance['num_jobs']
    processing_times = instance['processing_times']

    output_schedule = [{'job': job, 'tasks': []} for job in range(num_jobs)]
    for machine_id in range(num_machines):
        for task in schedule[machine_id]:
            job_id, op_idx, start_time, end_time = task
            job_id = int(job_id)
            op_id = sorted(processing_times[f'job_{job_id}'].keys(), key=lambda x: int(x.split('_')[-1]))[int(op_idx)]

            start_time = float(start_time)
            end_time = float(end_time)
            duration = processing_times[f'job_{job_id}'][op_id][f'machine_{machine_id}'][0]

            # if job_id not in output_schedule:
            #     output_schedule[job_id] = {'tasks': []}

            output_schedule[job_id]['tasks'].append({
                'duration': duration,
                'machine': machine_id,
                'start': start_time,
                'task': op_id
            })
    for job_id in range(num_jobs):
        output_schedule[job_id]['tasks'].sort(key=lambda x: x['task'])

    result = {'schedule': output_schedule}
    return result


##########################################################################################
# main
class GoalCOSolver:
    def __init__(self, problem_type: str):
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args()
        assert problem_type in model_path_dict
        args.pretrained_model = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_path_dict[problem_type])
        args.problems = [problem_type]
        # args.test_datasets = [os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/jssp10_10_test.npz')]
        args.test_datasets = [problem_type]
        args.test_batch_size = 1
        if problem_type in ['tsp', 'cvrp', 'cvrptw', 'op', 'kp', 'upms', 'jssp']:
            args.is_finetuning = False
        else:
            args.is_finetuning = True
        setup_experiment(args)
        net = setup_test_environment(args)

        self.problem_type = problem_type
        self.args = args
        self.tester = Tester(args, net)
        self.tester.load_model(args.pretrained_model)

    def solve(self, instances: List[Dict], **params) -> List[Dict]:
        result_list = []
        for instance in instances:
            test_data = format_instance([instance])
            data_iterator = DataIterator(self.args, ddp=False, test_data=test_data)
            metrics_per_problem, solutions_per_problem = self.tester.test(test_dataset=data_iterator.test_datasets)
            result = format_result(solutions_per_problem[self.problem_type][0], instance)
            result_list.append(result)
        return result_list


def test_solver():
    from envs.msp.env import SchedulingProblemEnv
    from envs.msp.generator import SchedulingProblemGenerator

    problem_type = 'ossp'
    generator = SchedulingProblemGenerator(problem_type)
    env = SchedulingProblemEnv(problem_type)
    instances = generator.generate(1)
    goalco_solver = GoalCOSolver(problem_type)
    schedule_list = goalco_solver.solve(instances)
    reward_list = []
    for instance, schedule in zip(instances, schedule_list):
        reward = env.get_reward(instance, schedule)
        reward_list.append(reward)
    print(np.mean(np.array(reward_list)))
    env.plot_schedule(schedule_list[0])


if __name__ == '__main__':
    test_solver()
