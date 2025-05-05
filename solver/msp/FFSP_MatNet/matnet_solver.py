"""
The MIT License

Copyright (c) 2021 MatNet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import numpy as np

##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

##########################################################################################
# import

import torch
from typing import Dict, List
from solver.msp.FFSP_MatNet.utils import copy_all_src
from solver.msp.FFSP_MatNet.FFSPTester import FFSPTester as Tester

##########################################################################################
# parameters

env_params = {
    'stage_cnt': 3,
    'machine_cnt_list': [4, 4, 4],
    'job_cnt': 20,
    'process_time_params': {
        'time_low': 2,
        'time_high': 10,
    },
    'pomo_size': 24  # assuming 4 machines at each stage! 4*3*2*1
}

model_params = {
    'stage_cnt': env_params['stage_cnt'],
    'machine_cnt_list': env_params['machine_cnt_list'],
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256 ** (1 / 2),
    'encoder_layer_num': 3,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16 ** (1 / 2),
    'head_num': 16,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1 / 2) ** (1 / 2),
    'ms_layer2_init': (1 / 16) ** (1 / 2),
    'eval_type': 'softmax',
    'one_hot_seed_cnt': 4,  # must be >= machine_cnt
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/saved_ffsp20_model',  # directory path of pre-trained model and log files saved.
        'epoch': 100,  # epoch version of pre-trained model to load.
    },
    'problem_count': 1,
    'test_batch_size': 1,
    'augmentation_enable': True,
    'aug_factor': 128,
    'aug_batch_size': 200,
}

model_load_dict = {
    20: {
        'path': os.path.join(os.path.dirname(os.path.realpath(__file__)), "result", "saved_ffsp20_model"),
        # directory path of pre-trained model and log files saved.
        'epoch': 100,  # epoch version of pre-trained model to load.
    },
    50: {
        'path': os.path.join(os.path.dirname(os.path.realpath(__file__)), "result", "saved_ffsp50_model"),
        # directory path of pre-trained model and log files saved.
        'epoch': 150,  # epoch version of pre-trained model to load.
    },
    100: {
        'path': os.path.join(os.path.dirname(os.path.realpath(__file__)), "result", "saved_ffsp100_model"),
        # directory path of pre-trained model and log files saved.
        'epoch': 200,  # epoch version of pre-trained model to load.
    }
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


def prepare_env(**params):
    stage_cnt = params.get('stage_cnt', env_params['stage_cnt'])
    machine_cnt_list = params.get('machine_cnt_list', env_params['machine_cnt_list'])
    job_cnt = params.get('job_cnt', env_params['job_cnt'])
    assert stage_cnt in [env_params['stage_cnt']]
    assert machine_cnt_list in [env_params['machine_cnt_list']]

    model_key_list = np.array(list(model_load_dict.keys()))
    model_key = model_key_list[(np.abs(model_key_list - job_cnt)).argmin()]

    env_params['job_cnt'] = job_cnt
    env_params['stage_cnt'] = stage_cnt
    env_params['machine_cnt_list'] = machine_cnt_list
    model_params['stage_cnt'] = stage_cnt
    model_params['machine_cnt_list'] = machine_cnt_list
    model_params['job_cnt'] = job_cnt

    tester_params['model_load'] = model_load_dict[model_key]
    print(tester_params['model_load'])
    _tester = Tester(env_params=env_params,
                     model_params=model_params,
                     tester_params=tester_params)

    return _tester


##########################################################################################
# format

def format_instance(instance: Dict):
    processing_times = instance['processing_times']
    # Get dimensions
    num_jobs = instance['num_jobs']
    machines_per_stage = instance['machines_per_stage']

    # Initialize 3D list (stage × machine × job)
    processing_time_matrix = [
        [
            [0 for machine_id in range(machines_per_stage[stage])]
            for job_id in range(num_jobs)
        ]
        for stage in machines_per_stage
    ]

    stage_cumulative_machines = np.concatenate(([0], np.cumsum(list(machines_per_stage.values()))[:-1]))

    # Fill the matrix
    for job_id, job_data in processing_times.items():
        job_idx = int(job_id.split('_')[-1])  # Extract '0' from 'job_0'
        for stage_id, stage_data in job_data.items():
            stage_idx = int(stage_id.split('_')[-1])  # Extract '0' from 'stage_0'
            for machine_id, (proc_time, _) in stage_data.items():  # Ignore setup time
                machine_idx = int(machine_id.split('_')[-1]) - stage_cumulative_machines[stage_idx]
                # Extract '0' from 'machine_0'
                processing_time_matrix[stage_idx][job_idx][machine_idx] = proc_time

    processing_time_matrix = torch.tensor(processing_time_matrix).unsqueeze(1)
    return processing_time_matrix


def format_result(schedule, score, instance, machines_per_stage):
    global_to_stage = []
    for stage_id, num_machines in enumerate(machines_per_stage.values()):
        for local_machine_id in range(num_machines):
            global_to_stage.append(stage_id)

    processing_times = instance['processing_times']
    # Get dimensions
    num_jobs = instance['num_jobs']
    machines_per_stage = instance['machines_per_stage']
    num_machines = sum(machines_per_stage.values())

    output_schedule = [{'job': job, 'tasks': []} for job in range(num_jobs)]

    for global_machine_id in range(num_machines):
        for job_id in range(num_jobs):
            start_time = schedule[global_machine_id][job_id]

            if start_time >= 0:  # 如果 job 在 machine 上执行
                task_id = int(len(output_schedule[job_id]['tasks']))
                stage_id = global_to_stage[global_machine_id]
                output_schedule[job_id]['tasks'].append({
                    'duration':
                        processing_times[f'job_{job_id}'][f'stage_{stage_id}'][f'machine_{global_machine_id}'][0],
                    'machine': global_machine_id,
                    'start': start_time,
                    'task': task_id
                })

    result = {'schedule': output_schedule, 'obj_value': score}
    return result


##########################################################################################
# main

class MatNetSolver:
    def __init__(self):
        job_cnt = 50
        self._tester = prepare_env(job_cnt=job_cnt, )

    def solve(self, instances: List[Dict], **params) -> List[Dict]:
        result_list = []
        for instance in instances:
            job_cnt = instance['num_jobs']
            machines_per_stage = instance['machines_per_stage']
            # _tester = prepare_env(job_cnt=job_cnt, )
            # copy_all_src(_tester.result_folder)
            matnet_instance = format_instance(instance)
            self._tester.env.job_cnt = job_cnt

            self._tester.run(matnet_instance)
            score, schedule = self._tester.run(matnet_instance)
            result = format_result(schedule[0], score[0], instance, machines_per_stage)
            result_list.append(result)
        return result_list


##########################################################################################
# test

def test_solver():
    from envs.msp.generator import SchedulingProblemGenerator
    from envs.msp.env import SchedulingProblemEnv

    problem_type = 'hfssp'
    generator = SchedulingProblemGenerator(problem_type, )
    env = SchedulingProblemEnv(problem_type)
    instances = generator.generate(batch_size=1)

    print(instances)
    matnet_solver = MatNetSolver()
    schedule_list = matnet_solver.solve(instances)

    reward_list = []
    for instance, schedule in zip(instances, schedule_list):
        reward = env.get_reward(instance, schedule)
        reward_list.append(reward)
    print(np.mean(np.array(reward_list)))


if __name__ == '__main__':
    test_solver()
