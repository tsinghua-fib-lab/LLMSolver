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

import torch

import os
from logging import getLogger

from solver.msp.FFSP_MatNet.FFSPEnv import FFSPEnv as Env
from solver.msp.FFSP_MatNet.FFSPModel import FFSPModel as Model
from solver.msp.FFSP_MatNet.utils import get_result_folder, AverageMeter, TimeEstimator
from solver.msp.FFSP_MatNet.FFSProblemDef import load_problems_from_file


class FFSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

        # Load all problems
        # self.logger.info(" *** Loading Saved Problems *** ")
        # saved_problem_folder = self.tester_params['saved_problem_folder']
        # saved_problem_filename = self.tester_params['saved_problem_filename']
        # filename = os.path.join(saved_problem_folder, saved_problem_filename)
        # self.ALL_problems_INT_list = load_problems_from_file(filename, device=self.device)
        self.logger.info("Done. ")

    def run(self, all_problems_list):

        # save_solution = self.tester_params['save_solution']['enable']
        # solution_list = []

        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['problem_count']
        episode = 0

        aug_score_all = []
        schedule_all = []
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            problems_INT_list = []
            for stage_idx in range(self.env.stage_cnt):
                problems_INT_list.append(all_problems_list[stage_idx][episode:episode + batch_size])

            job_durations = torch.empty(size=(batch_size, self.env.job_cnt + 1, self.env.total_machine_cnt),
                                        dtype=torch.long)
            # shape: (batch, job+1, total_machine)
            job_durations[:, :self.env.job_cnt, :] = torch.cat(problems_INT_list, dim=2)
            job_durations[:, self.env.job_cnt, :] = 0

            score, aug_score, best_schedule = self._test_one_batch(problems_INT_list)

            aug_score_all.extend(aug_score.tolist())
            schedule_all.extend(best_schedule.tolist())

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            # elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            # self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
            #     episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))
            #
            # all_done = (episode == test_num_episode)
            #
            # if all_done:
            #     self.logger.info(" *** Test Done *** ")
            #     self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
            #     self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))
        return aug_score_all, schedule_all

    def _test_one_batch(self, problems_INT_list):

        batch_size = problems_INT_list[0].size(0)

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
            batch_size = aug_factor * batch_size
            for stage_idx in range(self.env.stage_cnt):
                problems_INT_list[stage_idx] = problems_INT_list[stage_idx].repeat(aug_factor, 1, 1)
                # shape: (batch*aug_factor, job_cnt, machine_cnt)
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems_manual(problems_INT_list)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            while not done:
                job_selected, _ = self.model(state)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(job_selected)

            # Return
            ###############################################
            batch_size = batch_size // aug_factor
            aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
            # shape: (augmentation, batch, pomo)

            max_pomo_reward, max_pomo_idx = aug_reward.max(dim=2)  # get best results from pomo
            # shape: (augmentation, batch)
            no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

            max_aug_pomo_reward, max_aug_idx = max_pomo_reward.max(dim=0)  # get best results from augmentation
            # shape: (batch,)
            aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

            schedule = self.env.schedule.reshape(aug_factor, batch_size, *self.env.schedule.shape[1:])
            best_schedule = schedule[
                max_aug_idx,  # (2,) -> 选择最佳 aug
                torch.arange(batch_size),  # (2,) -> 遍历 batch
                max_pomo_idx[max_aug_idx, torch.arange(batch_size)],  # (2,) -> 最佳 pomo 索引
                ...  # 保持剩余所有维度
            ]
            return no_aug_score.item(), -max_aug_pomo_reward, best_schedule
