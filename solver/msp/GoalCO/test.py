"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import argparse
import os
import time
from args import add_common_args, add_common_training_args
from solver.msp.GoalCO.learning.data_iterators import DataIterator
from solver.msp.GoalCO.learning.tester import Tester
from solver.msp.GoalCO.utils.exp import setup_experiment, setup_test_environment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    args.pretrained_model = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/multi.best')
    args.problems = ['jssp']
    args.test_datasets = ['jssp']
    args.test_datasets = [os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/jssp10_10_test.npz')]

    setup_experiment(args)

    net = setup_test_environment(args)

    data_iterator = DataIterator(args, ddp=False)

    tester = Tester(args, net, data_iterator.test_datasets)
    tester.load_model(args.pretrained_model)

    start_time = time.time()
    tester.test()
    print("inference time", time.time() - start_time)
