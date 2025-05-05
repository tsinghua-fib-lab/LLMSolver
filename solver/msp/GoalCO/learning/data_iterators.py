"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

from argparse import Namespace
from solver.msp.GoalCO.learning.tsp.dataset import load_dataset as load_tsp_dataset
from solver.msp.GoalCO.learning.pctsp.dataset import load_dataset as load_pctsp_dataset
from solver.msp.GoalCO.learning.cvrp.dataset import load_dataset as load_cvrp_dataset
from solver.msp.GoalCO.learning.cvrptw.dataset import load_dataset as load_cvrptw_dataset
from solver.msp.GoalCO.learning.op.dataset import load_dataset as load_op_dataset
from solver.msp.GoalCO.learning.kp.dataset import load_dataset as load_kp_dataset
from solver.msp.GoalCO.learning.mvc.dataset import load_dataset as load_mvc_dataset
from solver.msp.GoalCO.learning.mis.dataset import load_dataset as load_mis_dataset
from solver.msp.GoalCO.learning.upms.dataset import load_dataset as load_upms_dataset
from solver.msp.GoalCO.learning.jssp.dataset import load_dataset as load_jssp_dataset
from solver.msp.GoalCO.learning.mclp.dataset import load_dataset as load_mclp_dataset


class DataIterator:

    def __init__(self, args: Namespace, ddp: bool = False):

        loaders = {"tsp": load_tsp_dataset,
                   "trp": load_tsp_dataset,
                   "sop": load_tsp_dataset,
                   "pctsp": load_pctsp_dataset,
                   "cvrp": load_cvrp_dataset,
                   "sdcvrp": load_cvrp_dataset,
                   "ocvrp": load_cvrp_dataset,
                   "dcvrp": load_cvrp_dataset,
                   "cvrptw": load_cvrptw_dataset,
                   "op": load_op_dataset,
                   "mclp": load_mclp_dataset,
                   "kp": load_kp_dataset,
                   "mvc": load_mvc_dataset,
                   "mis": load_mis_dataset,
                   "upms": load_upms_dataset,
                   "jssp": load_jssp_dataset,
                   "ossp": load_jssp_dataset}

        self.train_datasets = dict()
        if "train_datasets" in args and args.train_datasets is not None:
            assert len(args.problems) == len(args.train_datasets)
            for problem_no, problem in enumerate(args.problems):
                self.train_datasets[problem] = loaders[problem](args.train_datasets[problem_no],
                                                                args.train_batch_size, args.train_datasets_size,
                                                                True, True, "train", ddp)
        else:
            for problem_no, problem in enumerate(args.problems):
                self.train_datasets[problem] = None

        if "val_datasets" in args and args.val_datasets is not None:
            assert len(args.problems) == len(args.val_datasets)
            self.val_datasets = dict()
            for problem_no, problem in enumerate(args.problems):
                self.val_datasets[problem] = loaders[problem](args.val_datasets[problem_no], args.val_batch_size,
                                                              args.val_datasets_size, False, False, "val", False)
        assert len(args.problems) == len(args.test_datasets)
        self.test_datasets = dict()
        for problem_no, problem in enumerate(args.problems):
            self.test_datasets[problem] = loaders[problem](args.test_datasets[problem_no], args.test_batch_size,
                                                           args.test_datasets_size, False, False, "test", False)
