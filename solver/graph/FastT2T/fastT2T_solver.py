import os
from argparse import ArgumentParser

import numpy as np
import torch
from typing import Dict, List
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

from solver.graph.FastT2T.diffusion.pl_tsp_model import TSPModel
from solver.graph.FastT2T.diffusion.pl_mis_model import MISModel
from envs.graph.generator import GraphGenerator, data2graph

torch.cuda.amp.autocast(enabled=True)
torch.cuda.empty_cache()
import warnings

warnings.filterwarnings("ignore")


##########################################################################################
# parameters

def arg_parser():
    parser = ArgumentParser(
        description="Train a Pytorch-Lightning diffusion model on a TSP dataset."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", type=str, default="mis")
    parser.add_argument("--storage_path", type=str, default="./")
    parser.add_argument("--training_split", type=str, default=None)
    parser.add_argument(
        "--training_split_label_dir",
        type=str,
        default=None,
        help="Directory containing labels for training split (used for MIS).",
    )
    parser.add_argument("--validation_split", type=str, default=None)
    parser.add_argument(
        "--validation_split_label_dir",
        type=str,
        default=None,
        help="Directory containing labels for validation split (used for MIS).",
    )
    parser.add_argument("--test_split", type=str, default=None)
    parser.add_argument(
        "--test_split_label_dir",
        type=str,
        default=None,
        help="Directory containing labels for test split (used for MIS).",
    )
    parser.add_argument("--validation_examples", type=int, default=64)
    parser.add_argument(
        "--graph_type",
        type=str,
        default="undirected",
        choices=["undirected", "directed"],
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", type=str, default="constant")

    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_activation_checkpoint", action="store_true")

    parser.add_argument("--diffusion_schedule", type=str, default="linear")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--inference_diffusion_steps", type=int, default=1000)
    parser.add_argument("--inference_schedule", type=str, default="cosine")
    parser.add_argument("--inference_trick", type=str, default="ddim")
    parser.add_argument("--sequential_sampling", type=int, default=1)
    parser.add_argument("--parallel_sampling", type=int, default=1)

    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--sparse_factor", type=int, default=-1)
    parser.add_argument("--aggregation", type=str, default="sum")
    parser.add_argument("--two_opt_iterations", type=int, default=0)
    parser.add_argument("--save_numpy_heatmap", action="store_true")

    parser.add_argument("--project_name", type=str, default="tsp_diffusion")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_logger_name", type=str, default=None)
    parser.add_argument(
        "--resume_id", type=str, default=None, help="Resume training on wandb."
    )
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--resume_weight_only", action="store_true")

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument("--rewrite_ratio", type=float, default=0.25)
    parser.add_argument("--norm", action="store_true", default=False)
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--rewrite_steps", type=int, default=1)
    parser.add_argument("--steps_inf", type=int, default=1)
    parser.add_argument("--guided", action="store_true")

    parser.add_argument(
        "--consistency", action="store_true", help="used for consistency training"
    )
    parser.add_argument("--boundary_func", default="truncate")
    parser.add_argument("--alpha", type=float)

    parser.add_argument(
        "--use_intermediate",
        action="store_true",
        help="set true when use intermediate x0 to decode tours",
    )
    parser.add_argument("--c1", type=float, default=50, help="coefficient of F1")
    parser.add_argument("--c2", type=float, default=50, help="coefficient of F2")

    parser.add_argument(
        "--offline", action="store_true", help="set true when use offline wandb"
    )

    args = parser.parse_args()
    return args


##########################################################################################
# main

class FastT2TSolver:
    def __init__(self, problem_type: str, **params):
        self.problem_type = problem_type
        assert problem_type == "mis"
        graph_model = params.get('graph_model', 'rb')
        assert graph_model in ["er", "rb", "sat"]

        hidden_dim_dict = {
            "er": 128,
            "rb": 256,
        }
        args = arg_parser()
        args.do_test = True
        args.inference_schedule = "cosine"
        args.inference_diffusion_steps = 5
        args.ckpt_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoint", "mis",
                                      f"mis_{graph_model}.ckpt")
        args.resume_weight_only = True
        args.parallel_sampling = 1
        args.sequential_sampling = 1
        args.consistency = True
        args.hidden_dim = hidden_dim_dict[graph_model]
        args.rewrite = True
        args.guided = True
        args.rewrite_steps = 5
        args.rewrite_ratio = 0.2
        args.c1 = 2
        args.c2 = 2
        self.args = args

    def _solve(self, graph_list):
        epochs = self.args.num_epochs

        if self.args.task == "tsp":
            model_class = TSPModel
            saving_mode = "min"
        elif self.args.task == "mis":
            model_class = MISModel
            saving_mode = "max"
        else:
            raise NotImplementedError

        model = model_class(param_args=self.args, test_graph_list=graph_list)

        rank_zero_info(self.args)

        checkpoint_callback = ModelCheckpoint(
            monitor="val/solved_cost",
            mode=saving_mode,
            save_top_k=1,
            save_last=True,
        )
        lr_callback = LearningRateMonitor()
        trainer = Trainer(
            accelerator="auto",
            # devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
            devices=1,
            max_epochs=epochs,
            callbacks=[TQDMProgressBar(refresh_rate=1), checkpoint_callback, lr_callback],
            check_val_every_n_epoch=1,
            strategy=DDPStrategy(static_graph=True),
            precision=16 if self.args.fp16 else 32,
            inference_mode=False,
        )
        ckpt_path = self.args.ckpt_path
        trainer.test(model, ckpt_path=ckpt_path)
        return model.test_predictions

    def solve(self, instances: List[Dict] = None, ):
        graph_list = [data2graph(instance) for instance in instances]
        solutions = self._solve(graph_list)
        node_solutions = [np.where(row == 1)[0].tolist() for row in solutions]
        return node_solutions


def test_solve():
    from envs.graph.env import GraphEnv

    problem_type = 'mis'
    generator = GraphGenerator(problem_type=problem_type)
    grapg_env = GraphEnv(problem_type=problem_type)
    instances = generator.generate(10)
    fast_t2t_solver = FastT2TSolver(problem_type=problem_type)
    node_solutions = fast_t2t_solver.solve(instances, )
    print(node_solutions)
    rewards = []
    for instance, solution in zip(instances, node_solutions):
        reward = grapg_env.get_reward(instance, solution, problem_type)
        rewards.append(reward)
    print(np.mean(rewards))
    return node_solutions


if __name__ == '__main__':
    test_solve()
