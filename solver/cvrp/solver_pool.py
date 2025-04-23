import os
import torch
from torch import Tensor
from tensordict.tensordict import TensorDict
from rl4co.utils.ops import gather_by_index, unbatchify
from functools import partial
from multiprocessing import Pool

from envs.cvrp.mtvrp import MTVRPEnv
from envs.cvrp.utils import rollout, greedy_policy
from solver.cvrp.model_utils import get_policy, get_model
from solver.cvrp.utils import mtvrp2anyvrp


class NoSolver:
    def solve(self, *args, **kwargs):
        pass


class SolverPool:
    model_solver_problem_dict = {
        'mtpomo': ['cvrp', 'ovrp', 'vrpb', 'vrpl', 'vrptw'],
        'rf-pomo': ['cvrp', 'ovrp', 'vrpb', 'vrpl', 'vrptw'],
        'rf-moe': ['cvrp', 'ovrp', 'vrpb', 'vrpl', 'vrptw'],
        'rf-transformer': ['cvrp', 'ovrp', 'vrpb', 'vrpl', 'vrptw'],
    }
    algo_solver_problem_dict = {
        "lkh": ['cvrp', 'ovrp', 'vrpb', 'vrpl', 'vrptw'],
        "pyvrp": ['cvrp', 'ovrp', 'vrpb', 'vrpl', 'vrptw'],
        "ortools": ['cvrp', 'ovrp', 'vrpb', 'vrpl', 'vrptw'],
    }

    def __init__(self, **kwargs):
        """
        Initializes the RL Agent with optional parameters.

        Parameters:
        ----------
        **kwargs : dict
            Arbitrary keyword arguments. Expected keys:
            - env (str or object, optional): The environment for the RL agent (e.g., "MTVRPEnv()").
            - policy_dir (str, optional): Path to the directory storing policy files. Defaults to "./model_checkpoints/100/".
            - device (str, optional): Device to run the policy on. Defaults to "cpu".
        """
        self.problem2solver_dict = {}
        self.model_solver_dict = {}
        self.algo_solver_dict = {}
        self.device = kwargs.get("device", "cpu")

        for solver_name in self.algo_solver_problem_dict.keys():
            try:
                if solver_name == "pyvrp":
                    import solver.cvrp.pyvrp_solver.pyvrp_solver as pyvrp_solver
                    self.algo_solver_dict["pyvrp"] = pyvrp_solver.solve
                elif solver_name == "lkh":
                    import solver.cvrp.lkh_solver.lkh_solver as lkh_solver
                    self.algo_solver_dict["lkh"] = lkh_solver.solve
                    self.lkh_path = kwargs.get("lkh_path", "lkh_solver/LKH-3.0.13/LKH/")
                    self.lkh_num_runs = kwargs.get("lkh_num_runs", 10)
                    self.lkh_max_trials = kwargs.get("lkh_max_trials", 10000)
                elif solver_name == "ortools":
                    import solver.cvrp.ortools_solver.ortools_solver as ortools_solver
                    self.algo_solver_dict["ortools"] = ortools_solver.solve
                for problem_name in self.algo_solver_problem_dict[solver_name]:
                    if problem_name not in self.problem2solver_dict:
                        self.problem2solver_dict[problem_name] = []
                    self.problem2solver_dict[problem_name].append(solver_name)
            except ImportError:
                print(f"WARNING: {solver_name} is not installed. Please install it using `pip install -e .[solvers]`.")
            except Exception as e:
                print(e)

        for solver_name in self.model_solver_problem_dict.keys():
            try:
                if solver_name == "rf-moe":
                    pass
                policy = get_policy(solver_name)
                if policy is None:
                    raise ImportError
                policy_dir = kwargs.get("policy_dir", "model_checkpoints/100")
                policy.load_state_dict(torch.load(os.path.join(policy_dir, f"{solver_name}.pth")))
                self.model_solver_dict[solver_name] = policy
                self.model_solver_dict[solver_name].to(self.device)

                for problem_name in self.model_solver_problem_dict[solver_name]:
                    if problem_name not in self.problem2solver_dict:
                        self.problem2solver_dict[problem_name] = []
                    self.problem2solver_dict[problem_name].append(solver_name)

            except ImportError:
                print(f"WARNING:{solver_name} is not installed.")
            except Exception as e:
                print(f"ERROR:{solver_name} has error {e}.")

        print(f"Avaliable solvers: ")
        for solver_name in self.algo_solver_dict.keys():
            print(f"\t{solver_name}: {self.algo_solver_problem_dict[solver_name]}")
        for solver_name in self.model_solver_dict.keys():
            print(f"\t{solver_name}: {self.model_solver_problem_dict[solver_name]}")

    def get_problem_list(self) -> list | None:
        return list(self.problem2solver_dict.keys())

    @classmethod
    def get_solver_list(cls, problem_name: str) -> list | None:
        algo_solver_list = list(cls.algo_solver_problem_dict.get(problem_name, []))
        model_solver_list = list(cls.model_solver_problem_dict.get(problem_name, []))
        return algo_solver_list + model_solver_list

    def algo_solve(
            self,
            instances: TensorDict,
            solver_name: str = "pyvrp",
            problem_type: str = "cvrp",
            max_runtime: float = 30,
            num_procs: int = 1,
            data_type: str = "mtvrp",
            **kwargs,
    ) -> Tensor:
        """
        Solves the AnyVRP instances with PyVRP.

        Parameters
        ----------
        instances
            TensorDict containing the AnyVRP instances to solve.
        max_runtime
            Maximum runtime for the solver.
        num_procs
            Number of processers to use to solve instances in parallel.
        data_type
            Environment mode. If "mtvrp", the instance data will be converted first.
        solver_name
            The solver to use. One of ["pyvrp", "ortools", "lkh"].

        Returns
        -------
        tuple[Tensor, Tensor]
            A Tensor containing the actions for each instance and a Tensor
            containing the corresponding costs.
        """
        if data_type == "mtvrp":
            instances = mtvrp2anyvrp(instances)

        if solver_name not in self.algo_solver_dict:
            raise ValueError(f"Unknown baseline solver: {solver_name}")

        _solve = self.algo_solver_dict[solver_name]
        if solver_name == "lkh":
            kwargs["num_runs"] = self.lkh_num_runs
            kwargs["max_trials"] = self.lkh_max_trials
            kwargs["solver_loc"] = self.lkh_path

        func = partial(_solve, max_runtime=max_runtime, problem_type=problem_type, **kwargs)

        if num_procs > 1:
            with Pool(processes=num_procs) as pool:
                results = pool.map(func, instances)
        else:
            results = [func(instance) for instance in instances]

        actions = results

        # Pad to ensure all actions have the same length.
        max_len = max(len(action) for action in actions)
        actions = [action + [0] * (max_len - len(action)) for action in actions]

        return Tensor(actions).long()

    def model_solve(
            self,
            instances: TensorDict,
            solver_name: str = "rf-transformer",
            problem_type: str = "cvrp",
            num_augment=8,
            num_starts=None,
            **kwargs,
    ):
        policy = self.model_solver_dict[solver_name]
        instances = instances.to(self.device)
        env = kwargs.get("env", MTVRPEnv())

        with torch.inference_mode():
            n_start = env.get_num_starts(instances) if num_starts is None else num_starts

            if num_augment > 1:
                model = get_model(solver_name, env, policy)
                instances = model.augment(instances)

            # Evaluate policy
            out = policy(
                instances, env, phase="test", num_starts=n_start, return_actions=True
            )

            # Unbatchify reward to [batch_size, num_augment, num_starts].
            reward = unbatchify(out["reward"], (num_augment, n_start))

            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (num_augment, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if num_augment > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

            return out["best_aug_actions"]

    def _solve(self, instances: TensorDict, solver_name: str = "rf-transformer", problem_type="cvrp",
               max_runtime: int = 30, num_procs=32, **kwargs):
        if solver_name == 'greedy':
            env = kwargs.get("env", MTVRPEnv())
            return rollout(env, instances, greedy_policy)
        elif solver_name in self.model_solver_dict:
            return self.model_solve(instances=instances, solver_name=solver_name, problem_type=problem_type, **kwargs)
        elif solver_name in self.algo_solver_dict:
            return self.algo_solve(max_runtime=max_runtime, instances=instances, solver_name=solver_name,
                                   problem_type=problem_type, num_procs=num_procs, **kwargs)
        else:
            raise ValueError(f"Unknown solver: {solver_name}")

    def solve(self, instances: TensorDict, solver_name: str = "rf-transformer", problem_type: str = "cvrp",
              timeout: int = 30, num_procs=32, **kwargs):
        try:
            score = self._solve(instances=instances, solver_name=solver_name, problem_type=problem_type,
                                max_runtime=timeout, num_procs=num_procs, **kwargs)
        except Exception as e:
            print(e)
            return "<RuntimeError>"
        return score
