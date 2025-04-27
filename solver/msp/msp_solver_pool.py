class SchedulingSolverPool:
    solver_problem_dict = {
        'matnet': ['hfssp'],
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
        self.solver_dict = {}
        self.device = kwargs.get("device", "cpu")

        for solver_name in self.solver_problem_dict.keys():
            try:
                if solver_name == "matnet":
                    import solver.msp.FFSP_MatNet.matnet_solver as matnet_solver
                    self.solver_dict[solver_name] = matnet_solver

                for problem_name in self.solver_problem_dict[solver_name]:
                    if problem_name not in self.problem2solver_dict:
                        self.problem2solver_dict[problem_name] = []
                    self.problem2solver_dict[problem_name].append(solver_name)
            except ImportError:
                print(f"WARNING: {solver_name} is not exist.")
            except Exception as e:
                print(e)

        print(f"Avaliable solvers: ")
        for problem_name in self.problem2solver_dict.keys():
            print(f"\t{problem_name}: {self.problem2solver_dict[problem_name]}")

    def get_problem_list(self) -> list | None:
        return list(self.problem2solver_dict.keys())

    def get_solver_list(self, problem_name: str) -> list | None:
        solver_list = list(self.problem2solver_dict.get(problem_name, []))
        return solver_list

    def _solve(
            self,
            instances: dict,
            solver_name: str = "matnet",
            problem_type: str = "hfssp",
            max_runtime: float = 30,
            num_procs: int = 1,
            **kwargs,
    ) -> dict:
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

        if solver_name not in self.solver_dict:
            raise ValueError(f"Unknown baseline solver: {solver_name}")

        _solve_func = self.solver_dict[solver_name]

        results = _solve_func.solve(instances=instances)

        return results

    def solve(self, instances: dict, solver_name: str = "matnet", problem_type: str = "hfssp",
              timeout: int = 30, num_procs=32, **kwargs):
        try:
            score = self._solve(instances=instances, solver_name=solver_name, problem_type=problem_type,
                                max_runtime=timeout, num_procs=num_procs, **kwargs)
        except Exception as e:
            print(e)
            return "<RuntimeError>"
        return score
