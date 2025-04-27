from msp_solver_pool import SchedulingSolverPool
from envs.msp.env import validate_schedule, plot_schedule, calculate_makespan
from envs.msp.generator import SchedulingProblemGenerator, SchedulingProblemType

if __name__ == '__main__':
    solver_pool = SchedulingSolverPool()
    problem_type = 'hfssp'
    solver_name = solver_pool.get_solver_list(problem_name=problem_type)[0]

    if problem_type == 'hfssp':
        job_cnt = 21
        generator = SchedulingProblemGenerator(SchedulingProblemType.HFSSP,
                                               min_jobs=job_cnt,
                                               max_jobs=job_cnt, )

        machines_per_stage = {
            f'stage_{0}': 4,
            f'stage_{1}': 4,
            f'stage_{2}': 4,
        }

        instances = generator.generate_problem_instance(machines_per_stage=machines_per_stage)
        schedules = solver_pool.solve(instances=instances, solver_name=solver_name)
        print(schedules)

        is_valid, message = validate_schedule(schedules[0])
        print(f"Validation: {is_valid}, Message: {message}")
        plot_schedule(schedules[0])

        makespan = calculate_makespan(schedules[0])
        print(f"Makespan: {makespan}")
        print(schedules[0]['objValue'])