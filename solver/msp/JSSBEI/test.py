from solver.msp.JSSBEI.data.data_parsers.custom_instance_parser import parse
from solver.msp.JSSBEI.visualization import gantt_chart, precedence_chart

from solver.msp.JSSBEI.solution_methods.GA.src.initialization import initialize_run
from solver.msp.JSSBEI.solution_methods.GA.run_GA import run_GA
from solver.msp.JSSBEI.solution_methods.cp_sat.run_cp_sat import run_CP_SAT

from envs.msp.generator import SchedulingProblemType, SchedulingProblemGenerator
from envs.msp.convert import convert_to_target_format
if __name__ == '__main__':
    generator = SchedulingProblemGenerator(SchedulingProblemType.HFSSP)
    processing_info = generator.generate_problem_instance()
    processing_info = convert_to_target_format(processing_info)
    print(processing_info)

    jobShopEnv = parse(processing_info)
    precedence_chart.plot(jobShopEnv)

    parameters = {"instance": {"problem_instance": "custom_problem_instance"},
                  "solver": {"time_limit": 3600, "model": "fajsp"},
                  "output": {"logbook": True}
                  }

    jobShopEnv = parse(processing_info)
    results, jobShopEnv = run_CP_SAT(jobShopEnv, **parameters)

    plt = gantt_chart.plot(jobShopEnv)
    plt.show()
