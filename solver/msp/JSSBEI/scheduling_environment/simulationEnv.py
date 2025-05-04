import random
from typing import Dict, Optional

import simpy

from scheduling_environment.job import Job
from scheduling_environment.jobShop import JobShop
from scheduling_environment.machine import Machine
from scheduling_environment.operation import Operation


class SimulationEnv:
    """
    Main scheduling_environment class for the an online job shop (with online job arrivals)
    """

    def __init__(self, online_arrivals: bool):
        self.simulator = simpy.Environment()
        self.jobShopEnv = JobShop()
        self.online_arrivals = online_arrivals
        self.machine_resources = []
        self.processed_operations = set()

        # Parameters related to online job arrivals
        self.inter_arrival_time: Optional[int] = None
        self.min_nr_operations_per_job = None
        self.max_nr_operations_per_job = None
        self.min_duration_per_operation = None
        self.max_duration_per_operation = None

    def set_online_arrival_details(self, parameters: Dict[str, int]) -> None:
        """set interarrival time of online jobs."""
        self.inter_arrival_time = parameters['inter_arrival_time']
        self.min_nr_operations_per_job = parameters['min_nr_operations_per_job']
        self.max_nr_operations_per_job = parameters['max_nr_operations_per_job']
        self.min_duration_per_operation = parameters['min_duration_per_operation']
        self.max_duration_per_operation = parameters['max_duration_per_operation']

    def add_machine_resources(self) -> None:
        """Add a machine to the environment."""
        self.machine_resources.append(simpy.Resource(self.simulator, capacity=1))

    def perform_operation(self, operation, machine):
        """Perform operation on the machine (block resource for certain amount of time)"""
        if machine.machine_id in operation.processing_times:
            with self.machine_resources[machine.machine_id].request() as req:
                yield req
                if machine.scheduled_operations != []:
                    if self.jobShopEnv._sequence_dependent_setup_times != []:
                        setup_time = self.jobShopEnv._sequence_dependent_setup_times[machine.machine_id][machine.scheduled_operations[-1].operation_id][
                            operation.operation_id]
                    else:
                        setup_time = 0
                else:
                    setup_time = 0
                start_time = self.simulator.now + setup_time
                processing_time = operation.processing_times[machine.machine_id]
                machine.add_operation_to_schedule_at_time(operation, start_time, processing_time, setup_time)
                yield self.simulator.timeout(processing_time + setup_time - 0.0001)
                self.processed_operations.add(operation)

    def generate_online_job_arrivals(self):
        """generate online arrivals of jobs (online arrivals==True)"""
        job_id = 0
        operation_id = 0

        for id_machine in range(0, self.jobShopEnv.nr_of_machines):
            self.jobShopEnv.add_machine((Machine(id_machine)))
            self.add_machine_resources()

        while True:
            inter_arrival_time = random.expovariate(1.0 / self.inter_arrival_time)
            yield self.simulator.timeout(inter_arrival_time)

            # Job generation logic
            counter = 0
            job = Job(job_id)

            # Add some logic to generate operations for each job
            num_operations = random.randint(self.min_nr_operations_per_job,
                                            self.max_nr_operations_per_job)
            for i in range(num_operations):
                operation = Operation(job, job_id, operation_id)
                self.jobShopEnv.add_operation(operation)
                for machine_id in range(self.jobShopEnv.nr_of_machines):
                    duration = random.randint(self.min_duration_per_operation,
                                              self.max_duration_per_operation)
                    operation.add_operation_option(machine_id, duration)
                job.add_operation(operation)
                if counter != 0:
                    self.jobShopEnv.precedence_relations_operations[operation_id] = [
                        self.jobShopEnv.get_operation(operation_id - 1)]
                    operation.add_predecessors([self.jobShopEnv.get_operation(operation_id - 1)])
                else:
                    self.jobShopEnv.precedence_relations_operations[operation_id] = []

                counter += 1
                operation_id += 1

            self.jobShopEnv.add_job(job)
            self.jobShopEnv.set_nr_of_jobs(self.jobShopEnv.nr_of_jobs + 1)
            # print(f"Job {job_id} generated with {num_operations} operations")  # Debugging print statement
            job_id += 1