from collections import OrderedDict
from typing import Dict, List, Optional


class Operation:
    def __init__(self, job, job_id, operation_id):
        self._job = job
        self._job_id = job_id
        self._operation_id = operation_id
        self._processing_times = OrderedDict()
        self._predecessors: List = []
        self._scheduling_information = {}

    def __repr__(self):
        return (
            f"<Operation(job_id={self._job_id}, operation_id={self._operation_id})>"
        )

    def reset(self):
        self._scheduling_information = {}

    @property
    def job(self):
        """Return the job object of the operation."""
        return self._job

    @property
    def job_id(self) -> int:
        """Return the job's id of the operation."""
        return self._job_id

    @property
    def operation_id(self) -> int:
        """Return the operation's id."""
        return self._operation_id

    @property
    def scheduling_information(self) -> Dict:
        """Return the scheduling information of the operation."""
        return self._scheduling_information

    @property
    def processing_times(self) -> dict:
        """Return a dictionary of machine ids and processing time durations."""
        return self._processing_times

    @property
    def scheduled_start_time(self) -> Optional[int]:
        """Return the scheduled start time of the operation."""
        return self._scheduling_information.get('start_time', None)

    @property
    def scheduled_end_time(self) -> Optional[int]:
        """Return the scheduled end time of the operation."""
        return self._scheduling_information.get('end_time', None)

    @property
    def scheduled_duration(self) -> Optional[int]:
        """Return the scheduled duration of the operation."""
        return self._scheduling_information.get('processing_time', None)

    @property
    def scheduled_machine(self) -> Optional[int]:
        """Return the machine id that the operation is scheduled on."""
        return self._scheduling_information.get('machine_id', None)

    @property
    def predecessors(self) -> List:
        """Return the list of predecessor operations."""
        return self._predecessors

    @property
    def optional_machines_id(self) -> List:
        """Returns the list of machine ids that are eligible for processing this operation."""
        return list(self._processing_times.keys())

    @property
    def finishing_time_predecessors(self) -> int:
        """Return the finishing time of the latest predecessor."""
        if not self.predecessors:
            return 0
        end_times_predecessors = [operation.scheduled_end_time for operation in self.predecessors]
        return max(end_times_predecessors)

    def update_job_id(self, new_job_id: int) -> None:
        """Update the id of a job (used for assembly scheduling problems, with no pre-given job id)."""
        self._job_id = new_job_id

    def update_job(self, job) -> None:
        """Update job information (edge case for FAJSP)."""
        self._job = job

    def add_predecessors(self, predecessors: List) -> None:
        """Add a list of predecessor operations to the current operation."""
        self.predecessors.extend(predecessors)

    def add_operation_option(self, machine_id, duration) -> None:
        """Add an machine option to the current operation."""
        self._processing_times[machine_id] = duration

    def update_scheduled_sequence_dependent_setup_times(self, start_time_setup, setup_duration):
        """Update the sequence dependent setup times of this operation (used for backfilling logic)."""
        self._scheduling_information['start_setup'] = start_time_setup
        self._scheduling_information['end_setup'] = start_time_setup + setup_duration
        self._scheduling_information['setup_time'] = setup_duration

    def add_operation_scheduling_information(self, machine_id: int, start_time: int, setup_time: int, duration) -> None:
        """Add scheduling information to the current operation."""
        self._scheduling_information = {
            'machine_id': machine_id,
            'start_time': start_time,
            'end_time': start_time + duration,
            'processing_time': duration,
            'start_setup': start_time - setup_time,
            'end_setup': start_time,
            'setup_time': setup_time}
