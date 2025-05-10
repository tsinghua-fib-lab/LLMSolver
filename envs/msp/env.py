import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def validate_schedule(schedule):
    """Validate the schedule for conflicts and constraints"""
    machine_schedules = {}  # {machine: [(start, end)]}
    job_end_times = {}  # {job_id: last_end_time}

    for job_data in schedule['Schedule']:
        job_id = job_data['job']
        prev_end = 0

        for task in sorted(job_data['tasks'], key=lambda x: x['task']):
            # Check task ordering
            if task['start'] < prev_end:
                return False, f"Job {job_id} has overlapping tasks"

            # Check time values
            if task['start'] < 0 or task['duration'] <= 0:
                return False, f"Invalid times for job {job_id} task {task['task']}"

            # Track machine usage
            machine = task['machine']
            task_end = task['start'] + task['duration']

            if machine not in machine_schedules:
                machine_schedules[machine] = []

            # Check machine conflicts
            for existing_start, existing_end in machine_schedules[machine]:
                if not (task_end <= existing_start or task['start'] >= existing_end):
                    return False, f"Machine {machine} conflict between job {job_id} and existing task"

            machine_schedules[machine].append((task['start'], task_end))
            prev_end = task_end

    return True, "Schedule is valid"


def check_jssp_schedule_valid(instance, schedule):
    machine_usage = {}
    processing_times = instance['processing_times']
    for job_data in schedule['schedule']:
        job_id = f'job_{job_data["job"]}'
        tasks = job_data['tasks']
        tasks = sorted(tasks, key=lambda t: t['task'])

        ope_key_list = sorted(processing_times[job_id].keys(), key=lambda x: int(x.split('_')[1]))
        if len(tasks) != len(ope_key_list):
            return False
        for i, task in enumerate(tasks):
            op_id = f'op_{task["task"]}'
            if op_id != ope_key_list[i]:
                return False

            machine = f'machine_{task["machine"]}'
            start = task['start']
            duration = task['duration']

            # Check duration matches the instance
            expected_duration = processing_times[job_id][op_id][machine][0]
            if duration != expected_duration:
                return False

            # Check task order within job
            if i > 0:
                prev = tasks[i - 1]
                if start < prev['start'] + prev['duration']:
                    return False

            # Check machine availability (no overlaps)
            for t in range(start, start + duration):
                if (machine, t) in machine_usage:
                    return False
                machine_usage[(machine, t)] = (job_id, op_id)

    return True


def check_fjssp_schedule_valid(instance, schedule):
    machine_usage = {}
    processing_times = instance['processing_times']

    for job_data in schedule['schedule']:
        job_id = f'job_{job_data["job"]}'
        tasks = sorted(job_data['tasks'], key=lambda x: x['task'])

        ope_key_list = sorted(processing_times[job_id].keys(), key=lambda x: int(x.split('_')[1]))
        if len(tasks) != len(ope_key_list):
            return False
        for i, task in enumerate(tasks):
            op_id = f'op_{task["task"]}'
            if op_id != ope_key_list[i]:
                return False
            machine = f'machine_{task["machine"]}'
            start = task['start']
            duration = task['duration']

            # Check if machine is allowed for this operation
            if machine not in processing_times[job_id][op_id]:
                return False

            # Check if duration matches the machine-specific value
            expected_duration = processing_times[job_id][op_id][machine][0]
            if duration != expected_duration:
                return False

            # Check task order (precedence)
            if i > 0:
                prev = tasks[i - 1]
                if start < prev['start'] + prev['duration']:
                    return False

            # Check for machine conflicts
            for t in range(start, start + duration):
                if (machine, t) in machine_usage:
                    return False
                machine_usage[(machine, t)] = (job_id, op_id)

    return True


def check_fssp_schedule_valid(instance, schedule):
    machine_usage = {}
    processing_times = instance['processing_times']

    for job_data in schedule['schedule']:
        job_id = f'job_{job_data["job"]}'
        tasks = sorted(job_data['tasks'], key=lambda x: x['task'])

        ope_key_list = sorted(processing_times[job_id].keys(), key=lambda x: int(x.split('_')[1]))
        if len(tasks) != len(ope_key_list):
            return False

        for i, task in enumerate(tasks):
            op_id = f'op_{task["task"]}'
            if op_id != ope_key_list[i]:
                return False
            machine = f'machine_{task["machine"]}'
            start = task['start']
            duration = task['duration']

            # Check machine validity
            if machine not in processing_times[job_id][op_id]:
                return False

            # Check duration match
            expected_duration = processing_times[job_id][op_id][machine][0]
            if duration != expected_duration:
                return False

            # Check task order
            if i > 0:
                prev = tasks[i - 1]
                if start < prev['start'] + prev['duration']:
                    return False

            # Check machine conflicts
            for t in range(start, start + duration):
                if (machine, t) in machine_usage:
                    return False
                machine_usage[(machine, t)] = (job_id, op_id)

    return True


def check_hfssp_schedule_valid(instance, schedule):
    def get_stage_from_machine(machine_id, machines_per_stage):
        cumulative = 0
        for stage_index, count in machines_per_stage.items():
            if machine_id < cumulative + count:
                return stage_index
            cumulative += count
        return None

    machine_usage = {}
    proc_times = instance['processing_times']

    for job_data in schedule['schedule']:
        job_id = f"job_{job_data['job']}"
        tasks = job_data['tasks']

        # Check stage order
        sorted_tasks = sorted(tasks, key=lambda x: x['task'])

        for i, task in enumerate(sorted_tasks):
            stage = f"stage_{task['task']}"
            machine = f"machine_{task['machine']}"
            start = task['start']
            duration = task['duration']

            # Check machine is valid for this stage
            if get_stage_from_machine(task['machine'], instance['machines_per_stage']) != stage:
                return False

            # Check machine and duration in instance
            if machine not in proc_times[job_id][stage]:
                return False

            expected_duration = proc_times[job_id][stage][machine][0]
            if duration != expected_duration:
                return False

            # Check machine conflict
            for t in range(start, start + duration):
                if (machine, t) in machine_usage:
                    return False
                machine_usage[(machine, t)] = job_id

            # Check stage order
            if i > 0:
                prev = sorted_tasks[i - 1]
                if start < prev['start'] + prev['duration']:
                    return False

    return True


def check_ossp_schedule_valid(instance, schedule):
    machine_usage = {}
    processing_times = instance['processing_times']

    for job_data in schedule['schedule']:
        job_id = f'job_{job_data["job"]}'
        tasks = job_data['tasks']
        seen_ops = set()

        for task in tasks:
            op_id = f'op_{task["task"]}'
            if op_id in seen_ops:
                return False  # Duplicate operation
            seen_ops.add(op_id)

            machine = f'machine_{task["machine"]}'
            start = task['start']
            duration = task['duration']

            # Check if operation exists in instance and machine is valid
            if op_id not in processing_times[job_id]:
                return False
            if machine not in processing_times[job_id][op_id]:
                return False

            # Check duration match
            expected_duration = processing_times[job_id][op_id][machine][0]
            if duration != expected_duration:
                return False

            # Check machine conflicts
            for t in range(start, start + duration):
                if (machine, t) in machine_usage:
                    return False
                machine_usage[(machine, t)] = (job_id, op_id)

        # Optional: Check all required ops are scheduled
        if seen_ops != set(processing_times[job_id].keys()):
            return False

    return True


def check_asp_schedule_valid(instance, schedule):
    machine_usage = {}
    op_start_times = {}

    processing_times = instance['processing_times']
    precedence_relations = instance['precedence_relations']

    for job_data in schedule['schedule']:
        job_id = f"job_{job_data['job']}"
        for task in job_data['tasks']:
            op_id = f"op_{task['task']}"
            machine = f"machine_{task['machine']}"
            start = task['start']
            duration = task['duration']

            # Check if operation is defined
            if op_id not in processing_times[job_id]:
                return False
            if machine not in processing_times[job_id][op_id]:
                return False

            expected_duration = processing_times[job_id][op_id][machine][0]
            if duration != expected_duration:
                return False

            # Save start time for precedence check
            op_start_times[(job_id, op_id)] = (start, duration)

            # Machine conflict check
            for t in range(start, start + duration):
                if (machine, t) in machine_usage:
                    return False
                machine_usage[(machine, t)] = f"{job_id}-{op_id}"

    # Check precedence constraints
    for (pred, succ) in precedence_relations:
        pred_end = op_start_times[pred][0] + op_start_times[pred][1]
        succ_start = op_start_times[succ][0]
        if pred_end > succ_start:
            return False

    return True


class SchedulingProblemEnv:
    def __init__(self, problem_type: str, obj: str = 'makespan'):
        assert problem_type in ['jssp', 'fjssp', 'fssp', 'hfssp', 'ossp', 'asp']
        self.problem_type = problem_type
        self.obj = obj

    def calculate_makespan(self, schedule):
        """Calculate the makespan (total completion time) of the schedule"""
        max_end_time = 0

        for job_data in schedule['schedule']:
            for task in job_data['tasks']:
                task_end_time = task['start'] + task['duration']
                if task_end_time > max_end_time:
                    max_end_time = task_end_time

        return max_end_time

    def get_reward(self, instances: dict, solution: dict, ):
        if self.obj == 'makespan':
            return self.calculate_makespan(solution)
        else:
            raise NotImplementedError

    def check_valid(self, instances: dict, solution: dict, ):
        if self.problem_type == 'jssp':
            is_valid = check_jssp_schedule_valid(instances, solution)
        elif self.problem_type == 'fjssp':
            is_valid = check_fjssp_schedule_valid(instances, solution)
        elif self.problem_type == 'fssp':
            is_valid = check_fssp_schedule_valid(instances, solution)
        elif self.problem_type == 'hfssp':
            is_valid = check_hfssp_schedule_valid(instances, solution)
        elif self.problem_type == 'ossp':
            is_valid = check_ossp_schedule_valid(instances, solution)
        elif self.problem_type == 'asp':
            is_valid = check_asp_schedule_valid(instances, solution)
        else:
            raise NotImplementedError
        if not isinstance(is_valid, bool):
            is_valid = False
        return is_valid

    def plot_schedule(self, schedule):
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = [
            'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
            'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
            'lightblue', 'gold', 'lightgreen', 'salmon', 'plum',
            'sienna', 'violet', 'lightgrey', 'olive', 'turquoise'
        ]
        color_idx = 0
        job_colors = {}

        # Find all machine ids
        all_machines = set()
        for job in schedule['schedule']:
            for task in job['tasks']:
                all_machines.add(task['machine'])

        all_machines = sorted(all_machines)
        machine_to_y = {machine: idx for idx, machine in enumerate(all_machines)}

        # Plot each task
        for job in schedule['schedule']:
            job_id = job['job']
            if job_id not in job_colors:
                job_colors[job_id] = colors[color_idx % len(colors)]
                color_idx += 1
            for task in job['tasks']:
                start = task['start']
                duration = task['duration']
                machine = task['machine']
                y = machine_to_y[machine]
                ax.barh(y, duration, left=start, color=job_colors[job_id], edgecolor='black')
                # Label placed inside the bar
                ax.text(start + duration / 2, y, f'J{job_id}({task["task"]})',
                        ha='center', va='center', color='black', fontsize=8, fontweight='bold')

        # Customize axes
        ax.set_yticks(range(len(all_machines)))
        ax.set_yticklabels(all_machines)
        ax.invert_yaxis()  # 0 at top, max at bottom
        ax.set_xlabel('Time')
        ax.set_ylabel('Machine')
        ax.set_title('Job Shop Schedule')
        ax.grid(True, axis='x')

        # Horizontal lines for each machine
        for y in range(len(all_machines)):
            ax.axhline(y, color='grey', linestyle='--', linewidth=0.5)

        # Create a legend OUTSIDE
        patches = [mpatches.Patch(color=color, label=f'Job {jid}') for jid, color in job_colors.items()]
        ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1.0, 0.5))  # Move legend outside

        plt.tight_layout()
        plt.show()


def test_valid():
    # Test with your example
    schedule = {
        'Schedule': [
            {'job': 0, 'tasks': [
                {'duration': 5, 'machine': 1, 'start': 4, 'task': 0},
                {'duration': 1, 'machine': 6, 'start': 9, 'task': 1},
                {'duration': 3, 'machine': 8, 'start': 10, 'task': 2}
            ]},
            {'job': 1, 'tasks': [
                {'duration': 1, 'machine': 2, 'start': 3, 'task': 0},
                {'duration': 4, 'machine': 5, 'start': 6, 'task': 1},
                {'duration': 4, 'machine': 11, 'start': 12, 'task': 2}
            ]}
        ]
    }

    is_valid, message = validate_schedule(schedule)
    print(f"Validation: {is_valid}, Message: {message}")


def test_plot():
    schedule = {
        'Schedule': [
            {'job': 0, 'tasks': [
                {'duration': 5, 'machine': 1, 'start': 4, 'task': 0},
                {'duration': 1, 'machine': 6, 'start': 9, 'task': 1},
                {'duration': 3, 'machine': 8, 'start': 10, 'task': 2},
            ]},
            {'job': 1, 'tasks': [
                {'duration': 1, 'machine': 2, 'start': 3, 'task': 0},
                {'duration': 4, 'machine': 5, 'start': 6, 'task': 1},
                {'duration': 4, 'machine': 11, 'start': 12, 'task': 2},
            ]}
        ]
    }

    env = SchedulingProblemEnv(problem_type='jssp')
    env.plot_schedule(schedule)


if __name__ == '__main__':
    test_plot()
