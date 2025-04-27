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


def plot_schedule(schedule):
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
    for job in schedule['Schedule']:
        for task in job['tasks']:
            all_machines.add(task['machine'])

    all_machines = sorted(all_machines)
    machine_to_y = {machine: idx for idx, machine in enumerate(all_machines)}

    # Plot each task
    for job in schedule['Schedule']:
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


def calculate_makespan(schedule):
    """Calculate the makespan (total completion time) of the schedule"""
    max_end_time = 0

    for job_data in schedule['Schedule']:
        for task in job_data['tasks']:
            task_end_time = task['start'] + task['duration']
            if task_end_time > max_end_time:
                max_end_time = task_end_time

    return max_end_time


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

    plot_schedule(schedule)


if __name__ == '__main__':
    test_plot()
