# ===================================================================
# Docker Parallel Performance Monitor
# ===================================================================
# This script runs multiple Docker containers in parallel, each
# executing a different computational workload. It monitors their
# CPU and memory usage in real-time and logs performance metrics
# to a CSV file for later analysis.
# ===================================================================

import docker
import threading
import time
import os
import csv
from dateutil.parser import parse
import psutil

# ===================================================================
# Monitors a container's CPU and memory usage in real-time.
# Runs inside a separate thread until the container stops.
# Updates a shared stats_data dictionary with the latest metrics.
# ===================================================================
def monitor_container_resources(container, stats_data):
    cpu_usage = 0.0
    memory_usage = 0.0
    memory_percentage = 0.0
    memory_available = None

    for stat in container.stats(stream=True, decode=True):
        try:
            # Extract CPU usage statistics from Docker API
            cpu_stats = stat.get("cpu_stats", {})
            precpu_stats = stat.get("precpu_stats", {})
            if "cpu_usage" in cpu_stats and "cpu_usage" in precpu_stats and "system_cpu_usage" in cpu_stats and "system_cpu_usage" in precpu_stats:
                cpu_delta = cpu_stats["cpu_usage"]["total_usage"] - precpu_stats["cpu_usage"]["total_usage"]
                system_cpu_delta = cpu_stats["system_cpu_usage"] - precpu_stats["system_cpu_usage"]
                cpu_usage = (cpu_delta / system_cpu_delta) * 100.0 if system_cpu_delta > 0 else 0.0

            # Capture CPU frequency from host system
            freq = psutil.cpu_freq()
            cpu_freq_ghz = round(freq.current / 1000, 2) if freq and freq.current else 0.0
            stats_data["cpu_freq_ghz"] = cpu_freq_ghz

            # Extract memory usage statistics from Docker API
            memory_stats = stat.get("memory_stats", {})
            if "usage" in memory_stats and "limit" in memory_stats:
                memory_usage = memory_stats["usage"]
                memory_limit = memory_stats["limit"]
                memory_percentage = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0.0
                memory_available = memory_limit - memory_usage

            # Get total host memory for context
            host_total_memory = round(psutil.virtual_memory().total / (1024 ** 2), 2)

            # Update shared stats dictionary
            stats_data["cpu_usage"] = cpu_usage
            stats_data["memory_usage"] = memory_usage
            stats_data["memory_percentage"] = memory_percentage
            stats_data["memory_available"] = memory_available if memory_available is not None else 0
            stats_data["host_total_memory"] = host_total_memory if host_total_memory is not None else 0

            # Check if container is still running
            try:
                container.reload()
                if container.status != "running":
                    break
            except docker.errors.NotFound:
                break

        except Exception as e:
            print(f"Monitoring error: {e}")
            break


# ===================================================================
# Writes performance metrics to a CSV file.
# Creates a header row if the file does not exist yet.
# Each record includes a timestamp for chronological analysis.
# ===================================================================
def write_metrics_to_csv(data, filename="parallel_metrics.csv"):
    file_exists = os.path.isfile(filename)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    data['Timestamp'] = timestamp

    with open(filename, mode='a', newline='') as csv_file:
        fieldnames = [
            'Timestamp',
            'Function',
            'Input',
            'Execution Time (s)',
            'CPU Usage (%)',
            'CPU Frequency (GHz)',
            'Number of Logical Cores',
            'Number of Physical Cores',
            'Memory Usage (MB)',
            'Memory Usage (%)',
            'Memory Available (MB)',
            'Host Total Memory (MB)'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


# ===================================================================
# Runs a single container with specified parameters and monitors it.
# Launches a thread to monitor its resource usage while executing.
# After the container finishes, writes metrics to the CSV file.
# ===================================================================
def run_and_monitor(client, image_tag, input_value, function_id):
    print(f"Starting container {function_id}...")

    # Map image tags to their corresponding script filenames
    image_to_script = {
        "fibonacci-app": "fibonacci.py",
        "hello-app": "hello.py",
        "isprime-app": "isprime.py",
        "yolo-runner-app-ultralytics": "yolo_runner.py",
        "yolo-runner-app-pytorch": "yolo_runner.py"
    }

    # Determine the script to execute
    script = image_to_script.get(image_tag, "")

    # Run the Docker container (detached) with or without input
    if input_value is not None and input_value != "":
        container = client.containers.run(
            image=image_tag,
            command=["python3", script, str(input_value)],
            detach=True
        )
    else:
        container = client.containers.run(
            image=image_tag,
            command=["python3", script],
            detach=True
        )

    # Shared data structure for monitoring thread
    stats_data = {
        "cpu_usage": 0.0,
        "memory_usage": 0.0,
        "memory_percentage": 0.0,
        "max_cpu_freq_ghz": 0.0,
        "memory_limit": 0.0,
        "host_total_memory": 0.0,
        "num_logical_cores": psutil.cpu_count(logical=True),
        "num_physical_cores": psutil.cpu_count(logical=False)
    }

    # Start a daemon thread to monitor container resource usage
    monitor_thread = threading.Thread(
        name=f"monitor_{function_id}",
        target=monitor_container_resources,
        args=(container, stats_data)
    )
    monitor_thread.daemon = True
    monitor_thread.start()

    # Wait for container to finish and stop the monitoring thread
    exit_code = container.wait()
    monitor_thread.join()

    # Calculate execution time from container start/finish timestamps
    details = client.api.inspect_container(container.id)
    start_time_str = details["State"]["StartedAt"]
    end_time_str = details["State"]["FinishedAt"]

    start_time = parse(start_time_str)
    end_time = parse(end_time_str)
    execution_time = (end_time - start_time).total_seconds()

    # Remove container to free up resources
    container.remove()

    # Organize metrics data into a consistent dictionary
    metrics_data = {
        'Function': function_id,
        'Input': input_value,
        'Execution Time (s)': round(execution_time, 2),
        'Number of Logical Cores': stats_data["num_logical_cores"],
        'Number of Physical Cores': stats_data["num_physical_cores"],
        'CPU Usage (%)': round(stats_data["cpu_usage"], 2),
        'CPU Frequency (GHz)': stats_data.get("cpu_freq_ghz", 0.0),
        'Memory Usage (MB)': round(stats_data["memory_usage"] / (1024 ** 2), 2),
        'Memory Usage (%)': round(stats_data["memory_percentage"], 2),
        'Memory Available (MB)': round((stats_data.get("memory_available") or 0) / (1024 ** 2), 2),
        'Host Total Memory (MB)': stats_data.get("host_total_memory", 0.0)
    }

    # Save metrics to CSV
    write_metrics_to_csv(metrics_data)


# ===================================================================
# Main execution loop
# Runs multiple containers simultaneously in N iterations to simulate
# parallel workloads and collect performance data under load.
# ===================================================================
def main():
    client = docker.from_env(timeout=300)  # Set timeout to 5 minutes

    # Define a list of functions and corresponding Docker images to run
    functions = [
        {"name": "Fibonacci", "image": "fibonacci-app", "input": 20000},
        {"name": "Hello", "image": "hello-app", "input": None},
        {"name": "IsPrime", "image": "isprime-app", "input": 29937646239629496719},
        {"name": "YOLOv8", "image": "yolo-runner-app-ultralytics", "input": "input.jpg"},
        {"name": "YOLOv8", "image": "yolo-runner-app-ultralytics", "input": "input1.jpg"},
        {"name": "YOLOv8", "image": "yolo-runner-app-ultralytics", "input": "input2.jpg"},
        {"name": "YOLOv8", "image": "yolo-runner-app-pytorch", "input": "input.jpg"},
        {"name": "YOLOv8", "image": "yolo-runner-app-pytorch", "input": "input1.jpg"},
        {"name": "YOLOv8", "image": "yolo-runner-app-pytorch", "input": "input2.jpg"}
    ]

    N = 10  # Number of repetitions for statistical robustness

    # Run all functions in parallel N times
    for i in range(N):
        print(f"--- Iteration {i+1} ---")
        threads = []
        for func in functions:
            t = threading.Thread(
                target=run_and_monitor,
                args=(client, func["image"], func["input"] if func["input"] is not None else "", func["name"])
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        print()

    print("Simultaneous execution completed. Metrics saved to parallel_metrics.csv.")


if __name__ == "__main__":
    main()