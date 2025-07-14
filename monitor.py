# ===================================================================
# Docker Container Performance Monitoring System
# ===================================================================
# This module provides functionality to build Docker images, run containers,
# and monitor their resource usage (CPU, memory) during execution.
# It's designed for benchmarking and comparing different implementations
# of algorithms and applications running in containerized environments.
# ===================================================================

import docker
import time
import csv
import os
import pandas as pd
from tabulate import tabulate
from dateutil.parser import parse
import threading
import psutil


# ===================================================================
# Builds a Docker image from the current directory with the specified tag.
# This function handles the build process and streams the build logs to
# the console for real-time feedback.
# ===================================================================
def build_image(client, tag):
    print("Building Docker image...")
    # WHY COMMENT: We use the current directory (".") as the build context
    # because all Dockerfiles are expected to be in the same directory as this script
    image, logs = client.images.build(path=".", tag=tag)
    # GUIDE COMMENT: Iterate through build logs to show progress in real-time
    for chunk in logs:
        if 'stream' in chunk:
            print(chunk['stream'].strip())
    print("Image built successfully.")


# ===================================================================
# Runs a Docker container with the specified image tag and input value.
# Maps image tags to their corresponding Python scripts and configures
# the container to execute the appropriate script with the given input.
# ===================================================================
def run_container(client, image_tag, input_value):    
    # We use a dictionary to map image tags to script names
    # This allows for easy addition of new applications without modifying the core logic
    image_to_script = {
        "fibonacci-app": "fibonacci.py",
        "hello-app": "hello.py",
        "isprime-app": "isprime.py",
        "yolo-runner-app-ultralytics": "yolo_runner.py",
        "yolo-runner-app-pytorch": "yolo_runner.py"
    }
    
    # Get the script based on the image tag
    script = image_to_script.get(image_tag, "")

    # Container execution differs based on whether input is provided
    # For scripts that require input parameters vs those that don't
    if input_value is not None and input_value != "":
        print(f"Running container with input: {input_value}")
        # We use detach=True to run the container in the background so we can monitor its resources while it's running
        container = client.containers.run(
            image=image_tag,
            command=["python3", script, str(input_value)],
            detach=True
        )
    else:
        print(f"Running container without input")
        container = client.containers.run(
            image=image_tag,
            command=["python3", script],
            detach=True
        )

    return container

# ===================================================================
# Runs a container with the given parameters and monitors its resource usage.
# Creates a separate monitoring thread that collects performance metrics
# while the container is running, then calculates execution time and
# writes all metrics to a CSV file.
# ===================================================================
def run_container_with_input(client, image_tag, input_value, function_id):
    container = run_container(client, image_tag, input_value)
    
    # Initialize a dictionary to store container statistics
    # This shared data structure will be updated by the monitoring thread and read by the main thread after container execution completes
    stats_data = {
        "cpu_usage": 0.0,
        "memory_usage": 0.0,
        "memory_percentage": 0.0,
        "max_cpu_freq_ghz": 0.0,
        "memory_limit": 0.0,
        "host_total_memory": 0.0,

        # We capture both logical and physical cores to understand the relationship between CPU architecture and container performance
        "num_logical_cores": psutil.cpu_count(logical=True),
        "num_physical_cores": psutil.cpu_count(logical=False)
    }

    # Create and start a daemon thread for monitoring
    # Using a thread allows us to collect metrics while the container runs
    # Setting daemon=True ensures the thread will terminate if the main program exits
    monitor_thread = threading.Thread(
        name="monitor_container",
        target=monitor_container_resources,
        args=(container, stats_data)
    )
    monitor_thread.daemon = True
    monitor_thread.start()

    # Wait for container to finish execution 
    # container.wait() blocks until the container exits
    exit_code = container.wait()
    monitor_thread.join()

    # Calculate execution time using container timestamps
    # Docker provides precise container start and finish times
    details = client.api.inspect_container(container.id)
    start_time_str = details["State"]["StartedAt"]
    end_time_str = details["State"]["FinishedAt"]

    # Parse ISO format timestamps and calculate duration
    start_time = parse(start_time_str)
    end_time = parse(end_time_str)
    execution_time = (end_time - start_time).total_seconds()

    # Organize all metrics into a structured dictionary for consistent CSV output format and easier data analysis
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

    # Save metrics to CSV file
    write_metrics_to_csv(metrics_data)

# ===================================================================
# Monitors a running container's resource usage in real-time.
# This function runs in a separate thread and continuously collects
# CPU and memory metrics until the container stops running.
# ===================================================================
def monitor_container_resources(container, stats_data):
    cpu_usage = 0.0
    memory_usage = 0.0
    memory_percentage = 0.0
    memory_available = None

    # Docker's stats API provides a stream of container metrics
    # We continuously process this stream until the container stops
    for stat in container.stats(stream=True, decode=True):
        try:
            # Extract and calculate CPU usage percentage
            # Docker provides delta values that need to be processed to get usage percentage
            cpu_stats = stat.get("cpu_stats", {})
            precpu_stats = stat.get("precpu_stats", {})
            
            # We need to check for all these keys because the first stats event might not contain complete information, and we want to avoid KeyErrors
            if "cpu_usage" in cpu_stats and "cpu_usage" in precpu_stats and "system_cpu_usage" in cpu_stats and "system_cpu_usage" in precpu_stats:
                cpu_delta = cpu_stats["cpu_usage"]["total_usage"] - precpu_stats["cpu_usage"]["total_usage"]
                system_cpu_delta = cpu_stats["system_cpu_usage"] - precpu_stats["system_cpu_usage"]

                # Division by zero check is necessary as system_cpu_delta could be zero in the first iteration or if sampling interval is too small
                cpu_usage = (cpu_delta / system_cpu_delta) * 100.0 if system_cpu_delta > 0 else 0.0

            # Get current CPU frequency using psutil
            # This provides additional context about the host system during execution
            freq = psutil.cpu_freq()
            cpu_freq_ghz = round(freq.current / 1000, 2) if freq and freq.current else 0.0
            stats_data["cpu_freq_ghz"] = cpu_freq_ghz
            
            # Extract and calculate memory usage metrics
            memory_stats = stat.get("memory_stats", {})
            if "usage" in memory_stats and "limit" in memory_stats:
                memory_usage = memory_stats["usage"]
                memory_limit = memory_stats["limit"]
                memory_percentage = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0.0
                memory_available = memory_limit - memory_usage

            # Get host system memory for context
            # This helps understand container memory usage in relation to total system memory
            host_total_memory = round(psutil.virtual_memory().total / (1024 ** 2), 2)

            # Update the shared stats_data dictionary with latest values
            # This makes the most recent metrics available to the main thread
            stats_data["cpu_usage"] = cpu_usage
            stats_data["memory_usage"] = memory_usage
            stats_data["memory_percentage"] = memory_percentage
            stats_data["memory_available"] = memory_available if memory_available is not None else 0
            stats_data["host_total_memory"] = host_total_memory if host_total_memory is not None else 0

            # Check if container is still running
            # If not, exit the monitoring loop
            try:
                container.reload()
                if container.status != "running":
                    break
            except docker.errors.NotFound:
                 # Container might be removed before we can check its status
                 # This happens if the container exits and is automatically removed
                 break

        except KeyError as e:
            # First stats event often doesn't contain all metrics
            # This is expected behavior, so we handle it gracefully
            print(f"Missing key in stats (expected on first read): {e}")
        except Exception as e:
            # Generic exception handler to ensure the monitoring thread doesn't crash if unexpected errors occur
            print(f"Error monitoring container: {e}")
            break

# ===================================================================
# Writes container performance metrics to a CSV file.
# Creates a new file with headers if it doesn't exist,
# or appends to an existing file, adding a timestamp to each record.
# ===================================================================
def write_metrics_to_csv(data, filename="metrics.csv"):
    # Check if file already exists to determine if headers are needed
    file_exists = os.path.isfile(filename)
    
    # Add timestamp to allow chronological analysis of multiple runs
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    data['Timestamp'] = timestamp

    # Open file in append mode to preserve existing data
    with open(filename, mode='a', newline='') as csv_file:
        # Define column order explicitly to ensure consistent CSV format even if the input dictionary keys are in a different order
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
        # Only write header row when creating a new file to avoid duplicate headers in the middle of the data
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

# ===================================================================
# Converts a CSV file of metrics into a formatted table and saves it as text.
# Provides a human-readable view of the collected performance data.
# ===================================================================
def format_csv_as_table(input_file="metrics.csv", output_file="metrics_table.txt"):
    """
    Converts a CSV file into a table and saves it in a .txt file
    """

    # Verify input file exists before attempting to process
    if not os.path.exists(input_file):
        print(f"File {input_file} does not exist.") 
        return False
    
    try:
        df = pd.read_csv(input_file)
        
        # Format the data as a grid table using tabulate
        # This creates a visually appealing representation with aligned columns
        table = tabulate(df, headers='keys', tablefmt='grid', showindex=False)
        
        with open(output_file, 'w') as f:
            f.write(table)
        
        print(f"Formatted table saved in {output_file}")
        
        return True
    except Exception as e:
        # Comprehensive error handling to provide feedback if anything goes wrong during the formatting process
        print(f"Error during formatting: {e}")
        return False

# ===================================================================
# Main entry point that defines test functions and orchestrates
# the container execution and monitoring process.
# Runs each function multiple times to collect statistically significant data.
# ===================================================================
def main():
    # Define test functions with their parameters in a structured way
    # This makes it easy to add, remove, or modify test cases
    functions_to_test = [
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

    # Number of repetitions for statistical significance
    # Running each test multiple times helps account for system variability
    N = 10

    # Set a longer timeout (5 minutes) because some containers like YOLOv8 might take longer to complete their execution
    client = docker.from_env(timeout=300)

    # Iterate through each function and run it N times
    for func in functions_to_test:
        print(f"\n=== Testing function: {func['name']} ===")
            
        for i in range(N):
            print(f"\n--- Run {i+1} of {N} for function {func['name']} ---")
            run_container_with_input(
                client,
                func["image"],
                func["input"] if func["input"] is not None else "",
                func["name"]
            )

    # The following code is commented out but could be useful for generating formatted tables. Consider enabling if needed.
    #print("Metrics saved to CSV and formatted table generated.")
    #format_csv_as_table()

if __name__ == "__main__":
    main()