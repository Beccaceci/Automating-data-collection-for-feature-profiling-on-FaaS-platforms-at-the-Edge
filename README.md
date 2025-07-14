# Parallel and Single Docker Container Performance Monitoring

This project contains scripts and Docker containers to benchmark and monitor the performance of various computational workloads running inside Docker containers. It includes:

- `monitor.py`: Runs containers sequentially, monitors resource usage, and logs metrics.
- `parallel_monitor.py`: Runs containers in parallel with real-time resource monitoring.
- `plot_generator.py`: Generates high-resolution plots from collected CSV metrics.
- Dockerfiles and Python scripts for workloads like Fibonacci, Hello, IsPrime, and YOLOv8.

## Features

- Real-time CPU and memory usage monitoring of Docker containers.
- Support for sequential and parallel execution modes.
- Metrics saved in CSV format with timestamps.
- Automated plot generation for performance comparison.

## Requirements

- Docker installed and running.
- Python 3 with dependencies: `docker`, `psutil`, `pandas`, `matplotlib`, `scipy`, `tabulate`, `python-dateutil`.

## Usage

1. Build Docker images using provided Dockerfiles.
2. Run `monitor.py` for sequential tests or `parallel_monitor.py` for parallel tests.
3. Use `plot_generator.py` to visualize results.

## License

This project is licensed under the MIT License - see the LICENSE file for details.