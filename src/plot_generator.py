# ===================================================================
# Edge Node Performance Metrics Plotting System
# ===================================================================
# This script reads CSV files containing execution metrics collected from multiple edge nodes.
# It computes mean values and 95% confidence intervals, and generates high-resolution bar plots comparing performance across different machines and functions.
#
# The script is structured to automate the analysis and visualization of distributed performance data, supporting both single and parallel execution scenarios.
# 
# Using pandas and matplotlib enables efficient data manipulation and high-quality plotting, while the modular configuration allows easy extension to new nodes or metrics.
# ===================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ===================================================================
# CONFIGURATION SECTION
#
# Machine file paths and plotting parameters are centralized for maintainability and scalability.
# 
# This structure allows for easy updates when adding new machines or changing plot styles.
# ===================================================================
machine_files = {
    "172.16.6.119": "172_16_6_119/172_16_6_119.csv",
    "172.16.6.119 (parallel)": "172_16_6_119/parallel_172_16_6_119.csv",

    "172.16.5.133": "172_16_5_133/172_16_5_133.csv",
    "172.16.5.133 (parallel)": "172_16_5_133/parallel_172_16_5_133.csv",

    "172.16.5.19": "172_16_5_19/172_16_5_19.csv",
    "172.16.5.19 (parallel)": "172_16_5_19/parallel_172_16_5_19.csv",

    "192.168.1.19": "192_168_1_19/192_168_1_19.csv",
    "192.168.1.19 (parallel)": "192_168_1_19/parallel_192_168_1_19.csv",

    "192.168.1.10": "192_168_1_10/192_168_1_10.csv",
    "192.168.1.10 (parallel)": "192_168_1_10/parallel_192_168_1_10.csv"
}

# ===================================================================
# Lists of functions and metrics to include in plots
# Enables multi-function and multi-metric comparative analysis.
# ===================================================================
functions = ["Fibonacci", "IsPrime", "Hello", "YOLOv8"]
metrics = {
    "Execution Time (s)": "Execution Time (s)",
    "CPU Usage (%)": "CPU Usage (%)",
    "Memory Usage (%)": "Memory Usage (%)"
}

base_dir = os.path.dirname(os.path.abspath(__file__))
DPI = 300
FIGURE_WIDTH = 12
FIGURE_HEIGHT = 6

# ===================================================================
# DATA LOADING SECTION
#
# Reads CSV files into dictionaries separating single and parallel runs.
# 
# Data is organized by machine and execution type for straightforward comparative plotting.
# 
# This approach supports flexible analysis and avoids hardcoding data paths in plotting logic.
# ===================================================================
single_run_data = {}
parallel_run_data = {}

for machine, rel_file in machine_files.items():
    abs_file = os.path.join(base_dir, rel_file)
    if os.path.exists(abs_file):
        df = pd.read_csv(abs_file)
        if "(parallel)" in machine:
            machine_name = machine.replace(" (parallel)", "")
            parallel_run_data[machine_name] = df
        else:
            single_run_data[machine] = df

# ===================================================================
# STATISTICS SECTION
#
# Computes mean and 95% confidence interval for a numeric dataset.
# 
# This is used to generate error bars in the plots, providing statistical context for performance comparisons.
# ===================================================================
def compute_95_confidence_interval(data):
    data = np.array(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    degrees_of_freedom = len(data) - 1
    t_critical = stats.t.ppf(q=0.975, df=degrees_of_freedom)
    margin_of_error = t_critical * sem
    return mean, margin_of_error

# ===================================================================
# LABELS AND PLOTTING SECTION
#
# Human-friendly labels and consistent machine order improve plot readability and interpretability.
# 
# Including hardware specs in labels helps contextualize performance differences.
# ===================================================================
machine_labels = {
    "172.16.6.119": "Edge node 1\n2 vCPU, 2GB RAM",
    "172.16.5.133": "Edge node 2\n4 vCPU, 4GB RAM",
    "172.16.5.19": "Edge node3\n16 vCPU, 31.2GB RAM",
    "192.168.1.10": "Edge node 4\n4 vCPU, 4.15GB RAM",
    "192.168.1.19": "Edge node 5\n8 vCPU, 15.6GB RAM"
}

# ===================================================================
# Generates bar plots comparing performance across edge nodes.
# Handles data grouping, statistics, and high-resolution PNG output.
# 
# Modular plotting function allows reuse for both single and parallel execution data.
# 
# Error bars and consistent formatting ensure scientific rigor and publication-quality visuals.
# ===================================================================
def plot_group(data_dict, group_name):
    # Consistent machine order for X-axis across all plots
    edge_order = [
        "172.16.6.119",
        "172.16.5.133",
        "172.16.5.19",
        "192.168.1.10",
        "192.168.1.19"
    ]

    machine_names = [ip for ip in edge_order if ip in data_dict]
    machine_labels_list = [machine_labels.get(m, m) for m in machine_names]

    for metric_en, metric_label in metrics.items():
        x = range(len(machine_names))
        width = 0.18
        plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

        # Configure font sizes for high-resolution readability
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12
        })

        # Build bar chart for each function with error bars
        for i, function in enumerate(functions):
            values = []
            ci_values = []
            for machine in machine_names:
                df = data_dict[machine]
                df_function = df[df["Function"] == function]
                if not df_function.empty:
                    numeric_data = pd.to_numeric(df_function[metric_en], errors='coerce').dropna()
                    if len(numeric_data) > 0:
                        mean, margin_of_error = compute_95_confidence_interval(numeric_data)
                    else:
                        mean, margin_of_error = 0, 0
                else:
                    mean, margin_of_error = 0, 0
                values.append(mean)
                ci_values.append(margin_of_error)

            plt.bar(
                [pos + i * width for pos in x],
                values,
                width=width,
                yerr=ci_values,
                label=function,
                capsize=5
            )

        plt.xlabel("Machine")
        plt.ylabel(metric_label)
        plt.title(f"{metric_label} - {group_name}")
        plt.xticks([pos + width * (len(functions) / 2 - 0.5) for pos in x], machine_labels_list, rotation=0, ha='center')
        plt.legend()
        plt.tight_layout()

        # Ensure output directory exists and save plot
        plots_dir = os.path.join(base_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        file_name = f"plot_{group_name.replace(' ', '_')}_{metric_en.replace(' ', '_').replace('(', '').replace(')', '').replace('%','perc')}.png"
        file_path = os.path.join(plots_dir, file_name)
        plt.savefig(file_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"High-resolution PNG plot saved: {file_path} (DPI: {DPI})")

# ===================================================================
# MAIN EXECUTION SECTION
#
# Generates plots for both single and parallel executions.
# 
# Automating this step ensures all relevant comparisons are visualized without manual intervention.
# ===================================================================
plot_group(single_run_data, "Single Executions")
plot_group(parallel_run_data, "Parallel Executions")