#!/usr/bin/env python3
"""
Script to parse memory usage data from txt files and plot memory usage over time.
Can process a single file or all txt files in a directory.
"""

import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_memory_file(filepath):
    """
    Parse the memory usage file and return a pandas DataFrame.

    Args:
        filepath: Path to the txt file with memory usage data

    Returns:
        pandas DataFrame with datetime and memory usage data
    """
    try:
        # Read the file with whitespace as delimiter
        df = pd.read_csv(filepath, sep=r"\s+", engine="python")

        # Combine date and time columns into a single datetime column
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])

        # Set datetime as index for better time series plotting
        df.set_index("datetime", inplace=True)

        return df
    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}")
        return None


def get_txt_files(path):
    """
    Get all txt files from a directory or return single file.

    Args:
        path: Path to file or directory

    Returns:
        List of txt file paths
    """
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        txt_files = glob.glob(os.path.join(path, "*.txt"))
        if not txt_files:
            print(f"Warning: No .txt files found in directory: {path}")
        return sorted(txt_files)
    else:
        return []


def plot_memory_usage_multiple(data_dict, output_file=None):
    """
    Plot the used memory over time for multiple files.

    Args:
        data_dict: Dictionary mapping filenames to DataFrames
        output_file: Optional path to save the plot (default: show plot)
    """
    plt.figure(figsize=(14, 8))

    # Color palette for multiple lines
    colors = plt.cm.tab10(range(len(data_dict)))

    # Plot each dataset
    for idx, (label, df) in enumerate(data_dict.items()):
        # Convert datetime to relative time in seconds
        time_relative = (df.index - df.index[0]).total_seconds()

        plt.plot(time_relative, df["used"], linewidth=1.5, color=colors[idx], label=label, alpha=0.8)

    # Add labels and title
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Memory Usage (MB)", fontsize=12)
    plt.title("Memory Usage Comparison Over Time", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10, framealpha=0.9)

    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle="--")

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show()


def print_statistics_multiple(data_dict):
    """
    Print comparative statistics for multiple memory usage datasets.

    Args:
        data_dict: Dictionary mapping filenames to DataFrames
    """
    print("\n" + "=" * 80)
    print("Memory Usage Comparative Statistics")
    print("=" * 80)

    # Print header
    header = f"{'File':<30} {'Avg (MB)':>12} {'Max (MB)':>12} {'Min (MB)':>12} {'Std (MB)':>12}"
    print(header)
    print("-" * 80)

    # Print statistics for each file
    for label, df in data_dict.items():
        avg_used = df["used"].mean()
        max_used = df["used"].max()
        min_used = df["used"].min()
        std_used = df["used"].std()

        # Truncate long filenames
        display_label = label[:28] + ".." if len(label) > 30 else label

        print(f"{display_label:<30} {avg_used:>12.2f} {max_used:>12.2f} " f"{min_used:>12.2f} {std_used:>12.2f}")

    print("=" * 80)

    # Print detailed stats for each file
    for label, df in data_dict.items():
        print(f"\nDetailed stats for: {label}")
        print(f"  Total memory:       {df['total'].iloc[0]:>10.2f} MB")
        print(f"  Duration:           {df.index[0]} to {df.index[-1]}")
        print(f"  Data points:        {len(df)}")

    print()


def main():
    """Main function to parse arguments and create the plot."""
    # Default input directory
    default_path = "nvflare_2_client"

    # Get input path from command line or use default
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = default_path

    # Check if path exists
    if not os.path.exists(input_path):
        print(f"Error: Path '{input_path}' not found!")
        print(f"Usage: {sys.argv[0]} [file_or_directory] [output_file]")
        print("\nExamples:")
        print(f"  {sys.argv[0]} baseline.txt")
        print(f"  {sys.argv[0]} ./results/")
        print(f"  {sys.argv[0]} ./results/ output.png")
        sys.exit(1)

    # Get output file if specified
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    # Get all txt files
    txt_files = get_txt_files(input_path)

    if not txt_files:
        print(f"Error: No valid txt files found at '{input_path}'")
        sys.exit(1)

    print(f"Found {len(txt_files)} txt file(s) to process:")
    for f in txt_files:
        print(f"  - {f}")

    # Parse all files
    data_dict = {}
    for filepath in txt_files:
        # Use filename without extension as label
        label = Path(filepath).stem

        print(f"\nParsing: {filepath}")
        df = parse_memory_file(filepath)

        if df is not None:
            data_dict[label] = df
            print(f"  Successfully loaded {len(df)} data points")

    if not data_dict:
        print("\nError: No files were successfully parsed!")
        sys.exit(1)

    print(f"\nSuccessfully parsed {len(data_dict)} file(s)")

    # Print statistics
    print_statistics_multiple(data_dict)

    # Create the plot
    plot_memory_usage_multiple(data_dict, output_file)


if __name__ == "__main__":
    main()
