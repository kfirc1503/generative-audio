import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_all_data(folder_path):
    """
    Reads all .json files in `folder_path`, each containing
    metrics for nppc and mc_dropout.
    """
    rows = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".json"):
            fpath = os.path.join(folder_path, fname)
            with open(fpath, "r") as f:
                data = json.load(f)

            # NPPC values
            nppc_rmse = data["nppc"]["rmse"]
            nppc_res = data["nppc"]["residual_error"]

            # MC Dropout values
            mc_rmse = data["mc_dropout"]["rmse"]
            mc_res = data["mc_dropout"]["residual_error"]

            # Append rows with simplified labels
            rows.append({
                "metric": "Reconstruction Error",
                "method": "NPPC",
                "value": nppc_rmse
            })
            rows.append({
                "metric": "Residual Error",
                "method": "NPPC",
                "value": nppc_res
            })
            rows.append({
                "metric": "Reconstruction Error",
                "method": "MC Dropout",
                "value": mc_rmse
            })
            rows.append({
                "metric": "Residual Error",
                "method": "MC Dropout",
                "value": mc_res
            })

    return pd.DataFrame(rows)


def plot_grouped_bars(df, save_path="figures/metrics_comparison.png"):
    """
    Plots and saves a grouped bar chart with two metrics on the x-axis.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Set style with white background and grid
    sns.set_style("whitegrid")

    # Define the exact colors from the example
    colors = ['#3182bd', '#fd8d3c']  # Lighter blue and lighter orange

    # Create figure with specific size
    plt.figure(figsize=(10, 6))

    # Create the plot
    ax = sns.barplot(
        data=df,
        x="metric",
        y="value",
        hue="method",
        ci="sd",
        capsize=0.2,
        errcolor="black",
        palette=colors
    )

    # Set labels
    ax.set_title("")
    ax.set_ylabel("Error", fontsize=20, fontweight='bold')
    # ax.set_ylabel("Error", fontsize=20)

    ax.set_xlabel("")

    # Increase tick label sizes
    ax.tick_params(axis='x', labelsize=20)  # x-axis labels (increased)
    ax.tick_params(axis='y', labelsize=18)  # y-axis labels (increased)

    ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')


    # Legend in bottom-right corner
    ax.legend(title="", loc="lower right", framealpha=0.9, edgecolor='white', fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    print(f"Figure saved to: {save_path}")

    # Display the plot
    plt.show()


def main():
    folder_path = r"./validation_nppc_results/validation_metrics"

    # Create DataFrame
    df = load_all_data(folder_path)

    # Plot and save with PNG format
    plot_grouped_bars(df, save_path="figures/metrics_comparison.png")


if __name__ == "__main__":
    main()
