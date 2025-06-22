from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import os

############################################################################################################
# KS
def create_ks_results_dataframe(ks_results, scan, scanner, class_idx):
    row = []
    for scan_item in scan:
        key = f'{scanner}_{scan_item}'
        if key in ks_results:
            result = ks_results[key]
            row.append(result[f'Class {class_idx}']['KS Statistic'])
        else:
            row.append(None)  # Append None if key is missing
    # Create a 1xN DataFrame (1 row, N columns)
    return pd.DataFrame([row], index=[scanner], columns=scan).replace({None: np.nan})


def create_p_values_dataframe(ks_results, scan, scanner, class_idx):
    row = []
    for scan_item in scan:
        key = f'{scanner}_{scan_item}'
        if key in ks_results:
            result = ks_results[key]
            row.append(result[f'Class {class_idx}']['p-value'])
        else:
            row.append(None)  # Append None if key is missing
    # Create a 1xN DataFrame (1 row, N columns)
    return pd.DataFrame([row], index=[scanner], columns=scan).replace({None: np.nan})

def write_matrix_to_json(matrix, scan, scanner, savepath, filename):
    """
    Write a matrix to a JSON file in a structured format.
    """
    structured_data = {
        "scanner": scanner,
        "scan": scan,
        "matrix": matrix
    }
    json_path = os.path.join(savepath, filename)
    with open(json_path, 'w') as f:
        json.dump(structured_data, f, indent=4)
    print(f"Matrix written to {json_path}")

def save_all_matrices_to_json(all_results, scan, scanner, savepath, filename):
    """
    Save all matrices (KS or MMD results and p-values) into a single JSON file.
    """
    structured_data = {
        "scanner": scanner,
        "scanners": scan,
        "results": all_results
    }
    json_path = os.path.join(savepath, filename)
    with open(json_path, 'w') as f:
        json.dump(structured_data, f, indent=4)
    print(f"All matrices saved to {json_path}")

def create_ks_heatmap_matrices(ks_results, scanner, scan, n_classes, savepath=None, bootstrap=False):
    if savepath is None:
        raise ValueError("A savepath must be provided to save the heatmaps.")

    all_ks_results = {"KS Statistics": {}, "p-values": {}}

    for n in range(n_classes):
        ks_df = create_ks_results_dataframe(ks_results, scan, scanner, class_idx=n)
        all_ks_results["KS Statistics"][f"Class {n}"] = ks_df.values.tolist()  # Collect KS statistics

        plt.figure(figsize=(len(scan) * 2, 4))
        sns.heatmap(ks_df, annot=True, cmap='coolwarm', cbar=True, linewidths=.5)
        if bootstrap:
            plt.title(f'{scanner} KS Statistic Heatmap for Class {n} (Bootstrapped)')
            plt.savefig(f'{savepath}/{scanner}_KS_statistic_heatmap_class_{n}_bootstrap.png')
        else:
            plt.title(f'KS Statistic Heatmap for Class {n}')
            plt.savefig(f'{savepath}/{scanner}_KS_statistic_heatmap_class_{n}.png')
        plt.close()

    for n in range(n_classes):
        p_values_df = create_p_values_dataframe(ks_results, scan, scanner, class_idx=n)
        all_ks_results["p-values"][f"Class {n}"] = p_values_df.values.tolist()  # Save p-values

        plt.figure(figsize=(len(scan) * 2, 4))
        sns.heatmap(p_values_df, annot=True, cmap='coolwarm', cbar=True, linewidths=.5)
        if bootstrap:
            plt.title(f'{scanner} p-value Heatmap for Class {n} (Bootstrapped)')
            plt.savefig(f'{savepath}/{scanner}_KS_p_value_heatmap_class_{n}_bootstrap.png')
        else:
            plt.title(f'{scanner} p-value Heatmap for Class {n}')
            plt.savefig(f'{savepath}/{scanner}_KS_p_value_heatmap_class_{n}.png')
        plt.close()

    save_all_matrices_to_json(all_ks_results, scan, scanner, savepath, f"{scanner}_KS_results.json")

############################################################################################################
# MMD

def create_mmd_results_dataframe(mmd_results, scan, scanner):
    row = []
    for scan_item in scan:
        key = f'{scanner}_{scan_item}'
        if key in mmd_results:
            row.append(mmd_results[key])
        else:
            row.append(None)  # Append None if key is missing
    # Create a 1xN DataFrame (1 row, N columns)
    return pd.DataFrame([row], index=[scanner], columns=scan).replace({None: np.nan})

def create_p_values_mmd_dataframe(p_values, scan, scanner):
    row = []
    for scan_item in scan:
        key = f'{scanner}_{scan_item}'
        if key in p_values:
            row.append(p_values[key])
        else:
            row.append(None)  # Append None if key is missing
    # Create a 1xN DataFrame (1 row, N columns)
    return pd.DataFrame([row], index=[scanner], columns=scan).replace({None: np.nan})

def create_mmd_heatmap_matrices(mmd_results, p_values, scan, scanner, savepath=None, perms=None):
    if savepath is None:
        raise ValueError("A savepath must be provided to save the heatmaps.")
    if perms is None:
        raise ValueError("The number of permutations (perms) must be provided.")

    all_mmd_results = {"MMD Statistics": {}, "p-values": {}}

    # Create MMD heatmap
    mmd_df = create_mmd_results_dataframe(mmd_results, scan, scanner)
    all_mmd_results["MMD Statistics"] = mmd_df.values.tolist()  # Save MMD statistics

    plt.figure(figsize=(len(scan) * 2, 4))
    sns.heatmap(mmd_df, annot=True, cmap='coolwarm', cbar=True, linewidths=.5)
    plt.title(f'{scanner} MMD Heatmap ({perms} Permutations)')
    plt.savefig(f'{savepath}/{scanner}_MMD_heatmap_{perms}_perms.png')
    plt.close()

    # Create p-value heatmap
    p_values_df = create_p_values_mmd_dataframe(p_values, scan, scanner)
    all_mmd_results["p-values"] = p_values_df.values.tolist()  # Save p-values

    plt.figure(figsize=(len(scan) * 2, 4))
    sns.heatmap(p_values_df, annot=True, cmap='coolwarm', cbar=True, linewidths=.5)
    plt.title(f'{scanner} p-value Heatmap ({perms} Permutations)')
    plt.savefig(f'{savepath}/{scanner}_MMD_p_value_heatmap_{perms}_perms.png')
    plt.close()

    save_all_matrices_to_json(all_mmd_results, scan, scanner, savepath, f"{scanner}_MMD_results.json")
