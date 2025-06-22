import warnings
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
import torch

from helper_methods import inst_model
from scanner_domain_shift.classifier_training.model_train_test import parseName

#############################################################################################
# Softmax outputs

def plot_softmax_distribution(softmax_outputs, scanner, bins=50):
    """
    Plots a histogram of softmax output distributions for a 2-class classifier.
    
    Parameters:
        softmax_outputs (numpy.ndarray): Softmax probabilities with shape (N, 2).
        bins (int): Number of bins for the histogram.
    """
    class_0_probs = softmax_outputs[:, 0]  # Probabilities for class 0
    class_1_probs = softmax_outputs[:, 1]  # Probabilities for class 1

    plt.figure(figsize=(10, 5))
    
    # Plot histograms for both class probabilities
    plt.hist(class_0_probs, bins=bins, alpha=0.6, color='blue', label='Class 0')
    plt.hist(class_1_probs, bins=bins, alpha=0.6, color='red', label='Class 1')

    # Labels and title
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.title("Softmax Output Distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"softmax_distribution_{scanner}.png")  # Saves the plot as an image



#extract softmax outputs from a model for a test_loader
def extract_softmax_outputs(test_loader, model, modelname):
    """
    Extracts softmax outputs from a pretrained model for a given test loader (one scanner)
    """

    print("------------------------------------")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
        
    model.eval()
    softmax_outputs = []

    if modelname == "TwoLayerNN" or modelname == "ThreeLayerNN":
        flatten_patches = True
    else:
        flatten_patches = False

    with torch.no_grad():
        for patches, _, _ in test_loader:
            patches = patches.to(device)

            patches = patches.view(patches.size(0), -1) if flatten_patches else patches
            
            outputs = model(patches)
            softmax_output = torch.nn.functional.softmax(outputs, dim=1)
            
            softmax_outputs.extend(softmax_output.cpu().numpy())

    softmax_outputs = np.array(softmax_outputs)
    # print(f"Softmax outputs shape: {softmax_outputs.shape}")

    return softmax_outputs

#wrapper function to extract softmax outputs for each scanner
def get_softmax_outs(modelpath, input_size,scan, test_loaders):
    """
    Extracts softmax outputs from a pretrained model for each scanner in the dataset.

    Returns:
        all_softmax_outputs (dict): Dictionary containing softmax outputs for each scanner.

    Example access on output:
    all_softmax_outputs['scanner1']
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_softmax_outputs = {}
    path = modelpath
    
    
    # Loading model
    _, _, _, modelname, hidden_size, hidden_size1, hidden_size2 = parseName(path)
    print(f"Loading model {modelname} from {path} for softmax extraction")

    model, _ = inst_model(modelname, input_size, hidden_size, hidden_size1, hidden_size2,device)
    model.load_state_dict(torch.load(path, weights_only=True))

    # Softmax extraction with a single loaded model on every scanner
    for scanner in scan:
        print(f"Selected scanner: {scanner}")     
        softmax_outputs = extract_softmax_outputs(test_loaders[scanner], model, modelname)
        # plot_softmax_distribution(softmax_outputs, scanner=scanner)
        all_softmax_outputs[scanner] = softmax_outputs
    return all_softmax_outputs
##################################################################################################

# Kolmogorov-Smirnov Test
def ks_test_softmax(softmax_A, softmax_B):
    """
    Performs Kolmogorov-Smirnov test on softmax outputs from two different scanners.

    Parameters:
        softmax_A (np.ndarray): Softmax outputs from Scanner A (shape: [N, num_classes])
        softmax_B (np.ndarray): Softmax outputs from Scanner B (shape: [M, num_classes])

    Returns:
        ks_results (dict): Dictionary containing KS statistics and p-values for each class.
    """
    num_classes = softmax_A.shape[1]
    ks_results = {}

    for class_idx in range(num_classes):
        ks_stat, p_value = ks_2samp(softmax_A[:, class_idx], softmax_B[:, class_idx])
        ks_results[f'Class {class_idx}'] = {'KS Statistic': ks_stat, 'p-value': p_value}

    return ks_results

#bootstraps the ks test to handle sample size imbalance and bonferroni-corrects the p-values
def ks_test_balanced_bonferroni(softmax_A, softmax_B, n_bootstraps):
    """
    Performs KS test on softmax outputs while handling sample size imbalance using bootstrapping.

    Parameters:
        softmax_A (np.ndarray): Softmax outputs from Scanner A (N samples, num_classes)
        softmax_B (np.ndarray): Softmax outputs from Scanner B (M samples, num_classes)
        n_bootstraps (int): Number of bootstrap iterations for stability.
        

    Returns:
        ks_results (dict): Contains median KS statistics and Bonferroni-corrected p-values across bootstraps.
    """
    num_classes = softmax_A.shape[1]
    ks_results = {}

    # Ensure same number of samples for both scanners by taking the minimum of the two sample sizes
    min_samples = min(len(softmax_A), len(softmax_B))

    ks_stats_list = []
    p_values_list = []

    for _ in range(n_bootstraps):
        sampled_A = softmax_A[np.random.choice(len(softmax_A), min_samples, replace=False)]
        sampled_B = softmax_B[np.random.choice(len(softmax_B), min_samples, replace=False)]

        ks_stats = []
        p_values = []

        for class_idx in range(num_classes):
            ks_stat, p_value = ks_2samp(sampled_A[:, class_idx], sampled_B[:, class_idx])
            ks_stats.append(ks_stat)
            p_values.append(p_value)

        ks_stats_list.append(ks_stats)
        p_values_list.append(p_values)

    
    
    ks_results = {
        f"Class {i}": {
            "KS Statistic": np.median([ks[i] for ks in ks_stats_list]),
            #multiply p-value by number of classes for Bonferroni correction
            "p-value": np.median([(p[i] * num_classes) for p in p_values_list])
        }
        for i in range(num_classes)
    }

    

    return ks_results

#print the results of the ks test in a separate matrix for each class
def print_ks_results_matrix(ks_results, scan, bootstrap=False):
    print("bootstrap ", bootstrap)

    # Print results in a matrix format
    print("\nKS Test Results Matrix Class 0:")
    header = "Scanners".ljust(10) + "\t" + "\t".join([scanner.ljust(80) for scanner in scan])
    print(header)
    for i in range(len(scan)):
        row = [scan[i].ljust(10)]
        for j in range(len(scan)):
            if i <= j:
                key = f'{scan[i]}_{scan[j]}'
                if key in ks_results:
                    result = ks_results[key]
                    row.append(f"Class 0: KS={result['Class 0']['KS Statistic']:.10f}, p={result['Class 0']['p-value']:.10f} | Class 1: KS={result['Class 1']['KS Statistic']:.10f}, p={result['Class 1']['p-value']:.10f}".ljust(80))
                else:
                    row.append("KeyError".ljust(80))
            else:
                key = f'{scan[j]}_{scan[i]}'
                if key in ks_results:
                    result = ks_results[key]
                    row.append(f"Class 0: KS={result['Class 0']['KS Statistic']:.10f}, p={result['Class 0']['p-value']:.10f} | Class 1: KS={result['Class 1']['KS Statistic']:.10f}, p={result['Class 1']['p-value']:.10f}".ljust(80))
                else:
                    row.append("KeyError".ljust(80))
        print("\t".join(row))    

    print("\nKS Test Results Matrix Class 1:")
    header = "Scanners".ljust(10) + "\t" + "\t".join([scanner.ljust(80) for scanner in scan])
    print(header)
    for i in range(len(scan)):
        row = [scan[i].ljust(10)]
        for j in range(len(scan)):
            if i <= j:
                key = f'{scan[i]}_{scan[j]}'
                if key in ks_results:
                    result = ks_results[key]
                    row.append(f"Class 1: KS={result['Class 1']['KS Statistic']:.10f}, p={result['Class 1']['p-value']:.10f}".ljust(80))
                else:
                    row.append("KeyError".ljust(80))
            else:
                key = f'{scan[j]}_{scan[i]}'
                if key in ks_results:
                    result = ks_results[key]
                    row.append(f"Class 1: KS={result['Class 1']['KS Statistic']:.10f}, p={result['Class 1']['p-value']:.10f}".ljust(80))
                else:
                    row.append("KeyError".ljust(80))
        print("\t".join(row))      

#wrapper function to perform ks test for all scanners
def get_ks_results(all_softmax_outputs, scanner, scan, bootstrap, n_bootstraps=100):    
    print(f"performing test for outputs obtained by {scanner} model.")
    ks_results = {}
    if bootstrap == True:
            print(f"bootstrapped ks test")
    else:
        print(f"ks test")

    softmax_A = all_softmax_outputs[scanner]


    for scanner_B in scan:
        softmax_B = all_softmax_outputs[scanner_B]

        softmax_A /= softmax_A.sum(axis=1, keepdims=True)
        softmax_B /= softmax_B.sum(axis=1, keepdims=True)

        if bootstrap:
            ks_results[f'{scanner}_{scanner_B}'] = ks_test_balanced_bonferroni(softmax_A, softmax_B, n_bootstraps=n_bootstraps)
            
        else:            
            warnings.warn("ks_test_softmax without bootstrapping does not account for bonferroni correction. Use with caution.")
            ks_results[f'{scanner}_{scanner_B}'] = ks_test_softmax(softmax_A, softmax_B)
    
    return ks_results

