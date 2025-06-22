import argparse
import json
import os
import time
import warnings
import torch
import numpy as np
from scanner_domain_shift.utilities_and_helpers.BBSD_logic import get_ks_results, get_softmax_outs
from scanner_domain_shift.utilities_and_helpers.MMD_logic import mmd_permutation_test
from torch.utils.data import DataLoader, TensorDataset
from scanner_domain_shift.utilities_and_helpers.BBSD_vishelpers import create_ks_heatmap_matrices, create_mmd_heatmap_matrices




def dataloading(scanners, arraypath, batch_size=64, balanced=False):
    def create_dataloader(features_path, labels_path, batch_size, shuffle=False, balanced=False):
        # Load the feature vectors and labels from the .npy files
        features = np.load(features_path)
        labels = np.load(labels_path)

        if balanced:
            # Find the indices of each class
            class_0_indices = np.where(labels == 0)[0]
            class_1_indices = np.where(labels == 1)[0]

            # Determine the minimum number of samples between the two classes
            min_samples = min(len(class_0_indices), len(class_1_indices))

            # Randomly sample indices to balance the classes
            np.random.shuffle(class_0_indices)
            np.random.shuffle(class_1_indices)
            balanced_indices = np.concatenate([class_0_indices[:min_samples], class_1_indices[:min_samples]])

            # Shuffle the balanced indices
            np.random.shuffle(balanced_indices)

            # Subset the features and labels
            features = features[balanced_indices]
            labels = labels[balanced_indices]

        # Print the number of samples per class
        unique, counts = np.unique(labels, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        print(f"Class distribution: {class_distribution}")

        # Convert the numpy arrays to PyTorch tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Create a TensorDataset to pair features and labels
        dataset = TensorDataset(features_tensor, labels_tensor)

        # Create a DataLoader to iterate through the dataset
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader
    
    testloaders = {}
    for scanner in scanners:
        # Example usage    
        features_path = f"{arraypath}/{scanner}_test_patch_features.npy"
        labels_path = f"{arraypath}/{scanner}_test_patch_labels.npy"
        testloaders[scanner] = create_dataloader(features_path, labels_path, batch_size=batch_size, shuffle=False, balanced=balanced)

    return testloaders
    

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

############################################################################################################
  
if __name__ == "__main__":
    # Suppress the FutureWarning from torch.load
    warnings.filterwarnings("ignore", category=FutureWarning)


    parser = argparse.ArgumentParser(description='Train and test a model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')

    args = parser.parse_args()
    config = load_config(args.config)
    start_time = time.time()
   
    assert 'scanners' in config, "scanners key not found in config"
    scan = config['scanners']
    print(f"Selected scanners: {scan}")

    assert 'arraypath' in config, "arraypath key not found in config"
    arraypath = config['arraypath']
    print("using feature vectors from ", arraypath)


    balanced = False
    
    # Load test data into dictionary (test_loaders[scanner])
    test_loaders = dataloading(arraypath=arraypath, scanners=scan, batch_size=64, balanced=balanced)
    print("All test loaders generated")

    input_size = 768  # Input size for latent features

    
    
    assert 'modelpaths' in config, "modelpaths key not found in config"
    modelpaths = config['modelpaths']
    
    
    assert "savepath" in config, "savepath key not found in config"
    savepath = config['savepath']
    
    assert "bootstrap" in config or "MMD_perms" in config, "Either bootstrap or MMD_perms key not found in config"
    
    if "bootstrap" in config:
        assert "n_bootstraps" in config, "n_bootstraps key not found in config"
        n_bootstraps = config['n_bootstraps']
        bootstrap = config['bootstrap']
        scannerwise_softmax_outputs = {}
        for scanner in scan:
            print(f"Selected scanner: {scanner}")
            #get softmax outputs for all scanners in a dictionary
            modelpath = modelpaths[scanner]       
            print(f"Selected model: {modelpath}")
            scannerwise_softmax_outputs[scanner] = get_softmax_outs(modelpath, input_size)

            # Inspect predicted class distribution
            # for scannerB in scan:
            #     predicted_classes = np.argmax(scannerwise_softmax_outputs[scanner][scannerB], axis=1)
            #     unique, counts = np.unique(predicted_classes, return_counts=True)
            #     class_distribution = dict(zip(unique, counts))
            #     print(f"Predicted class distribution for data from {scannerB} on scanner {scanner}: {class_distribution}")
            #     #additionally calculate the percentages of these classes inside the class_distribution variable
            #     total_count = sum(counts)
            #     percentages = {k: (v / total_count) * 100 for k, v in class_distribution.items()}
            #     formatted_percentages = {k: f"{v:.2f}%" for k, v in percentages.items()}
            #     print(f"Predicted class distribution for data from {scannerB} on scanner {scanner}: {formatted_percentages}")
            
            print(f"got softmax outputs extracted for all scanners on {scanner} model")
            print("------------------------------------")
        
    

        print("KS test")
        all_ks_results = {}

        for scanner in scan:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            all_ks_results[scanner] = get_ks_results(scannerwise_softmax_outputs[scanner], scanner, scan, bootstrap=bootstrap, n_bootstraps=100)
            print(f"All KS test results computed for scanner {scanner}")

            if not os.path.exists(savepath):
                os.makedirs(savepath)

            create_ks_heatmap_matrices(all_ks_results[scanner], scanner, scan, n_classes=2, savepath=savepath, bootstrap=bootstrap)
            print(f"Heatmap figures and JSON for {scanner} created")
            print("##############################################################")

    elif "MMD_perms" in config:       
        
        perms = config['MMD_perms']
        print(f"MMD permutation test with {perms} permutations")

        all_mmd_results = {}
        all_p_values = {}

        for scanner in scan:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

            all_mmd_results[scanner], all_p_values[scanner] =  mmd_permutation_test(test_loaders,scanner, scan, perms)
            print(f"All MMD permutation test results computed for scanner {scanner}")
            print("##############################################################")

            if not os.path.exists(savepath):
                os.makedirs(savepath)
            create_mmd_heatmap_matrices(all_mmd_results[scanner], all_p_values[scanner], scan,scanner, perms=perms, savepath=savepath)
            #save matrices to json file.
            print("All heatmap figures created")
    else:
        KeyError("missing key in config")

        
    #print execution duration    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken in minutes: {elapsed_time / 60:.2f}")
