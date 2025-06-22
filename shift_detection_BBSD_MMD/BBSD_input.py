import argparse
import json
import os
import time
import warnings
import torch
from scanner_domain_shift.utilities_and_helpers.BBSD_logic import get_ks_results, get_softmax_outs
from scanner_domain_shift.utilities_and_helpers.slide.customDataLoader import generate_testloader_only
from scanner_domain_shift.utilities_and_helpers.slide.custom_slide_container import SlideContainer
from scanner_domain_shift.utilities_and_helpers.BBSD_vishelpers import create_ks_heatmap_matrices



def get_testloaders(scan, annotation_file, dataprep_folder):
    reverse_dict = {  1 : "tumor", 0: "normal"}

    test_files_dict = {}

    for scanner in scan:            
        test_data = torch.load(os.path.join(dataprep_folder, f'{scanner}_test.pt'))

        # Reconstruct SlideContainer objects        
        test_files_dict[scanner] = [SlideContainer.from_dict(data, annotation_file) for data in test_data]

    batch_size = 64

    print("generating dataloaders...")
    test_loaders = {}   
    for scanner in scan:
        test_loaders[scanner] = generate_testloader_only(test_files_dict[scanner], batch_size, reverse_dict)    

    return test_loaders


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
  
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
    assert 'dataprep_folder' in config, "dataprep_folder key not found in config"
    dataprep_folder = config['dataprep_folder']
    assert 'annotation_file' in config, "annotation_file key not found in config"
    annotation_file = config['annotation_file']
    assert 'savepath' in config, "savepath key not found in config"
    savepath = config['savepath']
    assert 'n_bootstraps' in config, "n_bootstraps key not found in config"
    n_bootstraps = config['n_bootstraps']

    # Load test data into dictionary (test_loaders[scanner])
    test_loaders = get_testloaders(scan, annotation_file, dataprep_folder)
    print("All test loaders generated")

   
    input_size = 224 * 224 * 3  # Input size for 224x224 RGB images

    
    scannerwise_softmax_outputs = {}
    assert 'modelpaths' in config, "modelpaths key not found in config"
    modelpaths = config['modelpaths']
    for scanner in scan:
        print(f"Selected scanner: {scanner}")
        #get softmax outputs for all scanners in a dictionary
        modelpath = modelpaths[scanner]       
        print(f"Selected model: {modelpath}")
        scannerwise_softmax_outputs[scanner] = get_softmax_outs(modelpath, input_size, scan, test_loaders)

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
    
    
    


    #KS test loop that tests and creates heatmaps for each scanner
    print("ks test")
    all_ks_results = {}
    #do this for every scanner separately
    for scanner in scan:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        #perform ks test
        all_ks_results[scanner] = get_ks_results(scannerwise_softmax_outputs[scanner],scanner, scan, bootstrap=True, n_bootstraps=n_bootstraps)
        print("All ks test results computed")
        
        # Ensure the savepath folder exists and create it if not
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        #check what all_ks_results's structure looks like
        print(f"all_ks_results: {all_ks_results.keys()}")


        create_ks_heatmap_matrices(all_ks_results[scanner],scanner, scan, n_classes=2, savepath=savepath, bootstrap=True)

        print(f"heatmap figures for {scanner} created")
        print("##############################################################")


        
    #print execution duration    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken in minutes: {elapsed_time / 60:.2f}")
