import argparse
import json
import os
import time
import warnings

import numpy as np
from models import TwoLayerNN, ThreeLayerNN
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

scanners  = ["cs2", "nz20", "nz210", "p1000", "gt450"]
label_mappings = {"cs2": 0, "nz20": 1, "nz210": 2, "p1000": 3, "gt450": 4}
batch_size = 256

def dataloading(save_loaders_path):
    # Load the concatenated loaders from the specified path
    loaded_loaders = {"train": None, "valid": None, "test": None}

    for loader_type in ["train", "valid", "test"]:
        loader_path = os.path.join(save_loaders_path, f"{loader_type}_loader.pt")
        all_representations, all_labels = torch.load(loader_path)
        dataset = TensorDataset(all_representations, all_labels)
        loaded_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loaded_loaders[loader_type] = loaded_loader
        print(f"{loader_type.capitalize()} loader loaded from {loader_path}")
    
    print("Concatenated loaders successfully loaded from disk.")
    return loaded_loaders



def parseName(modelpath):
    # Extract filename from the path
    model_filename = os.path.basename(modelpath)
    parts = model_filename.replace('.pth', '').split('_')  # Remove .pth and split

    try:
        # Extract model name
        modelname = parts[0]

        # Extract epochs and learning rate
        epoch_index = parts.index('epochs') + 1
        lr_index = parts.index('lr') + 1

        epochs = int(parts[epoch_index])
        lr = float(parts[lr_index])  

        # Extract hidden sizes
        if modelname == "TwoLayerNN":
            # Check if hidden size exists in the filename
            if len(parts) > epoch_index + 1:
                hidden_size = int(parts[epoch_index + 1])  # Hidden size follows learning rate
            else:
                hidden_size = None  # Default to None if not present
            hidden_size1 = None
            hidden_size2 = None
        elif modelname == "ThreeLayerNN":
            hidden_size1 = int(parts[epoch_index + 1])  # First hidden layer
            hidden_size2 = int(parts[epoch_index + 2])  # Second hidden layer
            hidden_size = None
        else:
            hidden_size = hidden_size1 = hidden_size2 = None

    except (ValueError, IndexError):
        return None

    return lr, epochs, modelname, hidden_size, hidden_size1, hidden_size2

def test_model(model_path, dataloader, input_size):
    """
    Load a trained model from the specified path and evaluate its performance on the test dataset.
    """
    # 5 classes    
    output_size = 5
    lr, epochs, modelname, hidden_size, hidden_size1, hidden_size2 = parseName(model_path)
    print(f"Testing model: {modelname}, epochs: {epochs}, learning rate: {lr}, hidden sizes: {hidden_size}, {hidden_size1}, {hidden_size2}")

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    if modelname == "ThreeLayerNN":
        model = ThreeLayerNN(input_size, hidden_size1, hidden_size2, output_size).to(device)
    elif modelname == "TwoLayerNN":
        model = TwoLayerNN(input_size, hidden_size, output_size).to(device)
    else:
        print(f"Unknown model type: {modelname}")
        return

    # Load the model state
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Initialize metrics
    true_positive = [0] * output_size
    false_positive = [0] * output_size
    false_negative = [0] * output_size
    total_correct = 0
    total_samples = 0

    # Collect all predictions and true labels for sklearn
    all_preds = []
    all_labels = []

    # Evaluate the model
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            # Move data to device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = x_batch.view(x_batch.size(0), -1)

            # Forward pass
            outputs = model(x_batch)
            _, predicted = torch.max(outputs, 1)

            # Track metrics
            total_correct += (predicted == y_batch).sum().item()
            total_samples += y_batch.size(0)

            # Collect for sklearn
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

            # Optionally keep legacy metrics if needed
            for i in range(len(y_batch)):
                label = y_batch[i].item()
                pred = predicted[i].item()
                if label == pred:
                    true_positive[label] += 1
                else:
                    false_positive[pred] += 1
                    false_negative[label] += 1

    # Calculate overall accuracy
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

    # Calculate classwise F1 scores using sklearn
    f1_scores = f1_score(all_labels, all_preds, average=None, labels=list(range(output_size)))
    avg_f1_score = f1_score(all_labels, all_preds, average='macro')

    # Print overall accuracy
    print(f"Overall Accuracy: {overall_accuracy:.4f}")

    # Print class-wise metrics
    for i in range(output_size):
        # Accuracy per class (recall)
        class_recall = true_positive[i] / (true_positive[i] + false_negative[i]) if (true_positive[i] + false_negative[i]) > 0 else 0
        
        #calculate class error as FNR
        class_fnr = 1- class_recall

        #calculate the fraction of instances that were falsely predicted as this class
        class_fdr = false_positive[i] / (false_positive[i] + true_positive[i]) if (false_positive[i] + true_positive[i]) > 0 else 0
        print(f"Class {i}: Recall: {class_recall:.4f}, FNR: {class_fnr:.4f}, FDR: {class_fdr:.4f} , F1 Score: {f1_scores[i]:.4f}")


    # # Calculate mean absolute error (MAE) over all predictions
    mae = np.mean(np.array(all_labels) != np.array(all_preds))
    print(f"Mean Absolute Error (MAE) over all classes: {mae:.4f}")


    # Calculate PAD*_multi value
    # calculate MAE_random for 5 classes
    # Assuming MAE_random is the expected error rate for a random classifier
    # For 5 classes, the expected error rate is 1 - (1/5) = 0.8
    # This means that a random classifier would be correct 20% of the time, leading to an error rate of 80%
    # Therefore, MAE:
    MAE_random =  1 - ( 1 / output_size)
    PAD_star_multi = 1 - (mae / MAE_random)
    print(f"PAD*_multi: {PAD_star_multi:.4f}")

    print(f"Average F1 Score: {avg_f1_score:.4f}")

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


#########################

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    parser = argparse.ArgumentParser(description='Train and test a model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')

    args = parser.parse_args()

    config = load_config(args.config)

    assert 'models_folder' in config, "models_folder key not found in config"
    models_folder = config['models_folder']
    assert 'datapath' in config, "datapath key not found in config"
    datapath = config['datapath']
    assert 'input_size' in config, "input_size key not found in config"
    input_size = config['input_size']

    # Start time tracking
    start_time = time.time()

    
    concatenated_loaders = dataloading(datapath)
    
    

    # Iterate through the savepath folder and test each model
    for model_file in os.listdir(models_folder):
        model_path = os.path.join(models_folder, model_file)
        if model_file.endswith(".pth"):  # Ensure it's a model file
            print(f"Testing model: {model_file}")
            print(model_path)
            test_model(model_path, concatenated_loaders["test"], input_size)
    
    # End time tracking
    end_time = time.time()
    # Calculate elapsed time in minutes
    elapsed_time = (end_time - start_time) / 60
    print(f"Elapsed time: {elapsed_time:.2f} minutes")

