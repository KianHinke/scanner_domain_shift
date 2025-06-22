
import warnings
from scanner_domain_shift.utilities_and_helpers.helper_methods import inst_model
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os
from sklearn.metrics import confusion_matrix
import argparse
import json
import time
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset



def dataloading(scanner, arraypath, batch_size=64, test=False):
    def create_dataloader(features_path, labels_path, batch_size, shuffle=True):
        # Load the feature vectors and labels from the .npy files
        features = np.load(features_path)
        labels = np.load(labels_path)

        # Convert the numpy arrays to PyTorch tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Create a TensorDataset to pair features and labels
        dataset = TensorDataset(features_tensor, labels_tensor)

        # Create a DataLoader to iterate through the dataset
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    # Example usage
    if test:
        features_path = f"{arraypath}/{scanner}_test_patch_features.npy"
        labels_path = f"{arraypath}/{scanner}_test_patch_labels.npy"
        return create_dataloader(features_path, labels_path, batch_size= batch_size, shuffle=False)
    else:
        train_features_path = f"{arraypath}/{scanner}_train_patch_features.npy"
        train_labels_path = f"{arraypath}/{scanner}_train_patch_labels.npy"
        valid_features_path = f"{arraypath}/{scanner}_valid_patch_features.npy"
        valid_labels_path = f"{arraypath}/{scanner}_valid_patch_labels.npy"
        test_features_path = f"{arraypath}/{scanner}_test_patch_features.npy"
        test_labels_path = f"{arraypath}/{scanner}_test_patch_labels.npy"

        train_loader = create_dataloader(train_features_path, train_labels_path, batch_size=batch_size, shuffle=True)
        valid_loader = create_dataloader(valid_features_path, valid_labels_path, batch_size=batch_size, shuffle=False)
        test_loader = create_dataloader(test_features_path, test_labels_path, batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader

    

def train(train_loader, valid_loader, modelname, basepath, scanner, lr, epochs, input_size, hidden_size, hidden_size1, hidden_size2, fscores):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model, _ = inst_model(modelname, input_size, hidden_size, hidden_size1, hidden_size2, device)
   

    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if modelname == "Dino":
        optimizer = optim.Adam(model.head.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    train_accuracies = []
    valid_accuracies = []

    print("------------------------------------")
    print(f"training {modelname} with lr={lr} for {epochs} epochs")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (patches, labels) in enumerate(train_loader):
            patches, labels = patches.to(device), labels.to(device)

            
            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Calculate and print progress every 5th batch
            # if (i + 1) % 5 == 0:
            #     progress = (i + 1) * batch_size / len(train_loader.dataset) * 100
            #     print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Progress: {progress:.2f}%")
        
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Training Accuracy: {train_accuracy:.2f}%")
        
        # Validation phase
        model.eval()
        correct_valid = 0
        total_valid = 0        
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for patches, labels in valid_loader:
                patches, labels = patches.to(device), labels.to(device)
                
                
                outputs = model(patches)
                _, predicted = torch.max(outputs.data, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())        
        
        valid_accuracy = 100 * correct_valid / total_valid
        valid_accuracies.append(valid_accuracy)
        print(f"Epoch {epoch+1}, Validation Accuracy: {valid_accuracy:.2f}%")
    
    print('Finished Training')
    print("Last epoch validation confusion matrix")
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print("               Predicted Class 0  Predicted Class 1")
    print(f"Actual Class 0       {conf_matrix[0, 0]}                 {conf_matrix[0, 1]}")
    print(f"Actual Class 1       {conf_matrix[1, 0]}                 {conf_matrix[1, 1]}")
    print("------------------------------------")

    # Ensure the 'models' directory exists
    os.makedirs('models', exist_ok=True)

    # Calculate F1 score for the last epoch
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f"F1 Score for the last epoch: {f1:.2f}")
    print(f"all f1 scores: {fscores}")
    # Check if the F1 score is higher than all values in the array "fscores"
    if all(f1 > score for score in fscores):
        
        if modelname == "ResNet18" or modelname == "ResNet50" or modelname == "ResNet101":
            savepath = os.path.join(basepath, f'{scanner}_{modelname}_epoch_{epoch+1}_lr_{lr}.pth')
        elif modelname == "TwoLayerNN": 
            savepath = os.path.join(basepath, f'{scanner}_{modelname}_epoch_{epoch+1}_lr_{lr}_{hidden_size}.pth')
        elif modelname == "ThreeLayerNN":
            savepath = os.path.join(basepath, f'{scanner}_{modelname}_epoch_{epoch+1}_lr_{lr}_{hidden_size1}_{hidden_size2}.pth')
        elif modelname == "ThreeLayerCNN" or modelname == "Dino" or modelname == "DenseNet121":
            savepath = os.path.join(basepath, f'{scanner}_{modelname}_epoch_{epoch+1}_lr_{lr}.pth')
        else:
            savepath = None
    

        if savepath:
            torch.save(model.state_dict(), savepath)
            print(f"Saved model to {savepath}")
    else:
        print("F1 score is not the highest. Model not saved.")

    return model, train_accuracies, valid_accuracies, f1

def test(test_loader, model, modelname, results_file, train_accuracies=[], valid_accuracies=[], modelpath = ""):
    print("------------------------------------")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
        
    model.eval()
    correct_test = 0
    total_test = 0
    all_labels = []
    all_predictions = []

    
    
   
    print("testing ", modelname)

    with torch.no_grad():
        for patches, labels in test_loader:
            patches, labels = patches.to(device), labels.to(device)

            
            outputs = model(patches)
            _, predicted = torch.max(outputs.data, 1)
            
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_accuracy = 100 * correct_test / total_test
    # test_accuracies.append(test_accuracy)
    print(f"Testing Accuracy: {test_accuracy:.2f}%")

    # # class 1 sensitivity
    all_labels_array = np.array(all_labels)  # Convert to NumPy array
    total_class1 = np.sum(all_labels_array == 1)
    if total_class1 > 0:
        class1_sensitivity = 100 * confusion_matrix(all_labels_array, all_predictions)[1, 1] / total_class1
    else:
        class1_sensitivity = 0.0  # Set sensitivity to 0 if no class 1 samples are present
    

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print("               Predicted Class 0  Predicted Class 1")
    print(f"Actual Class 0       {conf_matrix[0, 0]}                 {conf_matrix[0, 1]}")
    print(f"Actual Class 1       {conf_matrix[1, 0]}                 {conf_matrix[1, 1]}")

    # Print training and testing accuracies
    print(f"Training Accuracies: {train_accuracies}")
    print(f"Validation Accuracies: {valid_accuracies}")
    print(f"Testing Accuracies: {test_accuracy}")

    print(f"Class 1 Sensitivity: {class1_sensitivity:.2f}%")

    # Calculate F1 score
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f"F1 Score: {f1:.2f}")
    #f1 score per class
    f1_classwise = f1_score(all_labels, all_predictions, average=None)
    print(f"F1 Score per class: {f1_classwise}")

    
    # Append results to a text file
    with open(results_file, 'a') as f:
        f.write(f"Model: {modelname}\n")
        f.write(f"Model Path: {modelpath}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Validation Accuracies: {valid_accuracies}\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
        f.write(f"Class 1 Sensitivity: {class1_sensitivity:.2f}%\n")
        f.write(f"F1 Score: {f1:.2f}\n")
        f.write(f"F1 Score per class: {f1_classwise}\n")
        f.write("Confusion Matrix:\n")
        f.write("               Predicted Class 0  Predicted Class 1\n")
        f.write(f"Actual Class 0       {conf_matrix[0, 0]}                 {conf_matrix[0, 1]}\n")
        f.write(f"Actual Class 1       {conf_matrix[1, 0]}                 {conf_matrix[1, 1]}\n")
        f.write("------------------------------------\n")
    print("------------------------------------")



#check the modelnames or paths in the config
def check_config(config):
    print("Checking model configurations")
    if 'models' in config:
        "Training Configurations found"
        valid_modelnames = ["TwoLayerNN", "ThreeLayerNN", "ThreeLayerCNN", "ResNet18", "ResNet50", "Dino", "DenseNet121"]
        for modelname in config['models']:
            assert modelname in valid_modelnames, f"Invalid model name: {modelname}. Must be one of {valid_modelnames}"
        print("All model names are valid")
    if 'modelpaths' in config:
        "Testing Configurations found"
        for modelpath in config['modelpaths']:
            print(f"Checking modelpath {modelpath}")
            assert os.path.exists(modelpath), f"Model path {modelpath} does not exist"
        print("All model paths are valid")
    elif 'models_folder_path' in config:
        "Testing Configurations found" 
        print(f"Checking models folder path {config['models_folder_path']}")
        assert os.path.exists(config['models_folder_path']), f"Models folder path {config['models_folder_path']} does not exist"
        print("models folder path is valid")
    else:
        print("No model configurations found")

    
def parseName(modelpath):
    # Extract filename from the path
    model_filename = os.path.basename(modelpath)
    parts = model_filename.replace('.pth', '').split('_')  # Remove .pth and split

    try:
        #Extract scanner name
        scannername = parts[0]
        print("scannername:", scannername)
        # Extract model name
        modelname = parts[1]
        print("modelname:", modelname)

        # Extract epochs and learning rate
        epoch_index = parts.index('epoch') + 1
        lr_index = parts.index('lr') + 1
        epochs = int(parts[epoch_index])
        lr = float(parts[lr_index])  # Converts '0.001' correctly

        # Extract hidden sizes
        if modelname == "TwoLayerNN":
            hidden_size = int(parts[lr_index + 1])  # Hidden size follows learning rate
            hidden_size1 = None
            hidden_size2 = None
        elif modelname == "ThreeLayerNN":
            hidden_size1 = int(parts[lr_index + 1])  # First hidden layer
            hidden_size2 = int(parts[lr_index + 2])  # Second hidden layer
            hidden_size = None
        else:
            hidden_size = hidden_size1 = hidden_size2 = None

    except (ValueError, IndexError) as e:
        print("Error parsing modelpath:", str(e))
        return None

    print(f"Parsed epochs: {epochs}, learning rate: {lr}, modelname: {modelname}, hidden sizes: {hidden_size}, {hidden_size1}, {hidden_size2}")
    return scannername, lr, epochs, modelname, hidden_size, hidden_size1, hidden_size2

def get_all_pth_file_paths(folder_path):
        pth_file_paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.pth'):
                    pth_file_paths.append(os.path.join(root, file))
        return pth_file_paths

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

if __name__ == "__main__":
    #config file should contain either models or modelpaths or models_folder_path
    #for training: models (list of modelnames), scanner (the scanner of which the data should be used), lrs (list of learning rates), epochs (list of epochs)
    #for testing: modelpaths (list of model paths) or models_folder_path (path to folder containing models)
    # Suppress the FutureWarning from torch.load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warnings.filterwarnings("ignore", category=FutureWarning)


    parser = argparse.ArgumentParser(description='Train and test a model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')

    args = parser.parse_args()

    config = load_config(args.config)

    start_time = time.time()

   
    
    check_config(config)
    print("All model configurations are valid")
    
    
    
    # Train and test each model configuration

    # for TwoLayerNN
    hidden_size = 512  # Number of neurons in the first hidden layer

    # for ThreeLayerNN
    hidden_size1 = 1024  # Number of neurons in the first hidden layer
    hidden_size2 = 512  # Number of neurons in the second hidden layer

    # Define the model
    # input_size = 224 * 224 * 3  # Input size for 224x224 RGB images
    input_size = 768 #Input size for extracted feature vectors

    #decide to load model for testing or train+test

    #test loop with loaded models
    #with path to folder containing models...
    assert "arraypath" in config, "Arraypath not found in config"
    arraypath = config['arraypath']
    assert "results_file" in config, "Results file not found in config"
    results_file = config['results_file']

   

    scanner_before = "none" #
    if 'models_folder_path' in config:        

        folder_path = config['models_folder_path']
        all_pth_file_paths = get_all_pth_file_paths(folder_path)
        all_pth_file_paths.sort() #sort to ensure that same scanners are loaded after each other to avoid reloading data
        print(f"found the following paths:")
        print(all_pth_file_paths)

        for path in all_pth_file_paths:
                #loading loop
                scanner, lr, epochs, modelname, hidden_size, hidden_size1, hidden_size2 = parseName(path)
                if scanner != scanner_before:
                    print(f"Loading data for testing {scanner}")  
                    test_loader =  dataloading(scanner, arraypath=arraypath, test=True)
                    scanner_before = scanner

                print(f"loading model {modelname} of scanner {scanner} from {path} for testing")
                model, _ = inst_model(modelname, input_size, hidden_size, hidden_size1, hidden_size2, device)
                model.load_state_dict(torch.load(path, weights_only=True))
                
                test(test_loader, model, modelname, results_file, modelpath=path)

    # ...or individually defined modelpaths
    if 'modelpaths' in config:

        for path in config['modelpaths']:
            #loading loop
            scanner, lr, epochs, modelname, hidden_size, hidden_size1, hidden_size2 = parseName(path)
            if scanner != scanner_before:
                print("Loading data for testing")
                test_loader =  dataloading(scanner, arraypath=arraypath, test=True)
                scanner_before = scanner

            print(f"loading model {modelname} of scanner {scanner} from {path} for testing")

            model, _ = inst_model(modelname, input_size, hidden_size, hidden_size1, hidden_size2, device)
            model.load_state_dict(torch.load(path, weights_only=True))
            
            test(test_loader, model, modelname,results_file, modelpath=path)
    
    #train loop
    if 'models' in config:
        assert "modelsavepath" in config, "Model save path not found in config"
        modelsavepath = config['modelsavepath']
        assert 'scanners' in config, "Scanner not found in config"
        scanners = config['scanners']
        for scanner in scanners:
            print(f"Loading data for training and testing {scanner}")
            train_loader, valid_loader, test_loader = dataloading(scanner =scanner, arraypath=arraypath)
            assert 'lrs' in config, "Learning rates not found in config"
            assert 'epochs' in config, "Epochs not found in config"
            lrs = config['lrs']
            epochs = config['epochs']
            for modelname in config['models']:
                print(f"training and testing model {modelname} for scanner {scanner}")
                for epochs in config['epochs']:
                    fscores = []
                    for lr in config['lrs']:
                        print(f"training with lr {lr} for {epochs} epochs")
                        basepath = modelsavepath
                        #ensure folder exists
                        os.makedirs(basepath, exist_ok=True)
                        # Train model
                        model, train_accuracies, valid_accuracies, f1score = train(train_loader, valid_loader, modelname, basepath, scanner, lr, epochs, input_size, hidden_size, hidden_size1, hidden_size2, fscores)
                        fscores.append(f1score)
                        # Test model
                        test(test_loader, model, modelname, results_file, train_accuracies, valid_accuracies)
                    


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken in minutes: {elapsed_time / 60:.2f}")
