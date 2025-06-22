import warnings
from scanner_domain_shift.utilities_and_helpers.slide.customDataLoader import generate_dataloaders, generate_testloader_only, generate_balanced_testloader_only
from scanner_domain_shift.utilities_and_helpers.slide.custom_slide_container import *
from scanner_domain_shift.utilities_and_helpers.helper_methods import inst_model
import numpy as np
from scanner_domain_shift.utilities_and_helpers.slide.label_helper import load_label_dict, reverse_label_dict
import torch
import torch.optim as optim
import torch.nn as nn
import os
from sklearn.metrics import auc, confusion_matrix, roc_curve
import argparse
import json
import time
from sklearn.metrics import f1_score



#load all data for training, validation and testing, concatenate all data into one dataloader per set
def dataloading(scanners, annotation_file, dataprep_folder, test=False):
    # Load label dictionary
    label_dict = load_label_dict(annotation_file)
    reverse_dict = reverse_label_dict(label_dict)


    all_train_files = []
    all_valid_files = []
    all_test_files = []

    for scanner in scanners:
        if test:
            print(f"Loading test data for scanner {scanner}")
            test_data = torch.load(os.path.join(dataprep_folder, f'{scanner}_test.pt'))
            all_test_files.extend([SlideContainer.from_dict(data, annotation_file) for data in test_data])
           
            
        else:
            print(f"Loading train, valid, and test data for scanner {scanner}")
            # Load serialized data
            train_data = torch.load(os.path.join(dataprep_folder, f'{scanner}_train.pt'))
            valid_data = torch.load(os.path.join(dataprep_folder, f'{scanner}_valid.pt'))
            test_data = torch.load(os.path.join(dataprep_folder, f'{scanner}_test.pt'))

            # Reconstruct SlideContainer objects and concatenate
            all_train_files.extend([SlideContainer.from_dict(data, annotation_file) for data in train_data])
            all_valid_files.extend([SlideContainer.from_dict(data, annotation_file) for data in valid_data])
            all_test_files.extend([SlideContainer.from_dict(data, annotation_file) for data in test_data])

    print("Finished loading data for all scanners")

    reverse_dict = {2: "excluded", 1: "tumor", 0: "normal"}
    label_dict = {"excluded": 2, "tumor": 1, "normal": 0}

    batch_size = 64


    print("Generating dataloaders...")
    if test:
        test_loader = generate_balanced_testloader_only(all_test_files, batch_size, reverse_dict)
        return test_loader
    else:
        train_loader, valid_loader, test_loader = generate_dataloaders(all_train_files, all_valid_files, all_test_files, batch_size, reverse_dict)
        return train_loader, valid_loader, test_loader
    

def train(train_loader, valid_loader, modelname, basepath, lr, epochs, input_size, hidden_size, hidden_size1, hidden_size2, fscores):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, flatten_patches = inst_model(modelname, input_size, hidden_size, hidden_size1, hidden_size2, device)
    

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()    
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
        
        for i, (patches, labels, slide_ids) in enumerate(train_loader):
            patches, labels = patches.to(device), labels.to(device)

            patches = patches.view(patches.size(0), -1) if flatten_patches else patches
            
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
            for patches, labels, slide_ids in valid_loader:
                patches, labels = patches.to(device), labels.to(device)
                
                patches = patches.view(patches.size(0), -1) if flatten_patches else patches

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
            savepath = os.path.join(basepath, f'PAD_{modelname}_epoch_{epoch+1}_lr_{lr}.pth')
        elif modelname == "TwoLayerNN": 
            savepath = os.path.join(basepath, f'PAD_{modelname}_epoch_{epoch+1}_lr_{lr}_{hidden_size}.pth')       
        else:
            savepath = None
            KeyError(f"Model name {modelname} not recognized for saving path")
    

        if savepath:
            torch.save(model.state_dict(), savepath)
            print(f"Saved model to {savepath}")
    else:
        print("F1 score is not the highest. Model not saved.")

    return model, train_accuracies, valid_accuracies, f1

def test(test_loader, model, modelname, train_accuracies=[], valid_accuracies=[], modelpath = ""):
    print("------------------------------------")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    model.eval()
    correct_test = 0
    total_test = 0
    all_labels = []
    all_predictions = []
    all_probs = []  # Collect probabilities for ROC

    if modelname == "TwoLayerNN" or modelname == "ThreeLayerNN":
        flatten_patches = True
    else:
        flatten_patches = False

    print("testing ", modelname)

    with torch.no_grad():
        for patches, labels, slide_ids in test_loader:
            patches, labels = patches.to(device), labels.to(device)
            patches = patches.view(patches.size(0), -1) if flatten_patches else patches

            outputs = model(patches)
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)  # shape: [batch, num_classes]
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability for class 1

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

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.close()
    print(f"ROC AUC: {roc_auc:.2f}")
    
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

    
    print("------------------------------------")



    
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # Suppress the FutureWarning from torch.load
    warnings.filterwarnings("ignore", category=FutureWarning)


    parser = argparse.ArgumentParser(description='Train and test a model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')

    args = parser.parse_args()

    config = load_config(args.config)
    assert 'models' in config, "Please specify models in the config file."  
    models = config['models']  
    assert "modelsavepath" in config, "modelsavepath  not found in config"
    basepath = config['modelsavepath']
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    
    assert 'scanners' in config, "Scanner not found in config"
    scanners = config['scanners']
    assert 'dataprep_folder' in config, "Dataprep folder not found in config"
    dataprep_folder = config['dataprep_folder']
    assert 'annotation_file' in config, "Annotation file not found in config"
    annotation_file = config['annotation_file']

    assert 'lrs' in config, "Learning rates not found in config"
    assert 'epochs' in config, "Epochs not found in config"
    lrs = config['lrs']
    epochs = config['epochs']

    start_time = time.time()

   

    
    
    
    # Train and test each model configuration

    # for TwoLayerNN
    hidden_size = 512  # Number of neurons in the first hidden layer

    # for ThreeLayerNN
    hidden_size1 = 2048  # Number of neurons in the first hidden layer
    hidden_size2 = 512  # Number of neurons in the second hidden layer

    # Define the model
    input_size = 224 * 224 * 3  # Input size for 224x224 RGB images
   
    
    #load data for training
    print(f"Loading data for training and testing {scanners}")
    train_loader, valid_loader, test_loader = dataloading(scanners =scanners, annotation_file=annotation_file, dataprep_folder=dataprep_folder, test=False)
   

    # train according to config
    for modelname in config['models']:
        print(f"training and testing model {modelname}")
        for epochs in config['epochs']:
            fscores = []
            for lr in config['lrs']:
                print(f"training with lr {lr} for {epochs} epochs")
                
                
                
                model, train_accuracies, valid_accuracies, f1score = train(train_loader, valid_loader, modelname, basepath, lr, epochs, input_size, hidden_size, hidden_size1, hidden_size2, fscores)
                fscores.append(f1score)
                


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken in minutes: {elapsed_time / 60:.2f}")
