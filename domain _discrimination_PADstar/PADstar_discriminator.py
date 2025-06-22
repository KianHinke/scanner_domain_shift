import argparse
import json
import os
import time
import warnings

import timm
from models import TwoLayerNN
import torch
from scanner_domain_shift.utilities_and_helpers.helper_methods import inst_model
from scanner_domain_shift.utilities_and_helpers.slide.customDataLoader import generate_dataloaders, generate_testloader_only
from scanner_domain_shift.utilities_and_helpers.slide.custom_slide_container import SlideContainer
from scanner_domain_shift.utilities_and_helpers.slide.label_helper import load_label_dict, reverse_label_dict
import random
from torch.utils.data import TensorDataset, DataLoader
"""
PAD_star.py
This script is designed for the PAD*_multi experiment. It operates in two main stages:
1. Feature Extraction and Concatenated Dataloader Creation:
    - The script first loads datasets from all scanners and balances the number of samples across scanners for train, validation, and test splits.
    - It then uses a pretrained task-classifier model (either a fine-tuned ResNet18 or a DINO foundation model) to extract feature vectors (reduced representations) from the image data.
    - For each sample, the corresponding scanner name is then used as the label, and both the feature vectors and labels are stored.
    - New dataloaders are created for each split (train, valid, test) and each scanner, containing the extracted feature vectors and scanner labels.
    - These dataloaders are then concatenated across all scanners to form single train, validation, and test loaders, which are saved to disk for later use.
    # The script first creates new concatenated dataloaders that contain the feature vectors data of all scanners extracted by a model trained on the original tumor task and the scanner names as labels.
2. PAD* Domain Discriminator Training:
    - In the second part of the script, the concatenated dataloaders are loaded from disk.
    - A simple neural network (PAD* domain discriminator) is trained to predict the scanner label from the extracted feature vectors.
    - The model's performance is evaluated on the validation set, and the trained model is saved to disk.
    # In the second part of the script, a simple PAD* domain discriminator model is trained and saved.

Note:
- Paths and hyperparameters should be adjusted as needed for your environment and experimental setup.
"""

scanners  = ["cs2", "nz20", "nz210", "p1000", "gt450"]
label_mappings = {"cs2": 0, "nz20": 1, "nz210": 2, "p1000": 3, "gt450": 4}
batch_size = 256


output_size = 5
#For TwoLayerNN
hidden_size = 256



def load_Dino():
    # Create the DINO model
    model = timm.create_model(
        'vit_base_patch16_224.dino',
        pretrained=True,
        num_classes=0,  
    )
    return model.to(device)

def dataloading(scanner,annotation_file, dataprep_folder, test=False):
    # Load label dictionary   
    label_dict = load_label_dict(annotation_file)
    reverse_dict = reverse_label_dict(label_dict)

    if test:
        print("loading test data")
        test_data = torch.load(os.path.join(dataprep_folder, f'{scanner}_test.pt'))
        test_files = [SlideContainer.from_dict(data, annotation_file) for data in test_data]
        print("generating test dataloader...")
        test_loader = generate_testloader_only(test_files, batch_size=batch_size, reverse_dict=reverse_dict)
        return test_loader
    else:
        print("loading train, valid and test data")
        # Load serialized data
        train_data = torch.load(os.path.join(dataprep_folder, f'{scanner}_train.pt'))
        valid_data = torch.load(os.path.join(dataprep_folder, f'{scanner}_valid.pt'))
        test_data = torch.load(os.path.join(dataprep_folder, f'{scanner}_test.pt'))

        # Reconstruct SlideContainer objects
        train_files = [SlideContainer.from_dict(data, annotation_file) for data in train_data]
        valid_files = [SlideContainer.from_dict(data, annotation_file) for data in valid_data]
        test_files = [SlideContainer.from_dict(data, annotation_file) for data in test_data]

        print("generating dataloaders...")
        train_loader, valid_loader, test_loader = generate_dataloaders(
            train_files, valid_files, test_files, batch_size=batch_size, reverse_dict= reverse_dict
        )
        return train_loader, valid_loader, test_loader


def train(dataloaders, savepath, num_epochs, lr, input_size):
    

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model, loss function, and optimizer
    model = TwoLayerNN(input_size, hidden_size, output_size).to(device)  # Move model to GPU
    modelname = "TwoLayerNN"

    criterion = torch.nn.CrossEntropyLoss().to(device)  # move loss function to GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print(f"Training {modelname} model for {num_epochs} epochs with lr {lr},hidden size {hidden_size} on {device}")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        class_correct = torch.zeros(output_size, dtype=torch.int32)
        class_total = torch.zeros(output_size, dtype=torch.int32)

        train_loader = dataloaders["train"]
        total_batches = len(train_loader)  # Total number of batches
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader, start=1):
            # Print progress
            # if batch_idx % 10 == 0:
            #     print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{total_batches}]")

            # Move data to device (if using GPU)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = x_batch.view(x_batch.size(0), -1)  # Flatten the input

            # Forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

            # Track class-wise metrics
            for label in range(output_size):
                class_correct[label] += ((predicted == label) & (y_batch == label)).sum().item()
                class_total[label] += (y_batch == label).sum().item()

        
        # Evaluate on validation data
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_class_correct = torch.zeros(output_size, dtype=torch.int32)
        val_class_total = torch.zeros(output_size, dtype=torch.int32)
        with torch.no_grad():
            valid_loader = dataloaders["valid"]
            for x_batch, y_batch in valid_loader:
                # Move data to device
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                x_batch = x_batch.view(x_batch.size(0), -1)

                # Forward pass
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                # Track validation metrics
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == y_batch).sum().item()
                val_total += y_batch.size(0)

                # Track class-wise metrics
                for label in range(output_size):
                    val_class_correct[label] += ((predicted == label) & (y_batch == label)).sum().item()
                    val_class_total[label] += (y_batch == label).sum().item()

        # Print validation metrics for the epoch
        val_loss /= val_total
        val_accuracy = val_correct / val_total
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        for label in range(output_size):
            if val_class_total[label] > 0:
                val_class_acc = val_class_correct[label] / val_class_total[label]
                print(f"Class {label} Validation Accuracy: {val_class_acc:.4f}")

    #calculate fscore from last epoch's validation accuracy
    fscore = 2 * (val_accuracy * val_accuracy) / (val_accuracy + val_accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation F1 Score: {fscore:.4f}")
    # Save the trained model
    if modelname == "TwoLayerNN":
        model_filename = os.path.join(savepath, f"{modelname}_lr_{lr}_epochs{num_epochs}_{hidden_size}_fscore_{fscore}.pth")
    else: 
        KeyError("Model name not recognized for saving.")
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")

# Add this function to balance the test sets
def balance_sets(loaders):
    """
    Balance the number of samples across all test sets by limiting to the smallest dataset size.
    The subset is selected randomly.
    """
    # Find the smallest dataset size
    min_size = min(len(loader.dataset) for loader in loaders.values())
    print(f"Balancing test sets to the smallest size: {min_size}")

    # Create balanced test loaders
    balanced_loaders = {}
    for scanner, loader in loaders.items():
        # Randomly sample indices
        indices = random.sample(range(len(loader.dataset)), min_size)
        subset = torch.utils.data.Subset(loader.dataset, indices)
        balanced_loaders[scanner] = DataLoader(subset, batch_size=batch_size, shuffle=True)
    
    return balanced_loaders

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

######################################################
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Train and test a model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')

    args = parser.parse_args()

    config = load_config(args.config)

    assert "annotation_file" in config, "annotation_file key not found in config"
    annotation_file = config["annotation_file"]
    assert "dataprep_folder" in config, "dataprep_folder key not found in config"
    dataprep_folder = config["dataprep_folder"]
    assert "modelpath" in config, "modelpath key not found in config"
    modelpath = config["modelpath"]
    assert "modelsavepath" in config, "modelsavepath key not found in config"
    modelsavepath = config["modelsavepath"]
    # Ensure the save path exists
    os.makedirs(modelsavepath, exist_ok=True)
    assert "save_loaders_path" in config, "save_loaders_path key not found in config"
    save_loaders_path = config["save_loaders_path"]
    assert "input_size" in config, "input_size key not found in config"
    input_size = config["input_size"]
    assert "modelname" in config, "modelname key not found in config"
    modelname = config["modelname"]

    # Start time tracking
    start_time = time.time()

    
    

    
    

#########################################################################################
# this is the part that creates new loaders with reduced representations and saves them
# it uses a classifier model that was finetuned on the tumor task to extract the representations

    

    test_loaders = {}
    valid_loaders = {}
    train_loaders = {} 
    for scanner in scanners:
        print(f"Loading test data for scanner: {scanner}")
        train_loader, valid_loader, test_loader = dataloading(scanner,annotation_file, dataprep_folder, test=False)
        test_loaders[scanner] = test_loader
        valid_loaders[scanner] = valid_loader
        train_loaders[scanner] = train_loader


    

    train_loaders = balance_sets(train_loaders)
    valid_loaders = balance_sets(valid_loaders)
    test_loaders = balance_sets(test_loaders)

    # Print the number of samples in each dataset
    print("Number of samples in each train dataset:")
    for scanner in scanners:
        print(f"{scanner}: {len(train_loaders[scanner].dataset)}")
    print("Number of samples in each valid dataset:")
    for scanner in scanners:
        print(f"{scanner}: {len(valid_loaders[scanner].dataset)}")
    print("Number of samples in each test dataset:")
    for scanner in scanners:
        print(f"{scanner}: {len(test_loaders[scanner].dataset)}")


    if modelname == "DINO":
        # Load DINO model
        model = load_Dino()
        model.eval()
        print("DINO model loaded and set to evaluation mode.")
    elif modelname == "ResNet18":
        # load resnet18 model
        resnet18_model, _ = inst_model("ResNet18", None, None, None, None, device)
        resnet18_model.load_state_dict(torch.load(modelpath))
        # Remove the classification head to get reduced feature representations
        resnet18_model = torch.nn.Sequential(*list(resnet18_model.children())[:-1])
        model = resnet18_model
        model.eval()  # Set the model to evaluation mode
        print("ResNet18 model modified to output reduced feature representations.")

    



    
    new_loaders = {"train": {}, "valid": {}, "test": {}}

    for loader_type, loaders in [("train", train_loaders), ("valid", valid_loaders), ("test", test_loaders)]:
        for scanner, loader in loaders.items():
            print(f"Processing {loader_type} loader for scanner: {scanner}")
            all_representations = []
            all_labels = []

            with torch.no_grad():
                for x_batch, _, _ in loader:
                    x_batch = x_batch.to(device)
                    # Extract reduced representations
                    representations = model(x_batch)
                    representations = representations.view(representations.size(0), -1)  # Flatten the output
                    all_representations.append(representations.cpu())
                    all_labels.extend([label_mappings[scanner]] * representations.size(0))  # Assign scanner label

            # Combine all representations and labels
            all_representations = torch.cat(all_representations, dim=0)
            all_labels = torch.tensor(all_labels)

            # Create a dataset and loader
            dataset = TensorDataset(all_representations, all_labels)
            new_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            new_loaders[loader_type][scanner] = new_loader

    print("New loaders created with reduced representations and scanner labels for train, valid, and test sets.")

    # Inspect the content of one of the new loaders, e.g., "cs2"
    cs2_train_loader = new_loaders["train"]["nz20"]
    x_batch, y_batch = next(iter(cs2_train_loader))
    print(f"Shape of x_batch: {x_batch.shape}")
    print(f"Shape of y_batch: {y_batch.shape}")
    print(f"Labels in the batch: {y_batch}")

    # Concatenate the new_loaders["train"], new_loaders["valid"], and new_loaders["test"] for all scanners
    concatenated_loaders = {"train": None, "valid": None, "test": None}

    for loader_type in ["train", "valid", "test"]:
        all_representations = []
        all_labels = []

        for scanner, loader in new_loaders[loader_type].items():
            for x_batch, y_batch in loader:
                all_representations.append(x_batch)
                all_labels.append(y_batch)

        # Combine all representations and labels
        all_representations = torch.cat(all_representations, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Create a concatenated dataset and loader
        dataset = TensorDataset(all_representations, all_labels)
        concatenated_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        concatenated_loaders[loader_type] = concatenated_loader


    # Save the concatenated loaders to a specified path
    os.makedirs(save_loaders_path, exist_ok=True)

    for loader_type in ["train", "valid", "test"]:
        concatenated_loader = concatenated_loaders[loader_type]
        all_representations = []
        all_labels = []

        for x_batch, y_batch in concatenated_loader:
            all_representations.append(x_batch)
            all_labels.append(y_batch)

        # Combine all representations and labels
        all_representations = torch.cat(all_representations, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Save the data
        torch.save((all_representations, all_labels), os.path.join(save_loaders_path, f"{loader_type}_loader.pt"))
        print(f"{loader_type.capitalize()} loader saved to {os.path.join(save_loaders_path, f'{loader_type}_loader.pt')}")

    # #############################################################################################################
    # # load the loaders from disk

    
    # Load the concatenated loaders from the specified path
    loaded_loaders = {"train": None, "valid": None, "test": None}

    for loader_type in ["train", "valid", "test"]:
        loader_path = os.path.join(save_loaders_path, f"{loader_type}_loader.pt")
        all_representations, all_labels = torch.load(loader_path)
        dataset = TensorDataset(all_representations, all_labels)
        loaded_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loaded_loaders[loader_type] = loaded_loader
        print(f"{loader_type.capitalize()} loader loaded from {loader_path}")
    concatenated_loaders = loaded_loaders

    print("Concatenated loaders successfully loaded from disk.")

    # Check if the concatenated loaders contain data
    for loader_type in ["train", "valid", "test"]:
        concatenated_loader = concatenated_loaders[loader_type]
        x_batch, y_batch = next(iter(concatenated_loader))
        print(f"Loader type: {loader_type}")
        print(f"Shape of x_batch: {x_batch.shape}")
        print(f"Shape of y_batch: {y_batch.shape}")
        print(f"Unique labels in {loader_type} loader: {torch.unique(y_batch)}")
    
    # Print the label distribution for each loader in concatenated loaders
    for loader_type in ["train", "valid", "test"]:
        concatenated_loader = concatenated_loaders[loader_type]
        all_labels = []
        for _, y_batch in concatenated_loader:
            all_labels.extend(y_batch.tolist())
        label_counts = {label: all_labels.count(label) for label in set(all_labels)}
        print(f"Label distribution for {loader_type} loader: {label_counts}")

   
    # Train the PAD* domain discriminator model using the concatenated loaders
    train(concatenated_loaders, modelsavepath, num_epochs=10, lr=0.0001, input_size=input_size)


    # End time tracking
    end_time = time.time()
    # Calculate elapsed time in minutes
    elapsed_time = (end_time - start_time) / 60
    print(f"Elapsed time: {elapsed_time:.2f} minutes")

