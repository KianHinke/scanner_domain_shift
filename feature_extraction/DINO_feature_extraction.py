import argparse
import json
import os
import time
import warnings
import numpy as np
import timm
import torch

from scanner_domain_shift.utilities_and_helpers.slide.customDataLoader import generate_dataloaders
from scanner_domain_shift.utilities_and_helpers.slide.custom_slide_container import SlideContainer
from transformers import AutoModel

#this script extracts patch features from the preprocessed images using DINO or Dinov2 models for latent space experiments

def get_all_dataloaders(scanners, annotation_file, dataprep_folder):
    reverse_dict = { 2: "excluded", 1 : "tumor", 0: "normal"}

    train_files_dict = {}
    valid_files_dict = {}
    test_files_dict = {}
    

    for scanner in scanners:    

        # Load serialized data
        train_data = torch.load(os.path.join(dataprep_folder, f'{scanner}_train.pt'))        
        valid_data = torch.load(os.path.join(dataprep_folder, f'{scanner}_valid.pt'))
        test_data = torch.load(os.path.join(dataprep_folder, f'{scanner}_test.pt'))
        
        train_files_dict[scanner] = [SlideContainer.from_dict(data, annotation_file) for data in train_data]
        valid_files_dict[scanner] = [SlideContainer.from_dict(data, annotation_file) for data in valid_data]
        test_files_dict[scanner] = [SlideContainer.from_dict(data, annotation_file) for data in test_data]
        
    batch_size = 256

    print("generating dataloaders...")
    
    valid_loaders = {}
    train_loaders = {}
    test_loaders = {}

    for scanner in scanners:        
        train_loaders[scanner], valid_loaders[scanner], test_loaders[scanner] = generate_dataloaders(train_files_dict[scanner], valid_files_dict[scanner], test_files_dict[scanner], batch_size, reverse_dict)
        
        print(f"Generated dataloaders for {scanner}")
    
    return train_loaders, valid_loaders, test_loaders

def load_Dino():
    # Create the DINO model
    model = timm.create_model(
        'vit_base_patch16_224.dino',
        pretrained=True,
        num_classes=0,  # No classification head
    )
    return model

def load_Dinov2():
    # Create the DINOv2 model
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    return model

def count_patches_per_slide(test_loader):
    # Count the number of patches for each slide in the test_loader
    slide_patch_count = {}

    for _, _, slideIDs in test_loader:
        for slideID in slideIDs:
            slideID = slideID.item() if isinstance(slideID, torch.Tensor) else slideID
            if slideID not in slide_patch_count:
                slide_patch_count[slideID] = 0
            slide_patch_count[slideID] += 1

    # Print the patch count for each slide
    # for slideID, count in slide_patch_count.items():
    #     print(f"Slide {slideID}: {count} patches")

    return slide_patch_count


def extract_patch_features(dataloader, loadertype, savepath, model, scanner, model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    feature_save_path = os.path.join(savepath, f"{scanner}_{loadertype}_patch_features.npy")
    label_save_path = os.path.join(savepath, f"{scanner}_{loadertype}_patch_labels.npy")
    all_patch_features = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (patches, y_patches, _) in enumerate(dataloader):
            patches = patches.to(device)
            y_patches = y_patches.to(device)
            if model_type == "Dinov2":
                outputs = model(pixel_values=patches)  # Forward pass for Dinov2
                patch_features = outputs.last_hidden_state[:, 0, :]  # get CLS token features
            else:
                patch_features = model(patches)  # Forward pass for DINO
            print(f"Batch {batch_idx + 1}: Extracted patch features shape: {patch_features.shape}")
            all_patch_features.append(patch_features.cpu().numpy())  #collect features
            all_labels.append(y_patches.cpu().numpy())  #Collect labels

    if all_patch_features and all_labels:
        concatenated_features = np.concatenate(all_patch_features, axis=0)  #Combine all features
        concatenated_labels = np.concatenate(all_labels, axis=0)  # combine all labels
        print(f"Final concatenated features shape: {concatenated_features.shape}")
        print(f"Final concatenated labels shape: {concatenated_labels.shape}")
        np.save(feature_save_path, concatenated_features)  # save features 
        np.save(label_save_path, concatenated_labels)  # Save labels 


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

if __name__ == '__main__':
    # Suppress the FutureWarning from torch.load
    warnings.filterwarnings("ignore", category=FutureWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    parser = argparse.ArgumentParser(description='Train and test a model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')

    args = parser.parse_args()

    config = load_config(args.config)

    #start time
    start = time.time()

    assert 'scanners' in config, "Please specify scanners in the config file."
    scanners = config['scanners']
    assert 'savepath' in config, "Please specify a savepath for the feature vectors in the config"
    savepath = config['savepath']
    assert 'model' in config, "Please specify a model in the config file."
    modelname = config['model']
    assert 'dataprep_folder' in config, "Please specify a dataprep_folder in the config file."
    dataprep_folder = config['dataprep_folder']
    assert 'annotation_file' in config, "Please specify an annotation_file in the config file."
    annotation_file = config['annotation_file']
    
    # Create the savepath directory if it doesn't exist
    if not os.path.exists(savepath):
        os.makedirs(savepath)


    #prepare all dataloaders for each scanner
    train_loaders, valid_loaders, test_loaders = get_all_dataloaders(scanners, annotation_file, dataprep_folder)
    for scanner in scanners:
        print(f"Extracting features for {scanner}...")
        test_loader = test_loaders[scanner]
        valid_loader = valid_loaders[scanner]
        train_loader = train_loaders[scanner]
        # count_patches_per_slide(test_loader)

        if modelname == "Dinov2":
            print("Using Dinov2 model")
            model = AutoModel.from_pretrained("facebook/dinov2-base")
            extract_patch_features(test_loader, "test", savepath, model, scanner, modelname)
            extract_patch_features(valid_loader, "valid", savepath, model, scanner, modelname)
            extract_patch_features(train_loader, "train", savepath, model, scanner, modelname)
        else:
            # Load the DINO model
            print("Using DINO model")
            model = load_Dino()
            extract_patch_features(test_loader, "test", savepath, model, scanner, modelname)
            extract_patch_features(valid_loader, "valid", savepath, model, scanner, modelname)
            extract_patch_features(train_loader, "train", savepath, model, scanner, modelname)
       
        
        
    print(f"completed in: {((time.time() - start) / 60)} minutes")