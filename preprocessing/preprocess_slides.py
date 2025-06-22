
import argparse
import json
from scanner_domain_shift.utilities_and_helpers.slide.label_helper import load_label_dict
from scanner_domain_shift.utilities_and_helpers.slide.process_slides import load_slides
import os
import torch

# This script preprocesses whole slide images (WSIs) for training, validation, and testing.

def load(scanner, annotation_file, splitconfig_path, images_path):
    
    label_dict = load_label_dict(annotation_file)
    train_files, valid_files, test_files = load_slides(
            splitconfig_path,
            patch_size=224,
            label_dict=label_dict,
            level=0,
            image_path=images_path,
            annotation_file=annotation_file,
            scanner=scanner,
            excluded_labels={-1,0}, # Exclude unassigned and background labels
            negative_class_labels = {1,2,3,4,5,6},
            positive_class_labels = {7,8,9,10,11,12,13} #tumor
        )
    return train_files, valid_files, test_files

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser(description='Train and test a model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')

    args = parser.parse_args()

    config = load_config(args.config)
    assert "dataprep_folder" in config, "Config must contain 'dataprep_folder' path to store preprocessed data."
    dataprep_folder = config["dataprep_folder"]
    assert "annotation_file" in config, "Config must contain 'annotation_file' path to the annotation file."
    annotation_file = config["annotation_file"]
    assert "splitconfig_path" in config, "Config must contain 'splitconfig_path' path to the split configuration file."
    splitconfig_path = config["splitconfig_path"]
    assert "images_path" in config, "Config must contain 'images_path' to the directory of raw .tif images."
    images_path = config["images_path"]

    # Create the dataprep folder if it doesn't exist    
    os.makedirs(dataprep_folder, exist_ok=True)

    scan = ['cs2', 'nz20', 'nz210', 'p1000', 'gt450']
    # Call load for each scanner and save the results
    for scanner in scan:
        train_files, valid_files, test_files = load(scanner, annotation_file,splitconfig_path, images_path)

        # Serialize SlideContainer objects
        train_data = [slide.to_dict() for slide in train_files]
        valid_data = [slide.to_dict() for slide in valid_files]
        test_data = [slide.to_dict() for slide in test_files]

        # Save serialized data
        torch.save(train_data, os.path.join(dataprep_folder, f'{scanner}_train.pt'))
        torch.save(valid_data, os.path.join(dataprep_folder, f'{scanner}_valid.pt'))
        torch.save(test_data, os.path.join(dataprep_folder, f'{scanner}_test.pt'))