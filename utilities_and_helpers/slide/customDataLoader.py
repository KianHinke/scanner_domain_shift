import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np

'''
Custom DataLoader utility for SlideContainer patch datasets.

Defines a SlideDataset class that organizes patches from multiple slides using the precomputed patch grid,
assigns binary labels (tumor/normal), and provides functions to generate balanced or unbalanced PyTorch DataLoaders
for training, validation, and testing.
'''

class SlideDataset(Dataset):
    def __init__(self, slide_containers, label_dict):
        self.label_dict = label_dict
        self.slide_containers = slide_containers
        self.patches_by_slide = self._generate_patches_by_slide()

    # Return a list of tuples, each containing a slide and a patch index
    def _generate_patches_by_slide(self):
        patches_by_slide = {}
        for slide in self.slide_containers:
            slide_id = slide.get_file_name()  
            patches_by_slide[slide_id] = [(slide, idx) for idx in range(len(slide.patch_grid))]
        return patches_by_slide

    def __len__(self):
        return sum(len(patches) for patches in self.patches_by_slide.values())

    def __getitem__(self, idx):
        # Find the slide and patch index based on the global index
        cumulative_patches = 0
        for slide_id, patches in self.patches_by_slide.items():
            if idx < cumulative_patches + len(patches):
                slide, patch_idx = patches[idx - cumulative_patches]
                break
            cumulative_patches += len(patches)

        patch, y_patch = slide.get_patch_from_grid(patch_idx)

        # Error if the slide still contains excluded patches
        if all(self.label_dict[label] == "excluded" for label in np.unique(y_patch)):
            raise ValueError("All labels in y_patch are excluded although the grid should not contain such patches at this point")

        # Label the patch as "tumor" if it contains any "tumor" labels
        elif any(self.label_dict[label] == "tumor" for label in np.unique(y_patch)):
            y_patch = 1
        else:
            y_patch = 0  # Otherwise, label it as "normal"

        patch = torch.tensor(patch).permute(2, 0, 1).float()  # Convert to CHW format and float tensor
        y_patch = torch.tensor(y_patch).long()  # Convert to long tensor for classification
        return patch, y_patch, slide_id 

    def get_patches_by_slide(self, slide_id):
        return self.patches_by_slide.get(slide_id, [])
 

def generate_dataloaders(train_files, valid_files, test_files, batch_size, reverse_dict):
    train_dataset = SlideDataset(train_files, reverse_dict)
    valid_dataset = SlideDataset(valid_files, reverse_dict)
    test_dataset = SlideDataset(test_files, reverse_dict)

    #calculate and print class counts for test data
    class_counts_test = torch.zeros(2)
    for _, labels, _ in DataLoader(test_dataset, batch_size=batch_size, shuffle=False):
        class_counts_test += torch.bincount(labels, minlength=2)
    print("class counts test", class_counts_test)

    # Calculate class counts from the training data
    class_counts = torch.zeros(2)
    for _, labels, _ in DataLoader(train_dataset, batch_size=batch_size, shuffle=False):
        class_counts += torch.bincount(labels, minlength=2)


    # Compute weights (inverse of frequency)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for _, labels, _ in DataLoader(train_dataset, batch_size=batch_size, shuffle=False) for label in labels]

    # Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Calculate class counts from the validation data
    valid_class_counts = torch.zeros(2)
    for _, labels, _ in DataLoader(valid_dataset, batch_size=batch_size, shuffle=False):
        valid_class_counts += torch.bincount(labels, minlength=2)

    # Compute weights (inverse of frequency) for validation data
    valid_class_weights = 1.0 / valid_class_counts
    valid_sample_weights = [valid_class_weights[label] for _, labels, _ in DataLoader(valid_dataset, batch_size=batch_size, shuffle=False) for label in labels]

    # Create a WeightedRandomSampler for validation data
    valid_sampler = WeightedRandomSampler(valid_sample_weights, len(valid_sample_weights))

    # Create DataLoaders with the sampler for the training set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader, test_loader

def generate_unbalanced_dataloaders(train_files, valid_files, test_files, batch_size, reverse_dict):
    train_dataset = SlideDataset(train_files, reverse_dict)
    valid_dataset = SlideDataset(valid_files, reverse_dict)
    test_dataset = SlideDataset(test_files, reverse_dict)

    # Create DataLoaders with the sampler for the training set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader, test_loader

def generate_testloader_only( test_files, batch_size, reverse_dict):
    
    test_dataset = SlideDataset(test_files, reverse_dict)

    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return test_loader

def generate_balanced_testloader_only(test_files, batch_size, reverse_dict):
    test_dataset = SlideDataset(test_files, reverse_dict)

    # Calculate class counts from the test data
    class_counts = torch.zeros(2)
    for _, labels, _ in DataLoader(test_dataset, batch_size=batch_size, shuffle=False):
        class_counts += torch.bincount(labels, minlength=2)

    # Compute weights (inverse of frequency)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for _, labels, _ in DataLoader(test_dataset, batch_size=batch_size, shuffle=False) for label in labels]

    # Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create DataLoader with the sampler for the test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)

    return test_loader

