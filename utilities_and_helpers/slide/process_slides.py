
import pandas as pd
import glob
from pathlib import Path
from slide.custom_slide_container import SlideContainer
from tqdm import tqdm

#used by preprocess_slides.py to load slides, create patch grids, and return lists of SlideContainer objects for training, validation, and testing

def create_indices(files):
    """ Generate indices corresponding to patches from all slides. """
    indices = []
    for i, slide in enumerate(files):
        num_patches = len(slide.patch_grid)  # Use the precomputed patch grid size
        indices += num_patches * [i]
    return indices

def load_slides(splitconfig_path, patch_size=256, label_dict=None, level=None, image_path=None, annotation_file=None, scanner=None, excluded_labels={}, negative_class_labels={}, positive_class_labels={}):
    train_files = []
    valid_files = []
    test_files = []

    splitconfig = pd.read_csv(splitconfig_path, delimiter=";")

    for index, row in tqdm(splitconfig.iterrows()):
        try:
            image_file = Path(glob.glob("{}/{}_{}.tif".format(str(image_path), row["Slide"], scanner), recursive=True)[0])
        except IndexError:
            print(f"Warning: Slide {row['Slide']} not found for scanner {scanner}. Skipping.")
            continue

        slide = SlideContainer(image_file, annotation_file, level, patch_size, patch_size, label_dict=label_dict, excluded_labels=excluded_labels, negative_class_labels=negative_class_labels, positive_class_labels=positive_class_labels)
        
        # Generate a grid of patches with minimal overlap
        slide.generate_patch_grid()

        if row["Dataset"] == "train":
            train_files.append(slide)
        elif row["Dataset"] == "val":
            valid_files.append(slide)
        elif row["Dataset"] == "test":
            test_files.append(slide)

    return train_files, valid_files, test_files


