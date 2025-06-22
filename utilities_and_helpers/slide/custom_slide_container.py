import openslide
import cv2
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import matplotlib.cm as cm

"""
Custom SlideContainer class for WSI patch extraction and annotation handling.

This class is adapted from the original SlideContainer implementation by Wilm et al. (2023):
- Retains core logic for slide/annotation loading, patch extraction, and label mask creation.
- Adds support for flexible excluded/positive/negative class labels, patch grid generation, and serialization.
- Includes new functionality for patch grid management, 2-class label transformation, and advanced visualization.

"""

class SlideContainer:

    def get_file_name(self):
        return self.file.name
    
    def get_slide_shape(self):
        return self.slide.level_dimensions[self._level]

    
    def __init__(self, file: Path, annotation_file, level: int = 0, width: int = 256, height: int = 256,
                sample_func=None, label_dict=None, excluded_labels={}, negative_class_labels={}, positive_class_labels={}):
        '''
        This init class was largely adapted from the original code
        and handles the annotation and background identification.
        '''
        self.file = file
        with open(annotation_file) as f:
            data = json.load(f)
            self.tissue_classes = dict(zip([cat["name"] for cat in data["categories"]], 
                                        [cat["id"] for cat in data["categories"]]))
            image_id = [i["id"] for i in data["images"] if i["file_name"] == file.name][0]
            self.polygons = [anno for anno in data['annotations'] if anno["image_id"] == image_id]

        self.labels = set([poly["category_id"] for poly in self.polygons])

        self.slide = openslide.open_slide(str(file))

        # Background detection using Otsu thresholding
        self.thumbnail = np.array(self.slide.read_region((0, 0), 3, self.slide.level_dimensions[3]))[:, :,:3]
        grayscale = cv2.cvtColor(self.thumbnail, cv2.COLOR_RGB2GRAY)
        grayscale[grayscale == 0] = 255
        blurred = cv2.GaussianBlur(grayscale, (5,5), 0)
        self.white, self.mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find tissue bounding box
        self.x_min, self.y_min = np.min(
            np.array([np.min(np.array(polygon['segmentation']).reshape((-1, 2)), axis=0) for polygon in self.polygons]),
            axis=0)
        self.x_max, self.y_max = np.max(
            np.array([np.max(np.array(polygon['segmentation']).reshape((-1, 2)), axis=0) for polygon in self.polygons]),
            axis=0)

        self.width = width
        self.height = height
        self.down_factor = self.slide.level_downsamples[level]

        if level is None:
            level = self.slide.level_count - 1
        self._level = level
        self.sample_func = sample_func
        self.label_dict = label_dict
        self.excluded_labels = excluded_labels
        self.negative_class_labels = negative_class_labels
        self.positive_class_labels = positive_class_labels
        

        # Initialize empty list for patch grid (for non-random patching)
        self.patch_grid = []


    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        self.down_factor = self.slide.level_downsamples[value]
        self._level = value

    @property
    def shape(self):
        return self.width, self.height

    @property
    def slide_shape(self):
        return self.slide.level_dimensions[self._level]
    ############################################################
    # serialize for preprocessing
    def to_dict(self):
        """Serialize the essential data of the SlideContainer."""
        return {
            "file": str(self.file),
            "level": self._level,
            "width": self.width,
            "height": self.height,
            "label_dict": self.label_dict,
            "excluded_labels": self.excluded_labels,
            "negative_class_labels": self.negative_class_labels,
            "positive_class_labels": self.positive_class_labels,
            "patch_grid": self.patch_grid,
        }

    @classmethod
    def from_dict(cls, data, annotation_file):
        """Reconstruct a SlideContainer from serialized data."""
        obj = cls(
            file=Path(data["file"]),
            annotation_file=annotation_file,
            level=data["level"],
            width=data["width"],
            height=data["height"],
            label_dict=data["label_dict"],
            excluded_labels=data["excluded_labels"],
            negative_class_labels=data["negative_class_labels"],
            positive_class_labels=data["positive_class_labels"],
        )
        obj.patch_grid = data["patch_grid"]
        return obj
    
    ############################################################
    # Patch grid methods and visualization
    
    def generate_patch_grid(self):
        """
        Generate a list of (x, y) coordinates for patches across the slide.
        Ensures full coverage by adaptively calculating the stride.
        Additionally excludes patches that contain only excluded labels.
        """
        patch_positions = []
        wsi_width, wsi_height = self.slide.level_dimensions[self._level]

        # Calculate the number of patches needed
        num_x_patches = math.ceil((wsi_width - self.width) / self.width) + 1
        num_y_patches = math.ceil((wsi_height - self.height) / self.height) + 1

        # Adjust stride to ensure perfect coverage
        self.x_stride = max(1, (wsi_width - self.width) // (num_x_patches - 1))
        self.y_stride = max(1, (wsi_height - self.height) // (num_y_patches - 1))

        for y in range(0, wsi_height - self.height + 1, self.y_stride):
            for x in range(0, wsi_width - self.width + 1, self.x_stride):
                y_patch = self.get_y_patch(x, y)
                unique_labels = np.unique(y_patch)

                # Exclude patches that contain only labels in excluded_labels
                if set(unique_labels).issubset(self.excluded_labels):
                    continue

                patch_positions.append((x, y))

        self.patch_grid = patch_positions  # Store for later retrieval
        return patch_positions

    def visualize_label_overlay_full_slide(self, alpha: float = 0.5, colormap_name='jet'):
        """
        Visualize the label mask overlayed on the entire slide image with a legend.
        Args:
            alpha (float): Transparency level for the overlay (0.0 to 1.0).
            colormap_name (str): Name of the colormap to use for the label mask.
        """
        # Read the whole slide image at the curent level
        slide_image = np.array(self.slide.read_region((0, 0), self._level, self.slide.level_dimensions[self._level]))[:, :, :3]

        #create an empty label mask for the entire slide
        wsi_width, wsi_height = self.slide.level_dimensions[self._level]
        label_mask = -1 * np.ones((wsi_height, wsi_width), dtype=np.int8)

        #Fill the label mask with annotations
        inv_map = {v: k for k, v in self.tissue_classes.items()}
        for poly in self.polygons:
            coordinates = np.array(poly['segmentation']).reshape((-1, 2)) / self.down_factor
            label = self.label_dict[inv_map[poly["category_id"]]]
            cv2.drawContours(label_mask, [coordinates.reshape((-1, 1, 2)).astype(int)], -1, label, -1)

        # normalize the label mask for visualization
        normalized_mask = (label_mask - label_mask.min()) / (label_mask.max() - label_mask.min() + 1e-5)

        # Apply colormap to the label mask
        colormap = cm.get_cmap(colormap_name)
        colored_mask = colormap(normalized_mask)[:, :, :3]  # Drop alpha channel

        # cverlay the label mask on the slide image
        overlay = (1 - alpha) * slide_image / 255.0 + alpha * colored_mask

        # Create the figure
        plt.figure(figsize=(12, 12))
        plt.imshow(overlay)
        plt.title("Label Overlay on Full Slide")
        plt.axis("off")

        # add a legend
        unique_labels = np.unique(label_mask[label_mask >= 0])  # exclude the background (-1)
        legend_patches = []
        for label in unique_labels:
            color = colormap((label - label_mask.min()) / (label_mask.max() - label_mask.min() + 1e-5))
            legend_patches.append(patches.Patch(color=color, label=inv_map[label]))

        plt.legend(handles=legend_patches, loc='upper right', title="Labels", bbox_to_anchor=(1.2, 1))
        plt.show()
    
    def get_patches(self):
        """
        Retrieve the precomputed patch grid.
        Returns:
            List of (x, y) coordinates for patches.
        """
        if not hasattr(self, "patch_grid") or len(self.patch_grid) == 0:
            raise ValueError("Patch grid not generated. Call generate_patch_grid() first.")
        return self.patch_grid
    
    
    
    
    def get_patch_from_grid(self, index: int):
        """
        Retrieve a patch using the index from the precomputed grid.
        Returns:
            Tuple (image_patch, label_mask)
        """
        if not hasattr(self, "patch_grid") or len(self.patch_grid) == 0:
            raise ValueError("Patch grid not generated. Call generate_patch_grid() first.")

        x, y = self.patch_grid[index]
        x_patch, y_patch = self.get_patch(x, y), self.get_2class_y_patch(x, y) #get y_patch that is transformed to 3 classes
        return x_patch, y_patch
    
    # visualize the grid of patches (non-excluded) on the original slide image
    def visualize_grid(self, colormap_name='hsv'):
        """
        Visualize the grid of patches on the original slide image.
        """
        if not hasattr(self, "patch_grid") or len(self.patch_grid) == 0:
            raise ValueError("Patch grid not generated. Call generate_patch_grid() first.")

        # read the whole slide image
        slide_image = np.array(self.slide.read_region((0, 0), self._level, self.slide.level_dimensions[self._level]))[:, :, :3]

        # create a figure and axis to display the slide image
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(slide_image)

        colormap = cm.get_cmap(colormap_name, len(self.patch_grid))

        # Overlay the grid on the slide image with different colors
        for i, (x, y) in enumerate(self.patch_grid):

            color = colormap(i % colormap.N)  # modulo to cycle through the colormap if there are more patches than colors
            rect = patches.Rectangle((x, y), self.width, self.height, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        plt.title(f"Grid of patches on slide: {self.get_file_name()}")
        plt.show()

    ############################################################
    #helper functions

    def get_new_level(self):
        return self._level

    def get_patch(self, x: int = 0, y: int = 0):
        patch = np.array(self.slide.read_region(location=(int(x * self.down_factor), int(y * self.down_factor)),
                                               level=self._level, size=(self.width, self.height)))
        # Some scanners use 0 in 4th dimension to indicate background -> fill with white
        patch[patch[:, :, -1] == 0] = [255, 255, 255, 0]
        return patch[:,:,:3]

    # Get y_patch with original label values (is used in get_2class_y_patch)
    def get_y_patch(self, x: int = 0, y: int = 0):
        y_patch = -1 * np.ones(shape=(self.height, self.width), dtype=np.int8)
        inv_map = {v: k for k, v in self.tissue_classes.items()}

        for poly in self.polygons:
            coordinates = np.array(poly['segmentation']).reshape((-1, 2)) / self.down_factor
            coordinates = coordinates - (x, y)
            label = self.label_dict[inv_map[poly["category_id"]]]
            
            cv2.drawContours(y_patch, [coordinates.reshape((-1, 1, 2)).astype(int)], -1, label, -1)

        white_mask = cv2.cvtColor(self.get_patch(x, y), cv2.COLOR_RGB2GRAY) > self.white
        excluded = (y_patch == -1)
        y_patch[np.logical_and(white_mask, excluded)] = 0
        
        return y_patch
    
    #transform y_patch to 2 classes (0, 1) using label dictionaries
    def transform_y_patch_twoClasses(self, y_patch):
        # Create a copy of y_patch (to avoid overwriting)
        y_patch_transformed = np.copy(y_patch)
        
        # transform positive class labels
        for label in self.positive_class_labels:
            y_patch_transformed = np.where(y_patch == label, 1, y_patch_transformed)
        
        # tarnsform negative class labels and excluded labels to 0
        for label in list(self.negative_class_labels) + list(self.excluded_labels):
            y_patch_transformed = np.where(y_patch == label, 0, y_patch_transformed)
        
        return y_patch_transformed

    # Get y_patch with 2 classes (0, 1)
    def get_2class_y_patch(self, x: int = 0, y: int = 0):
        y_patch = self.get_y_patch(x, y)
        y_patch = self.transform_y_patch_twoClasses(y_patch)
        return y_patch

    def __str__(self):
        return str(self.file)
