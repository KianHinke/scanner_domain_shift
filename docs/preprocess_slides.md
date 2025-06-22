## Slide Preprocessing Script (preprocess_slides.py)

This script preprocesses whole slide images (WSIs) for downstream machine learning tasks.  
It reads raw slide images and their annotations, splits them into training, validation, and test sets, and saves the processed data for each scanner.

### **What it does**

- Loads annotation and split configuration files.
- Iterates over a list of scanners.
- For each scanner:
  - Loads slide metadata and labels.
  - Splits slides into training, validation, and test sets.
  - Serializes the split data into PyTorch `.pt` files in a specified output folder.

### **Inputs (via config file)**

- `dataprep_folder`: Output directory for preprocessed data.
- `annotation_file`: Path to the annotation JSON file.
- `splitconfig_path`: Path to the split configuration file.
- `images_path`: Directory containing the raw `.tif` slide images.

### **Outputs**

For each scanner, the script saves:
- `{scanner}_train.pt`
- `{scanner}_valid.pt`
- `{scanner}_test.pt`

These files contain serialized slide metadata for each split, ready for use in model training and evaluation.

### **Usage Example**

```sh
python preprocess_slides.py --config path/to/preprocess_config.json
```

**Edit the config file to match your data locations before running the script.**