
## Training and Testing Input Space Task Classifiers

model_train_test.py can be used to **train** and/or **test** input space task classifiers on your data.

### **Example: Training**

```sh
python model_train_test.py --config config_model_train.json
```

### **Example: Testing**

```sh
python model_train_test.py --config config_model_test.json
```

### **Config File Fields**

- `annotation_file`: Path to the .json annotation file.
- `data_folder`: Folder containing preprocessed data as pytorch tensor (.pt).
- `results_file`: .txt file to save result stats.

additionally:

For training: models, `scanners`, `lrs`, `epochs`, `modelsavepath`

For testing, you can specify either:

- `modelpaths`: a list of exact model file paths to test individually.
- `models_folder_path`: a folder containing multiple model files; the script will automatically test all models in that folder.

Use `modelpaths` for selected models, or `models_folder_path` to test all models in a directory.

### Training with Multiple Models, Learning Rates, and Epochs

When using model_train_test.py for training, you can specify **multiple models**, **learning rates**, and **epochs** in your config file. The script will automatically train and evaluate **all combinations** of these settings for each scanner you list.

#### **How it works:**

- In your config (e.g., `config_model_train.json`), you can list several models, learning rates, and epochs:
  ```json
  {
      "models": ["ResNet18", "TwoLayerNN"],
      "lrs": [0.001, 0.0001],
      "epochs": [8, 16],
      "scanners": ["cs2", "nz20", "nz210",...],
      ...
  }
  ```

This makes it easy to compare different models and hyperparameters without running the script multiple times.




**Edit the config files to match your data and paths.**


