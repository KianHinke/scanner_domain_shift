
## Training and Testing Latent Space Task Classifiers

model_train_test_latent.py is used to **train** and/or **test** classifiers on extracted feature (latent) representations.

### **Usage**

The workflow and config structure are analogous to the input space script.  
**See the section above** for general usage, training/testing examples, and config conventions.

### **Key Config Fields for Latent Space**

- `arraypath`: Path to the directory containing extracted feature arrays (`*_features.npy` and `*_labels.npy`).
- `results_file`: File to save result statistics.
- For training:  
  - models, `scanners`, `lrs`, `epochs`, `modelsavepath`
- For testing:  
  - `modelpaths` (list of model files to test)  
  - or `models_folder_path` (directory with models to test)

### **Example: Training**

```sh
python model_train_test_latent.py --config config_model_train_latent.json
```

### **Example: Testing**

```sh
python model_train_test_latent.py --config config_model_test_latent.json
```

### **Notes**

- You can specify multiple models, learning rates, and epochs in the config to automatically train and evaluate all combinations for each scanner.
- For testing, use either `modelpaths` or `models_folder_path` as described above.

**Edit the config files to match your feature data and desired settings.**

