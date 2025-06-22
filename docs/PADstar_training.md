
# PADstar Training

This script (PADstar_training.py) trains and evaluates classification models (e.g., ResNet18, TwoLayerNN) on concatenated multi-scanner slide data for the PAD* experiment.

---

## Usage

```sh
python PADstar_training.py --config config_PADstar_training.json
```

---

## Config File

**Example:**

```json
{
    "models": ["ResNet18"],
    "scanners": ["cs2", "nz20", "nz210", "p1000", "gt450"],
    "dataprep_folder": "data/dataprep",
    "annotation_file": "data/scc.json",
    "lrs": [0.0001],
    "epochs": [8],
    "modelsavepath": "sc_domain_shift/allscanner_models"
}
```

**Config fields:**
- models: List of model names to train (e.g., `"ResNet18"`, `"TwoLayerNN"`).
- `scanners`: List of scanner names to include.
- `dataprep_folder`: Folder containing preprocessed slide data (`*_train.pt`, etc.).
- `annotation_file`: Path to the annotation JSON file.
- `lrs`: List of learning rates to try.
- `epochs`: List of epoch counts to try.
- `modelsavepath`: Directory to save trained model files.

---

## Notes

- The script will train and evaluate all combinations of models, learning rates, and epochs specified in the config.
- All paths should be adapted to your environment and data locations.
- Results, including confusion matrices and F1 scores, are printed to the console.  
- Trained models are saved to the specified `modelsavepath`.

---

**Edit the config as needed and run the script as shown above.**