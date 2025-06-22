```markdown
# BBSD_input.py

This script runs the BBSD (Black Box Shift Detection) pipeline for input space classifiers. It evaluates distributional differences between scanners by extracting softmax outputs from pretrained models and performing pairwise Kolmogorov-Smirnov (KS) tests, visualizing the results as heatmaps.

---

## What it Does

- Loads preprocessed test data for each scanner.
- Loads pretrained models (e.g., ResNet18, TwoLayerNN) for each scanner.
- Extracts softmax outputs for all test slides across all scanners.
- Performs pairwise KS tests (with bootstrapping and Bonferroni correction) between scanners.
- Saves and visualizes the results as heatmaps.

---

## Usage

```sh
python BBSD_input.py --config config_BBSD_input_Resnet18.json
```

---

## Config File

The script requires a JSON config file specifying all paths and parameters.  
**Example:**

```json
{
    "modelpaths": {
        "cs2": "sc_domain_shift/models/cs2_ResNet18_epoch_8_lr_0.0001.pth",
        "nz20": "sc_domain_shift/models/nz20_ResNet18_epoch_8_lr_0.0001.pth",
        "nz210": "sc_domain_shift/models/nz210_ResNet18_epoch_8_lr_0.0001.pth",
        "p1000": "sc_domain_shift/models/p1000_ResNet18_epoch_8_lr_0.0001.pth",
        "gt450": "sc_domain_shift/models/gt450_ResNet18_epoch_8_lr_0.0001.pth"
    },
    "scanners": ["cs2", "nz20", "nz210", "p1000", "gt450"],
    "savepath": "KS_heatmap_figures_Resnet18",
    "dataprep_folder": "data/dataprep",
    "annotation_file": "data/scc.json",
    "n_bootstraps": 100
}
```

**Config fields:**
- `modelpaths`: Dictionary mapping each scanner to its pretrained model path.
- `scanners`: List of scanners to include in the analysis.
- `savepath`: Directory to save the generated heatmap figures.
- `dataprep_folder`: Folder containing preprocessed test data (`*_test.pt` files).
- `annotation_file`: Path to the annotation JSON file.
- `n_bootstraps`: Number of bootstrap iterations for the KS test.

---

**Edit the config as needed and run the script as shown above.**
```