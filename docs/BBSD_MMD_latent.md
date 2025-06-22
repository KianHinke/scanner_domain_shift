```markdown
# BBSD_MMD_latent.py

This script runs Black Box Shift Detection (BBSD) and Maximum Mean Discrepancy (MMD) tests for **latent space experiments**.  
It evaluates distributional differences between scanners using either softmax outputs from trained latent-space classifiers (BBSD) or directly on extracted latent features (MMD), and visualizes the results as heatmaps.

---

## What it Does

- Loads extracted latent features and labels for each scanner.
- Loads pretrained latent-space classifiers for each scanner.
- **BBSD:** Extracts softmax outputs from each model and performs pairwise Kolmogorov-Smirnov (KS) tests (with bootstrapping and Bonferroni correction) between scanners.
- **MMD:** Performs pairwise MMD permutation tests directly on the latent features.
- Saves and visualizes the results as heatmaps.

---

## Usage

```sh
python BBSD_MMD_latent.py --config config_BBSD_latent_dinov2.json
```
or
```sh
python BBSD_MMD_latent.py --config config_MMD_latent_dinov2.json
```

---

## Config File

The script requires a JSON config file specifying all paths and parameters.

**Example for BBSD (KS test):**
```json
{
    "modelpaths": {
        "cs2": "sc_domain_shift/models_latent_dinov2/cs2_TwoLayerNN_epoch_12_lr_0.0001_512.pth",
        "nz20": "sc_domain_shift/models_latent_dinov2/nz20_TwoLayerNN_epoch_12_lr_0.0001_512.pth",
        "nz210": "sc_domain_shift/models_latent_dinov2/nz210_TwoLayerNN_epoch_12_lr_0.0001_512.pth",
        "p1000": "sc_domain_shift/models_latent_dinov2/p1000_TwoLayerNN_epoch_12_lr_0.0001_512.pth",
        "gt450": "sc_domain_shift/models_latent_dinov2/gt450_TwoLayerNN_epoch_12_lr_0.0001_512.pth"
    },
    "savepath": "KS_heatmap_figures_latent_dinov2_BONFERRONI",
    "arraypath": "data/extracted_features_dinov2",
    "scanners": ["cs2", "nz20", "nz210", "p1000", "gt450"],
    "bootstrap": true,
    "n_bootstraps": 100
}
```

**Example for MMD:**
```json
{
    "modelpaths": {
        "cs2": "sc_domain_shift/models_latent_dinov2/cs2_TwoLayerNN_epoch_12_lr_0.0001_512.pth",
        "nz20": "sc_domain_shift/models_latent_dinov2/nz20_TwoLayerNN_epoch_12_lr_0.0001_512.pth",
        "nz210": "sc_domain_shift/models_latent_dinov2/nz210_TwoLayerNN_epoch_12_lr_0.0001_512.pth",
        "p1000": "sc_domain_shift/models_latent_dinov2/p1000_TwoLayerNN_epoch_12_lr_0.0001_512.pth",
        "gt450": "sc_domain_shift/models_latent_dinov2/gt450_TwoLayerNN_epoch_12_lr_0.0001_512.pth"
    },
    "savepath": "MMD_heatmap_figures_latent_dinov2",
    "arraypath": "data/extracted_features_dinov2",
    "scanners": ["cs2", "nz20", "nz210", "p1000", "gt450"],
    "MMD_perms": 1000
}
```

**Config fields:**
- `modelpaths`: Dictionary mapping each scanner to its trained latent classifier.
- `savepath`: Directory to save the generated heatmap figures.
- `arraypath`: Directory containing extracted latent features and labels.
- `scanners`: List of scanner names to include in the analysis.
- For BBSD:  
  - `bootstrap`: Set to `true` to enable bootstrapped KS testing.
  - `n_bootstraps`: Number of bootstrap iterations.
- For MMD:  
  - `MMD_perms`: Number of permutations for the MMD test.

---

**Edit the config as needed and run the script as shown above.**
```