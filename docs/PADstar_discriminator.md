

# PADstar Domain Discriminator

This script (`PADstar_discriminator.py`) extracts features from preprocessed slides using a pretrained or out-of-the-box task-classifier (ResNet18 or DINO), assigns scanner labels, and trains a simple neural network (TwoLayerNN) to predict the scanner from these features.

---

## Usage

```sh
python PADstar_discriminator.py --config config_PADstar_discriminator.json
```

---

## Config File

**Example:**

```json
{
    "annotation_file": "data/scc.json",
    "dataprep_folder": "data/dataprep",
    "modelpath": "sc_domain_shift/allscanner_models/PAD_ResNet18_epoch_8_lr_0.0001.pth",
    "modelsavepath": "sc_domain_shift/PADdiscriminators",
    "save_loaders_path": "data/concatenated_loaders_PAD_ResNet18_finetuned",
    "input_size": 512,
    "modelname" : "ResNet18"
}
```

- `input_size`: Use `512` for ResNet18, `768` for DINO.
- `modelname`: Use either `ResNet18` or `Dino` depending on the classifier's architecture.



**Edit the config as needed and run the script as shown above.**