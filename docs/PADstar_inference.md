
# PADstar Inference

This script (`PADstar_inference.py`) evaluates all trained PAD* domain discriminator models in a specified folder on a concatenated multi-scanner test set. It reports accuracy, class-wise metrics, F1 scores, and the PAD*_multi value.

---

## Usage

```sh
python PADstar_inference.py --config config_PADstar_inference.json
```

---

## Config File

**Example:**

```json
{
    "models_folder": "sc_domain_shift/PADstar_models_resnet",
    "datapath": "sc_domain_shift/concatenated_loaders_PAD_resnet",
    "input_size": 512
}
```

- `models_folder`: Directory containing the trained model `.pth` files.
- `datapath`: Directory with the concatenated dataloader files (`train_loader.pt`, `valid_loader.pt`, `test_loader.pt`).
- `input_size`: Feature vector size (e.g., `512` for ResNet-based features and `768` for Dino-based features) 

---

**Edit the config as needed and run the script as shown above.**
```