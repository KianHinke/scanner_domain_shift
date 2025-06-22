# Scanner-Induced Domain Shift Experiments

This repository contains the main code used for the preprocessing, training, evaluation, and domain shift analysis of the experiments.
The following explains the purpose of the individual scripts and points to the respective docs that explain how to use them in combination with config files
---

## Workflow Overview

### 1. Preprocessing
Loads the WSIs to SlideContainers, creates dataset splits and serializes each split set to PyTorch .pt files.  
- [`preprocessing/preprocess_slides.py`](preprocessing/preprocess_slides.py)  
  [Documentation](docs/preprocess_slides.md)  
  [Example config](configs/config_preprocess_slides.json)

### 2. Feature Extraction
Extract latent features from preprocessed slides using foundation models (e.g., DINO).  
- [`feature_extraction/DINO_feature_extraction.py`](feature_extraction/DINO_feature_extraction.py)  
  [Config](configs/config_extract.json)

### 3. Classifier Training

#### Input Space
Train and test classifiers directly on image patches on the original tumor task.  
- [`classifier_training/model_train_test.py`](classifier_training/model_train_test.py)  
  [Documentation](docs/model_train_test.md)  
  [Train config](configs/config_model_train.json)  
  [Test config](configs/config_model_test.json)

#### Latent Space
Train and test classifiers on extracted feature representations on the original tumor task.  
- [`classifier_training/model_train_test_latent.py`](classifier_training/model_train_test_latent.py)  
  [Documentation](docs/model_train_test_latent.md)  
  [Train config](configs/config_model_train_latent.json)  
  [Test config](configs/config_model_test_latent.json)

#### All-scanner Classifier
Train models on concatenated multi-scanner data on the original tumor task.  
- [`domain _discrimination_PADstar/PADstar_training.py`](domain _discrimination_PADstar/PADstar_training.py)  
  [Documentation](docs/PADstar_training.md)  
  [Config](configs/config_PADstar_training.json)

### 4. PAD* (All-scanner Training & Domain Discrimination)

#### Domain Discriminator
Train a neural network to predict scanner origin (domain discrimination task) from extracted features.  
- [`domain _discrimination_PADstar/PADstar_discriminator.py`](domain _discrimination_PADstar/PADstar_discriminator.py)  
  [Documentation](docs/PADstar_discriminator.md)  
  [Config](configs/config_PADstar_discriminator.json)

#### Domain Discriminator Inference
Evaluate all trained PAD* domain discriminator models on a concatenated multi-scanner test set.
- [`domain _discrimination_PADstar/PADstar_inference.py`](domain _discrimination_PADstar/PADstar_inference.py)  
  [Documentation](docs/PADstar_inference.md)  
  [Config](configs/config_PADstar_inference.json)

### 5. Shift Detection

#### BBSD (KS Test) Input Space
Measure distributional shifts using softmax outputs of an input-space classifier.  
- [`shift_detection_BBSD_MMD/BBSD_input.py`](shift_detection_BBSD_MMD/BBSD_input.py)  
  [Documentation](docs/BBSD_input.md)  
  [ResNet18 config](configs/config_BBSD_input_Resnet18.json)  
  [TwoLayerNN config](configs/config_BBSD_input_TwoLayerNN.json)

#### BBSD & MMD (Latent)
Measure shifts using softmax outputs of a classifier trained on latent features (BBSD) or directly on latent features (MMD).  
- [`shift_detection_BBSD_MMD/BBSD_MMD_latent.py`](shift_detection_BBSD_MMD/BBSD_MMD_latent.py)  
  [Documentation](docs/BBSD_MMD_latent.md)  
  [BBSD Dinov2 config](configs/config_BBSD_latent_dinov2.json)  
  [BBSD DINO config](configs/config_BBSD_latent_dino.json)  
  [MMD Dinov2 config](configs/config_MMD_latent_dinov2.json)  
  [MMD DINO config](configs/config_MMD_latent_dino.json)



---

**For detailed instructions and config examples, see the linked documentation and config files above.**