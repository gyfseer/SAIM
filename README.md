# Model Detection and Training System

This project provides a client-server system for image model detection and online incremental learning.

---

## 1. Environment Setup

Create the virtual environment using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
conda activate your_env_name
```

> Replace `your_env_name` with the actual environment name defined in `environment.yaml`.

---

## 2. Model Detection (Detection Mode)

### Steps:

1. Run the client program:

    ```bash
    python setup-client.py
    ```

2. Run the server program:

    ```bash
    python setup-server.py
    ```

3. In the client interface, switch the mode to `DM` (Detection Mode).

    - The system will automatically detect all images in the `datasets/val` directory.
    - Detection results will be saved to the `results/SAIM/images` directory.

---

## 3. Model Training (Learning Mode)

### Steps:

1. Place the new training samples into the `datasets/insert` directory and prepare annotation files in COCO format.

2. Run the client program:

    ```bash
    python setup-client.py
    ```

3. Run the server program:

    ```bash
    python setup-server.py
    ```

4. In the client interface, switch the mode to `LM` (Learning Mode).

    - The system will automatically begin training using the new data.

---

## Mode Definitions

- `DM`: Detection Mode — performs model inference on validation images.
- `LM`: Learning Mode — performs model training using inserted samples.

---

## Example Directory Structure

```bash
datasets/
├── val/               # Images used for detection
├── insert/            # New images and annotations for training
results/
└── SAIM/
    └── images/        # Detection output results
```

---

## Contact

For questions, please contact the project maintainer.

---
