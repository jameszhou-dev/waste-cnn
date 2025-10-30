# waste-cnn

Project to train a convolutional neural network (CNN) to classify waste images.

This repository contains:

- `process_data.ipynb` — notebook to download/prepare the dataset and create stratified train/val/test splits.
- `cnn.ipynb` — notebook version of the training flow (data loading, model, training loop).
- `cnn.py` — runnable training script (safe for multiprocessing; use `python cnn.py`).
- `models/` — checkpoint directory (best model saved to `models/best_model.pth`).
- `datasets/` — original and split datasets. After running the split you'll have `datasets/waste_class_split/{train,val,test}/{class}`.

1. Install requirements:

```bash
pip install torch torchvision Pillow numpy
```

2. Prepare / split the dataset

- Open `process_data.ipynb` and run the last cell which creates a stratified split into `datasets/waste_class_split/{train,val,test}`. You can choose to copy files or create symlinks.
- Alternatively adapt the `get_dataloaders` helper in `cnn.py` to point at your existing split root (the script currently points to `datasets/waste_split`).

3. Run training (script)

```bash
python cnn.py
```

Or run interactively in the notebook `cnn.ipynb` (run all cells up to and including the training cell).

## Files & important notes

- `cnn.py` contains a small from-scratch `SimpleCNN` model and a full training loop. The script saves the best checkpoint to `models/best_model.pth` and writes `models/train_meta.json` with training metadata.
- DataLoader creation and the training loop are wrapped in `if __name__ == '__main__':` to avoid a multiprocessing spawn error on macOS / Python 3.8+ (see Troubleshooting below).
- Default transforms use ImageNet normalization; if you train from scratch you may compute dataset mean/std and replace those values.

## Hyperparameters you may want to change

- `batch_size` (in `cnn.ipynb` or `cnn.py`)
- `input_size` (default 224)
- `epochs`, `lr` (learning rate)
- `num_workers` (DataLoader worker processes) — set to `0` if you encounter multiprocessing problems or for debugging

## Troubleshooting

- RuntimeError about starting new process before bootstrapping: ensure you run scripts (not import them) and that heavy runtime code is under `if __name__ == '__main__':`. `cnn.py` already follows this pattern.
- If you see worker-related errors, set `num_workers=0` in `get_dataloaders()` or notebook DataLoader creation to disable multiprocessing for data loading.
- Out of Memory (OOM) on GPU: reduce `batch_size` or enable mixed precision training (AMP) to save memory.
