import os
from pathlib import Path
from typing import Optional
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
data_dir = Path("datasets/waste_split")
batch_size = 32

input_size = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

val_transform = transforms.Compose([
    transforms.Resize(int(input_size * 1.14)), 
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
])
def get_dataloaders(data_dir: Path, batch_size: int, num_workers: Optional[int] = None):
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 0)
        num_workers = min(4, os.cpu_count() or 0)

    train_ds = ImageFolder(data_dir / "train", transform=train_transform)
    val_ds = ImageFolder(data_dir / "val", transform=val_transform)
    test_ds = ImageFolder(data_dir / "test", transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


if __name__ == '__main__':
    import time
    from collections import Counter
    import torch.nn as nn
    import torch.optim as optim
    from pathlib import Path

    epochs = 15
    lr = 1e-4
    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create datasets and dataloaders here (safe for multiprocessing spawn)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(train_ds.classes)
    print(f"Number of classes: {num_classes}")

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes, input_size=224):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # /2

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # /4

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # /8
            )

            reduced = input_size // 8
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(128 * reduced * reduced, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    model = SimpleCNN(num_classes=num_classes, input_size=input_size).to(device)

    counts = Counter(train_ds.targets)
    class_counts = [counts[i] for i in range(num_classes)]
    class_weights = torch.tensor([1.0 / c if c > 0 else 0.0 for c in class_counts], dtype=torch.float)
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    def accuracy_from_outputs(outputs, labels):
        preds = outputs.argmax(dim=1)
        correct = (preds == labels).sum().item()
        return correct


    def train_one_epoch(model, loader, optimizer, criterion, device):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_corrects += accuracy_from_outputs(outputs, labels)
            total += images.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        return epoch_loss, epoch_acc


    def eval_model(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                running_corrects += accuracy_from_outputs(outputs, labels)
                total += images.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        return epoch_loss, epoch_acc

    best_val_acc = 0.0
    best_path = checkpoint_dir / "best_model.pth"
    train_history = []
    val_history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)

        train_history.append((train_loss, train_acc))
        val_history.append((val_loss, val_acc))
        scheduler.step(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{epochs} - {elapsed:.1f}s - train_loss: {train_loss:.4f} acc: {train_acc:.4f} | val_loss: {val_loss:.4f} acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': train_ds.classes
            }, best_path)
            print(f"  Saved new best model (val_acc={val_acc:.4f}) -> {best_path}")

    print(f"Training complete in {(time.time() - start_time)/60:.2f} minutes. Best val acc: {best_val_acc:.4f}")

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']} with val_acc={checkpoint['val_acc']:.4f}")

    test_loss, test_acc = eval_model(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    import json
    meta = {
        'epochs': epochs,
        'best_val_acc': best_val_acc,
        'test_loss': float(test_loss),
        'test_acc': float(test_acc)
    }
    with open(checkpoint_dir / 'train_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print('Done.')
