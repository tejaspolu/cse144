import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
from PIL import Image
from tqdm import tqdm
import timm  # timm provides many state-of-the-art vision models

# -----------------------------
# 1. Reproducibility
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# -----------------------------
# 2. Configuration
# -----------------------------
BATCH_SIZE = 8         # smaller batch for finer gradient updates
EPOCHS = 10            # full 20 epochs (no early stopping)
LR = 1e-4              # low LR for fine-tuning
IMG_SIZE = 224

TRAIN_DIR = "train"   # Adjust if nested differently
TEST_DIR = "test"       # Adjust if nested differently

SUBMISSION_FILE = "/home/jupyter/cse144/submission.csv"
BEST_MODEL_PATH = "/home/jupyter/cse144/best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 3. Data Transforms
# -----------------------------
# We add AutoAugment for extra augmentation
train_transforms = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),  # AutoAugment
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),               # random crop
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.25))
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# 4. Prepare Training & Validation Sets
# -----------------------------
full_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
# Use validation transforms for val set
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# -----------------------------
# 5. Custom Test Dataset
# -----------------------------
class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # Collect only .jpg files and sort by numeric prefix
        self.image_files = sorted(
            [f for f in os.listdir(root) if f.endswith('.jpg')],
            key=lambda x: int(x.split('.')[0])
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]   # e.g., "7.jpg"
        img_path = os.path.join(self.root, filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Return the full filename to match sample submission format
        return image, filename

test_dataset = TestDataset(TEST_DIR, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

if len(test_dataset) == 0:
    print(f"[Warning] No .jpg files found in '{TEST_DIR}'. Submission.csv will be empty!")

# -----------------------------
# 6. Build Model (Swin Transformer)
# -----------------------------
def build_model():
    # Create a Swin Transformer model using timm.
    # Here we use 'swin_base_patch4_window7_224', which is a powerful architecture.
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=100)
    
    # Partial freezing: Freeze all layers except the last two stages and the head.
    # For timm's swin models, layers are in model.layers (a ModuleList).
    # We'll allow training for layers 2 and 3 and the classification head.
    for name, param in model.named_parameters():
        if ("head" in name) or ("layers.2" in name) or ("layers.3" in name):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model.to(device)

# -----------------------------
# 7. Train/Val/Predict Functions
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def validate_one_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def predict_test(model, loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, filenames in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for fn, pred in zip(filenames, preds):
                predictions.append((fn, pred.item()))
    return predictions

# -----------------------------
# 8. Main Training Loop (No Early Stopping)
# -----------------------------
if __name__ == "__main__":
    model = build_model()
    # Use label smoothing (if using PyTorch >=1.10)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Only optimize parameters that are not frozen
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)
        print(f"[Epoch {epoch}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> New best val acc! Model saved. (acc={val_acc:.2f}%)")

    # Load best model for test predictions
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        print(f"Best model loaded with val acc = {best_val_acc:.2f}%")

    predictions = predict_test(model, test_loader)

    # Create submission DataFrame: must match sample format (e.g., "0.jpg",Label)
    df_preds = pd.DataFrame(predictions, columns=["ID", "Label"])
    # Sort by the numeric portion of the filename to match sample submission order
    df_preds["ID_int"] = df_preds["ID"].apply(lambda x: int(x.split('.')[0]))
    df_preds.sort_values("ID_int", inplace=True)
    df_preds.drop(columns=["ID_int"], inplace=True)

    df_preds.to_csv(SUBMISSION_FILE, index=False)
    print(f"\nSubmission saved to {SUBMISSION_FILE} (rows={len(df_preds)})")
    print(f"Best val accuracy achieved during training: {best_val_acc:.2f}%")
