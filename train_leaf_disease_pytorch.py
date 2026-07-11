import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = 'valid'
MODEL_DIR = 'models'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(MODEL_DIR, exist_ok=True)

# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_and_save(labels, model_name):
    print(f'\nTraining model for: {model_name}')
    temp_dir = f'_temp_{model_name}'
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    for label in labels:
        src = os.path.join(DATA_DIR, label)
        dst = os.path.join(temp_dir, label)
        import shutil
        shutil.copytree(src, dst)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(temp_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    model = SimpleCNN(len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        train_acc = 100. * correct / total
        val_loss, val_correct, val_total = 0.0, 0, 0
        model.eval()
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += targets.size(0)
        val_acc = 100. * val_correct / val_total
        print(f'Epoch {epoch+1}/{EPOCHS} - Train Loss: {running_loss/total:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/val_total:.4f}, Val Acc: {val_acc:.2f}%')
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{model_name}_leaf_model.pt'))
    print(f'Model saved: {MODEL_DIR}/{model_name}_leaf_model.pt')
    import shutil
    shutil.rmtree(temp_dir)

all_labels = os.listdir(DATA_DIR)
potato_labels = [d for d in all_labels if d.lower().startswith('potato')]
tomato_labels = [d for d in all_labels if d.lower().startswith('tomato')]
if potato_labels:
    train_and_save(potato_labels, 'potato')
if tomato_labels:
    train_and_save(tomato_labels, 'tomato') 