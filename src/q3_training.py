import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
df_train= pd.read_csv("data/processed/train_labels.csv")
df_test= pd.read_csv("data/processed/test_labels.csv")

transforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])
class LandCoverDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df= pd.read_csv(csv_file)
        self.img_dir= img_dir
        self.transform= transform
    def __len__(self):
        return len(self.df)
    mapping={
            "Built-up": 0,
            "Cropland": 1,
            "Water": 2,
            "Vegetation": 3,
            "Other": 4
        }
    def __getitem__(self, idx):
        filename = self.df.iloc[idx]["filename"]
        label = self.df.iloc[idx]["category"]
        label = self.mapping[label]
        path= f"{self.img_dir}/{filename}"
        image = Image.open(path).convert("RGB")
        if self.transform:
            image= self.transform(image)
        return image, label
train_dataset= LandCoverDataset("data/processed/train_labels.csv", "data/raw/sentinel_patches", transform=transforms)
test_dataset= LandCoverDataset("data/processed/test_labels.csv", "data/raw/sentinel_patches", transform=transforms)
train_loader= torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader= torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

#Cnn Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv_layers= nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding= 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers= nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16*16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x= self.conv_layers(x)
        x= self.fc_layers(x)
        return x
    def get_criterion(self):
        return nn.CrossEntropyLoss()
    def get_optimizer(self, lr=0.001):
        return torch.optim.Adam(self.parameters(), lr=lr)
    
model = SimpleCNN(num_classes=5)
criterion = model.get_criterion()
optimizer = model.get_optimizer(lr=0.001)


for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch :", epoch, "Loss:", loss.item())
torch.save(model.state_dict(), "model.pth")

# evaluate

model.eval()
with torch.no_grad():
    all_preds = []
    all_labels = []
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    y_true = all_labels
    y_pred = all_preds

accuracy = accuracy_score(y_true, y_pred)
print("Test Accuracy:", accuracy)
f1 = f1_score(y_true, y_pred, average="macro")
print("Test F1 Score:", f1)

# Confusion Matrix
cm= confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=["Built-up", "Cropland", "Water", "Vegetation", "Other"], yticklabels=["Built-up", "Cropland", "Water", "Vegetation", "Other"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

