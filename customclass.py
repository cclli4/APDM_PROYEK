import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# =========================
# 1. DATASET CLASS
# =========================
class BreastCancerDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")

        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================
# 2. PREPROCESSING
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# 3. LOAD DATA
# =========================
dataset = BreastCancerDataset(
    csv_file='metadata_fixed.csv',
    root_dir='dataset_png',
    transform=transform
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)


# =========================
# 4. TEST RUN
# =========================
if __name__ == "__main__":
    images, labels = next(iter(loader))
    print("Shape:", images.shape)
    print("Labels:", labels[:5])