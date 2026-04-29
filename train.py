from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.models as models

# =========================
# 1. DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. LOAD MODEL (RESNET)
# =========================
model = models.resnet50(pretrained=True)

# hapus classifier (fc layer)
model = nn.Sequential(*list(model.children())[:-1])

model = model.to(device)
model.eval()

# =========================
# 3. PREPROCESSING
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
# 4. LOAD DATASET
# =========================
dataset = datasets.ImageFolder(
    root="dataset_clean",
    transform=transform
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

# =========================
# 5. FORWARD PASS (EMBEDDING)
# =========================
if __name__ == "__main__":
    images, labels = next(iter(loader))

    images = images.to(device)

    with torch.no_grad():
        features = model(images)

    # sebelum flatten
    print("Input shape:", images.shape)
    print("Feature shape:", features.shape)

    # flatten embedding
    features = features.view(features.size(0), -1)

    print("Flattened:", features.shape)
    
# simpan embedding
torch.save(features.cpu(), "image_embeddings.pt")

# simpan label
torch.save(labels, "labels.pt")

print("Embedding berhasil disimpan!")