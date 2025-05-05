import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import pandas as pd
import numpy as np
import pywt  # wavelet
import skfuzzy as fuzz 
import cv2


# 1. Carica il CSV
names = ['image_path', 'organ', 'label']
df = pd.read_csv(r"..\Progetto\dataset.csv", names=names, header=0)

# 2. Mappa le etichette
label_map = {
    "l.adenocarcinoma": 0,
    "l.benign": 1,
    "l.scadenocarcinoma": 2,
    "c.adenocarcinoma": 3,
    "c.benign": 4
}
df["label"] = df["label"].map(label_map)

# === PREPROCESSAMENTI ===
#1. standard
#2. agumented
#3. fuzzy (fuzzy_cmeans)
#4. clahe
#5. wavelet
#6. wavelet augumented 

def standard_preprocessing():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def augmentation_preprocessing():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def fuzzy_preprocessing():
    def apply_fuzzy_cmeans(tensor_img):
        # Converti da torch tensor [C, H, W] a NumPy [H, W] H=altezza W=larghezza
        np_img = tensor_img.permute(1, 2, 0).numpy()
        gray = np.mean(np_img, axis=2)  # Converti in scala di grigi [H, W]

        # Flatten immagine
        X = gray.flatten()
        X = np.expand_dims(X, axis=0)  # [1, N]

        # Applica Fuzzy C-Means con 3 cluster che tenderà a segmentare l'immagine in modo che venga data "importanza" allo sfondo, tessuto sano e tessuto maligno( area sospetta/ lesione)
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            X, c=3, m=2.0, error=0.005, maxiter=1000, init=None
        )

        # Trova il cluster con max appartenenza per ogni pixel
        #heatmap implicita
        cluster_map = np.argmax(u, axis=0).reshape(gray.shape)

        # Normalizza mappa [0, 1] e convertila a tensor
        #Questa operazione ti assicura che i valori di appartenenza fuzzy vadano da 0 (appartenenza nulla) a 1 (appartenenza massima). In pratica stai solo rimappando la tua heatmap fuzzy su un intervallo standardizzato, in modo che diventi un’immagine valida (pixel floating point tra 0 e 1).
        cluster_map = (cluster_map - cluster_map.min()) / (cluster_map.ptp() + 1e-6)
        cluster_tensor = torch.tensor(cluster_map, dtype=torch.float32)

        # Ridimensiona in [1, H, W] per una singola canale
        return cluster_tensor.unsqueeze(0).repeat(3, 1, 1)  # Fake 3 canali

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(apply_fuzzy_cmeans),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]) #Questo secondo step NON riguarda la heatmap in sé, ma tutte le immagini (inclusa la tua heatmap “fittizia” su 3 canali) e serve a centrare e scalare i valori secondo la media e la deviazione standard di ImageNet. È fondamentale se usi un modello pre-addestrato (ResNet, VGG, ecc.), perché quei modelli si aspettano input normalizzati in quel modo.
    ])

def clahe_preprocessing():
    def apply_clahe(tensor_img):
        # Converti da [C, H, W] a NumPy [H, W, C]
        np_img = tensor_img.permute(1, 2, 0).numpy()
        
        # Assicurati che i valori siano tra 0 e 255
        np_img = (np_img * 255).astype(np.uint8)

        # Converti in scala di grigi
        gray_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        # Applica CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray_img)

        # Normalizza il risultato in [0, 1] e converti in tensore torch [C, H, W]
        clahe_img = clahe_img.astype(np.float32) / 255.0
        return torch.tensor(clahe_img).unsqueeze(0).repeat(3, 1, 1)

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(apply_clahe),
        transforms.Normalize([0.485, 0.456, 0.406],  # Media ImageNet
                             [0.229, 0.224, 0.225])  # Deviazione standard ImageNet
    ])


def wavelet_preprocessing():
    def apply_wavelet(pil_img):
        img_np = np.array(pil_img.convert('L'))  # Grayscale
        coeffs2 = pywt.dwt2(img_np, 'haar')
        cA, _ = coeffs2
        cA_img = Image.fromarray(cA).resize((224, 224))
        tensor = TF.to_tensor(cA_img).expand(3, -1, -1)
        return tensor
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(apply_wavelet),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

def wavelet_augmented_preprocessing():
    def apply_wavelet(pil_img):
        img_np = np.array(pil_img.convert('L'))
        coeffs2 = pywt.dwt2(img_np, 'haar')
        cA, _ = coeffs2
        cA_img = Image.fromarray(cA).resize((224, 224))
        return cA_img

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Lambda(apply_wavelet),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

# === SCELTA PREPROCESSAMENTO ===
use_standard = True
use_augmentation = False
use_fuzzy = False
use_clahe = False
use_wavelet = False
use_wavelet_augmented = False  # Attiva questa per wavelet + augmentation

if use_standard:
    transform = standard_preprocessing()
elif use_augmentation:
    transform = augmentation_preprocessing()
elif use_fuzzy:
    transform = fuzzy_preprocessing()
elif use_clahe:
    transform = clahe_preprocessing()
elif use_wavelet:
    transform = wavelet_preprocessing()
elif use_wavelet_augmented:
    transform = wavelet_augmented_preprocessing()
else:
    raise ValueError("Nessun preprocessamento selezionato!")

# === DATASET E DATALOADER ===
class CustomImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['label']
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Errore immagine {path}: {e}")
            return torch.zeros(3, 224, 224), 0

dataset = CustomImageDataset(df, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# === MODELLO ===
class CustomCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def load_resnet18(num_classes=5):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

use_custom_cnn = False
model = CustomCNN() if use_custom_cnn else load_resnet18()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# === TRAINING ===
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(dataloader):.4f}, Accuracy: {100*correct/total:.2f}%")

# === VALUTAZIONE ===
def evaluate_model(model, dataloader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# === ESECUZIONE ===
train_model(model, dataloader, criterion, optimizer, num_epochs=10)
evaluate_model(model, dataloader)
