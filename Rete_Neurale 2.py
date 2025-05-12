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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report
from sklearn.manifold import TSNE
import umap

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
def standard_preprocessing():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def augmentation_preprocessing():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def fuzzy_preprocessing():
    def apply_fuzzy_cmeans(tensor_img):
        np_img = tensor_img.permute(1, 2, 0).numpy()
        gray = np.mean(np_img, axis=2)
        X = gray.flatten()
        X = np.expand_dims(X, axis=0)
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X, c=3, m=2.0, error=0.005, maxiter=1000, init=None)
        cluster_map = np.argmax(u, axis=0).reshape(gray.shape)
        cluster_map = (cluster_map - cluster_map.min()) / (cluster_map.ptp() + 1e-6)
        cluster_tensor = torch.tensor(cluster_map, dtype=torch.float32)
        return cluster_tensor.unsqueeze(0).repeat(3, 1, 1)

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(apply_fuzzy_cmeans),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def clahe_preprocessing():
    def apply_clahe(tensor_img):
        np_img = tensor_img.permute(1, 2, 0).numpy()
        np_img = (np_img * 255).astype(np.uint8)
        gray_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray_img)
        clahe_img = clahe_img.astype(np.float32) / 255.0
        return torch.tensor(clahe_img).unsqueeze(0).repeat(3, 1, 1)

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(apply_clahe),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def wavelet_preprocessing():
    def apply_wavelet(pil_img):
        img_np = np.array(pil_img.convert('L'))
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

def retinex_sobel_preprocessing():
    def apply_retinex_sobel(pil_img):
        img_np = np.array(pil_img.convert('RGB'))
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Retinex algorithm
        retinex = np.log1p(img_gray) - np.log1p(cv2.GaussianBlur(img_gray, (5, 5), 0))
        
        # Sobel edge detection
        sobel_edges = cv2.Sobel(retinex, cv2.CV_64F, 1, 1, ksize=3)
        sobel_edges = cv2.convertScaleAbs(sobel_edges)
        
        # Normalize and return as tensor
        sobel_edges = sobel_edges.astype(np.float32) / 255.0
        return torch.tensor(sobel_edges).unsqueeze(0).repeat(3, 1, 1)

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(apply_retinex_sobel),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

# === SCELTA PREPROCESSAMENTO ===
use_standard = False
use_augmentation = False
use_fuzzy = False
use_clahe = False
use_wavelet = False
use_retinex_sobel = True

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
elif use_retinex_sobel:
    transform = retinex_sobel_preprocessing()
else:
    raise ValueError("Nessun preprocessamento selezionato!")

# === DATASET E DATALOADER ===
class CustomImageDataset(Dataset):
    def _init_(self, df, transform=None):
        self.df = df
        self.transform = transform

    def _len_(self):
        return len(self.df)

    def _getitem_(self, idx):
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

# === DIVISIONE DEL DATASET ===
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])
val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df['label'])

train_dataset = CustomImageDataset(train_df, transform=transform)
val_dataset = CustomImageDataset(val_df, transform=transform)
test_dataset = CustomImageDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# === CARICAMENTO DEL MODELLO ===
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 5)  # Modifica l'output per 5 classi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# === FUNZIONE PER L'ADDIESTRAMENTO ===
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    
    for epoch in range(num_epochs):
        model.train()
        total_train, correct_train, running_train_loss = 0, 0, 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss.append(running_train_loss / len(train_loader))
        train_acc.append(100 * correct_train / total_train)

        # Validation
        model.eval()
        total_val, correct_val, running_val_loss = 0, 0, 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss.append(running_val_loss / len(val_loader))
        val_acc.append(100 * correct_val / total_val)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss[-1]:.4f}, Train Accuracy: {train_acc[-1]:.2f}% | Val Loss: {val_loss[-1]:.4f}, Val Accuracy: {val_acc[-1]:.2f}%")
    
    return train_loss, train_acc, val_loss, val_acc

# === VALUTAZIONE ===
def evaluate_model(model, test_loader):
    model.eval()
    total, correct = 0, 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    plt.plot(recall, precision)
    plt.title("Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig("precision_recall_curve.png")
    
    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=label_map.keys())
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    
    # TSNE and UMAP
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(outputs.cpu().numpy())
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels)
    plt.title("TSNE")
    plt.savefig("tsne.png")

    umap_model = umap.UMAP()
    umap_results = umap_model.fit_transform(outputs.cpu().numpy())
    plt.scatter(umap_results[:, 0], umap_results[:, 1], c=all_labels)
    plt.title("UMAP")
    plt.savefig("umap.png")
    
    return accuracy

# === ESECUZIONE ===
train_loss, train_acc, val_loss, val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
test_accuracy = evaluate_model(model, test_loader)

# === GRAFICI FINALIZZATI ===
# Accuracy durante l'addestramento e validazione
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_acc) + 1), train_acc, label="Train Accuracy")
plt.plot(range(1, len(val_acc) + 1), val_acc, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy during Training and Validation")
plt.savefig("accuracy_during_training.png")

# Loss durante l'addestramento e validazione
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_loss) + 1), train_loss, label="Train Loss")
plt.plot(range(1, len(val_loss) + 1), val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss during Training and Validation")
plt.savefig("loss_during_training.png")