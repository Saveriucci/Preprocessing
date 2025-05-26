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
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from sklearn.preprocessing import StandardScaler

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

def apply_fuzzy_cmeans(tensor_img):
    np_img = tensor_img.permute(1, 2, 0).numpy()
    gray = np.mean(np_img, axis=2)
    X = gray.flatten()
    X = np.expand_dims(X, axis=0)
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X, c=3, m=2.0, error=0.005, maxiter=1000, init=None)
    cluster_map = np.argmax(u, axis=0).reshape(gray.shape)
    cluster_map = (cluster_map - cluster_map.min()) / (np.ptp(cluster_map) + 1e-6)
    cluster_tensor = torch.tensor(cluster_map, dtype=torch.float32)
    return cluster_tensor.unsqueeze(0).repeat(3, 1, 1)

def fuzzy_preprocessing():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(apply_fuzzy_cmeans),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def apply_clahe(tensor_img):
    np_img = tensor_img.permute(1, 2, 0).numpy()
    np_img = (np_img * 255).astype(np.uint8)
    gray_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_img)
    clahe_img = clahe_img.astype(np.float32) / 255.0
    return torch.tensor(clahe_img).unsqueeze(0).repeat(3, 1, 1)

def clahe_preprocessing():
        return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(apply_clahe),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
def apply_wavelet(pil_img):
    img_np = np.array(pil_img.convert('L'), dtype=np.float32)
    coeffs2 = pywt.dwt2(img_np, 'haar')
    cA, _ = coeffs2

    # 🔧 Normalizzazione dei coefficienti
    cA = (cA - np.min(cA)) / (np.max(cA) - np.min(cA) + 1e-6)

    cA_img = Image.fromarray((cA * 255).astype(np.uint8)).resize((224, 224))
    tensor = TF.to_tensor(cA_img).expand(3, -1, -1)
    return tensor

def wavelet_preprocessing():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(apply_wavelet),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
def apply_retinex_sobel(pil_img):
    img_np = np.array(pil_img.convert('RGB'))
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Retinex (log-based normalization)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    blurred[blurred == 0] = 1e-6  # prevenire log(0)
    retinex = np.log1p(img_gray) - np.log1p(blurred)

    # Sobel
    sobel_edges = cv2.Sobel(retinex, cv2.CV_64F, 1, 1, ksize=3)
    sobel_edges = np.abs(sobel_edges)
    sobel_edges = (sobel_edges - sobel_edges.min()) / (sobel_edges.max() - sobel_edges.min() + 1e-6)

    return torch.tensor(sobel_edges, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)

def retinex_sobel_preprocessing():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(apply_retinex_sobel),
        transforms.Normalize([0.5]*3, [0.5]*3)  # ← Commenta per evitare immagine scura
    ])

def apply_homomorphic_highpass(pil_img):
    img_np = np.array(pil_img.convert('L'), dtype=np.float32) / 255.0
    
    # --- Homomorphic Filter ---
    # Log-transform
    img_log = np.log1p(img_np)
    
    # FFT
    img_fft = np.fft.fft2(img_log)
    img_fft_shift = np.fft.fftshift(img_fft)
    
    rows, cols = img_np.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create Gaussian high pass filter mask
    D0 = 30  # cutoff frequency, puoi regolare
    H = np.ones((rows, cols))
    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - crow)**2 + (v - ccol)**2)
            H[u, v] = 1 - np.exp(-(D**2) / (2 * (D0**2)))
    
    # Apply filter in frequency domain
    img_fft_filt = img_fft_shift * H
    
    # Inverse FFT
    img_ifft_shift = np.fft.ifftshift(img_fft_filt)
    img_filtered_log = np.fft.ifft2(img_ifft_shift)
    img_filtered_log = np.real(img_filtered_log)
    
    # Exp to invert log transform and normalize
    img_filtered = np.expm1(img_filtered_log)
    img_filtered = np.clip(img_filtered, 0, 1)
    
    # --- High Pass Filter (Sobel) ---
    sobel_x = cv2.Sobel(img_filtered, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_filtered, cv2.CV_64F, 0, 1, ksize=3)
    highpass = np.sqrt(sobel_x**2 + sobel_y**2)
    highpass = (highpass - highpass.min()) / (np.ptp(highpass) + 1e-6)
    
    # Convert to tensor and repeat to 3 channels
    tensor_img = torch.tensor(highpass, dtype=torch.float32).unsqueeze(0).repeat(3,1,1)
    
    return tensor_img

def homomorphic_highpass_preprocessing():
        return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(apply_homomorphic_highpass),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])




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
            print(f"Errore caricando {path}: {e}")
            return torch.zeros(3, 224, 224), 0


# Disabilita ReLU inplace in tutto il modello
def disable_inplace_relu(module):
    if isinstance(module, nn.ReLU):
        module.inplace = False


def train_model(device, model, train_loader, val_loader, criterion, optimizer, num_epochs, preprocessing_name=""):
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    best_epoch = 0

    # Crea cartella per i grafici
    os.makedirs("plots", exist_ok=True)
    #early_stopping = EarlyStopping(patience=3, min_delta=0.01)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total, correct = 0, 0
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {accuracy:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1  # +1 perché le epoche partono da 0
            torch.save(model.state_dict(), f"best_model_{preprocessing_name}.pth") 
        # Early stopping check
        #if early_stopping(avg_val_loss):
         #   print(f"Early stopping triggered at epoch {epoch+1}")
          #  break

    # Grafici (usano val_losses ora)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy (%)', marker='o')
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'Training Loss and Validation Accuracy over Epochs ({preprocessing_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/training_metrics_comparison_{preprocessing_name}.png")
    plt.clf()

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, color='red', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Over Epochs ({preprocessing_name})')
    plt.grid(True)
    plt.savefig(f"plots/final_training_loss_{preprocessing_name}.png")
    plt.clf()

    plt.figure()
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, color='blue', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title(f'Validation Accuracy Over Epochs ({preprocessing_name})')
    plt.grid(True)
    plt.savefig(f"plots/final_validation_accuracy_{preprocessing_name}.png")
    plt.clf()
    return best_epoch


def evaluate_and_explain(label_map, device, model, test_loader, preprocessing_name=""):
    os.makedirs("plots", exist_ok=True)

    model.eval()
    all_labels, all_preds = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_loss = total_loss / len(test_loader)
    accuracy = 100 * sum(int(p == t) for p, t in zip(all_preds, all_labels)) / len(all_labels)
    report = classification_report(all_labels, all_preds, target_names=label_map.keys(), zero_division=0)
    # Costruisci il path del file
    filename = f"plots/classification_report_{preprocessing_name}.txt"
    # Scrivi il report nel file
    with open(filename, "w") as f:
        f.write(report)
        print()
    cm = confusion_matrix(all_labels, all_preds)

    # === PRECISION-RECALL CURVE ===
    model.eval()
    probs = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs.append(F.softmax(outputs, dim=1).cpu().numpy())
    probs = np.vstack(probs)
    true = np.array(all_labels)

    n_classes = probs.shape[1]
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(true == i, probs[:, i])
        plt.plot(recall, precision, lw=2, label=f"Classe {i} ({list(label_map.keys())[i]})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve per Classe ({preprocessing_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/precision_recall_curve_{preprocessing_name}.png")
    plt.clf()

    # === CONFUSION MATRIX ===
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.title(f"Confusion Matrix ({preprocessing_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"plots/confusion_matrix_{preprocessing_name}.png")
    plt.clf()

    # === GRAFICO COMPARATIVO LOSS + ACCURACY ===
    plt.figure(figsize=(6, 4))
    metrics = ['Accuracy (%)', 'Loss']
    values = [accuracy, test_loss]
    colors = ['skyblue', 'salmon']
    plt.bar(metrics, values, color=colors)
    plt.title(f"Test Accuracy and Loss ({preprocessing_name})")
    for i, v in enumerate(values):
        plt.text(i, v + 1 if i == 0 else v + 0.01, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(f"plots/test_metrics_comparison_{preprocessing_name}.png")
    plt.clf()

    # Un batch di immagini dal test_loader
    sample_img, sample_label = next(iter(test_loader))
    img = sample_img[0].to(device)    
    
    denorm_img = denorm(img)

    for class_idx in range(5):
        image_tensor = img.to(device)# Assicurati che sia Tensor già preprocessato
        cam, pred_class = generate_gradcam(device, model, image_tensor, class_idx=class_idx, target_layer='layer4')
        plt.subplot(1, 5, class_idx + 1)
        plt.imshow(denorm_img)
        plt.imshow(cam, cmap='jet', alpha=0.5)
        plt.title(f'Classe {class_idx}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"plots/gradcam_classes_{preprocessing_name}.png", bbox_inches='tight')  # salva una volta
    plt.show()
    plt.clf()
    


def generate_gradcam(device, model, image_tensor, class_idx=None, target_layer='layer4'):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Hook per feature map + gradienti
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    layer = dict([*model.named_modules()])[target_layer]
    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_backward_hook(backward_hook)

    # Forward
    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item() if class_idx is None else class_idx

    # Backward
    model.zero_grad()
    class_loss = output[0, pred_class]
    class_loss.backward()

    # Rimozione hook
    forward_handle.remove()
    backward_handle.remove()

    grad = gradients[0].detach().cpu().numpy()[0]
    fmap = features[0].detach().cpu().numpy()[0]

    # Ponderazione: media globale dei gradienti
    weights = np.mean(grad, axis=(1, 2))
    cam = np.sum(weights[:, np.newaxis, np.newaxis] * fmap, axis=0)
    cam = np.maximum(cam, 0)  # ReLU
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    return cam, pred_class

def denorm(img):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
    img = img * std + mean
    return img.permute(1, 2, 0).cpu().numpy()


def visualize_tsne(device, model, test_loader, preprocessing_name="default"):
    os.makedirs("plots", exist_ok=True)
    model.eval()
    
    all_features = []
    all_labels = []

    # Hook: usiamo l'output dell'ultimo layer prima della classificazione
    def hook_fn(module, input, output):
        all_features.append(output.detach().cpu())

    handle = model.avgpool.register_forward_hook(hook_fn)  # per ResNet-like, puoi adattare se hai un altro modello

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            all_labels.extend(labels.numpy())

    handle.remove()  # rimuove il hook

    features = torch.cat(all_features, dim=0).view(len(all_labels), -1)
    labels = np.array(all_labels)

    # Riduzione dimensionalità con t-SNE
    features_std = StandardScaler().fit_transform(features)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_std)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=labels, palette="tab10", legend="full", alpha=0.7)
    plt.title(f"t-SNE Visualization - {preprocessing_name}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Classe", loc='best')
    plt.tight_layout()
    plt.savefig(f"plots/tsne_visualization_{preprocessing_name}.png")
    plt.clf()
 

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        patience: numero di epoche da attendere senza miglioramento
        min_delta: miglioramento minimo considerato significativo
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


# === MAIN === #
def main():
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
    
    preprocessamenti ={
        "Standar": standard_preprocessing(),
        "Augmentation": augmentation_preprocessing(),
        "CLAHE": clahe_preprocessing(),
        "Wavelet": wavelet_preprocessing(),
        "RetinexSoobel": retinex_sobel_preprocessing(),
        "HomomorphicHghPass": homomorphic_highpass_preprocessing(),
        "Fuzzy": fuzzy_preprocessing()
        }
    
    # === SPLIT === #
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df['label'], random_state=42)
    
    for preprocessing_name, funzione in preprocessamenti.items():
        print(f"\n=== INIZIO {preprocessing_name.upper()} ===")
        transform = funzione
        
        train_dataset = CustomImageDataset(train_df, transform)
        val_dataset = CustomImageDataset(val_df, transform)
        test_dataset = CustomImageDataset(test_df, transform)
        
    
        train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers = 4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=48, shuffle=False, num_workers = 4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False, num_workers = 4, pin_memory=True)
        
        # === MODELLO === #
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 5)
        model.apply(disable_inplace_relu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        best_epoch = train_model(device, model, train_loader, val_loader, criterion, optimizer, num_epochs=30, preprocessing_name=preprocessing_name)
        #print(f"Training fermato all'epoca: {best_epoch}")
        evaluate_and_explain(label_map, device, model, test_loader, preprocessing_name=preprocessing_name)
        visualize_tsne(device, model, test_loader, preprocessing_name)

if __name__ == "__main__":
    main()
