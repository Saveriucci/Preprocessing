import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import pywt  # wavelet
import skfuzzy as fuzz 
import cv2
import matplotlib.pyplot as plt
import os

# === PREPROCESSAMENTI ===
def standard_preprocessing():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Ridimensiona a input compatibile con modelli pre-addestrati
        transforms.Grayscale(num_output_channels=1),  # Converte in scala di grigi
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalizzazione semplice per immagini grayscale
    ])

def augmentation_preprocessing():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # Piccole rotazioni per aumentare la varietà dei dati
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
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

# === CLAHE (migliora il contrasto localmente) ===
def clahe_preprocessing():
    def apply_clahe(pil_img):
        img_np = np.array(pil_img.convert('L'))  # Converti a scala di grigi
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_np)
        normalized = enhanced.astype(np.float32) / 255.0  # Scala 0-1
        return torch.tensor(normalized).unsqueeze(0)  # [1, H, W]

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(apply_clahe)
    ])

def wavelet_preprocessing():
    def apply_wavelet(pil_img):
        img_np = np.array(pil_img.convert('L'), dtype=np.float32)
        coeffs2 = pywt.dwt2(img_np, 'haar')
        cA, _ = coeffs2

        # 🔧 Normalizzazione dei coefficienti
        cA = (cA - np.min(cA)) / (np.max(cA) - np.min(cA) + 1e-6)

        cA_img = Image.fromarray((cA * 255).astype(np.uint8)).resize((224, 224))
        tensor = TF.to_tensor(cA_img).expand(3, -1, -1)
        return tensor

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(apply_wavelet),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

# === Retinex + Sobel (normalizzazione illuminazione + edge detection) ===
def retinex_sobel_preprocessing():
    def apply_retinex_sobel(pil_img):
        img_np = np.array(pil_img.convert('L')).astype(np.float32)
        blurred = cv2.GaussianBlur(img_np, (5, 5), 0)
        blurred[blurred == 0] = 1e-6  # Previene log(0)
        retinex = np.log1p(img_np) - np.log1p(blurred)  # Retinex

        # Applica filtro di Sobel
        sobel = cv2.Sobel(retinex, cv2.CV_64F, 1, 1, ksize=3)
        sobel = np.abs(sobel)
        sobel = (sobel - sobel.min()) / (sobel.max() - sobel.min() + 1e-6)

        return torch.tensor(sobel, dtype=torch.float32).unsqueeze(0)

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(apply_retinex_sobel)
    ])

# === Filtro Omomorfico + Alto Passo (accentua dettagli + rimuove luci non uniformi) ===
def homomorphic_highpass_preprocessing():
    def apply_homomorphic(pil_img):
        img_np = np.array(pil_img.convert('L'), dtype=np.float32) / 255.0

        # Filtro omomorfico: log, FFT, filtro, iFFT
        img_log = np.log1p(img_np)
        img_fft = np.fft.fft2(img_log)
        img_fft_shift = np.fft.fftshift(img_fft)

        rows, cols = img_np.shape
        crow, ccol = rows // 2, cols // 2
        D0 = 30  # Frequenza di taglio

        # Maschera high-pass gaussiana
        H = np.ones((rows, cols))
        for u in range(rows):
            for v in range(cols):
                D = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
                H[u, v] = 1 - np.exp(-(D ** 2) / (2 * (D0 ** 2)))

        img_fft_filt = img_fft_shift * H
        img_ifft = np.fft.ifft2(np.fft.ifftshift(img_fft_filt))
        img_filtered = np.real(np.expm1(img_ifft))
        img_filtered = np.clip(img_filtered, 0, 1)

        # Filtro Sobel per evidenziare i bordi
        sobel_x = cv2.Sobel(img_filtered, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_filtered, cv2.CV_64F, 0, 1, ksize=3)
        highpass = np.sqrt(sobel_x**2 + sobel_y**2)
        highpass = (highpass * 255).astype(np.uint8)

        # Migliora contrasto con CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        highpass_clahe = clahe.apply(highpass).astype(np.float32) / 255.0

        return torch.tensor(highpass_clahe).unsqueeze(0)

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(apply_homomorphic)
    ])


img_path = r"C:\Users\tomas\Desktop\universita\Magistrale\Primo Anno\Secondo Semestre\Artificial Intelligence From Engineer To Arts\Progetto\Preprocessamento_immagine\lung\lungaca65.jpeg"


save_dir = r"C:\Users\tomas\Desktop\universita\Magistrale\Primo Anno\Secondo Semestre\Artificial Intelligence From Engineer To Arts\Progetto\Preprocessamento_immagine\lung"
os.makedirs(save_dir, exist_ok=True)

# --- Carica immagine
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError("Errore nel caricamento dell'immagine")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)

# === Preprocessamenti disponibili ===
preprocessamenti = {
    "standard": standard_preprocessing(),
    "augmentation": augmentation_preprocessing(),
    "fuzzy": fuzzy_preprocessing(),
    "clahe": clahe_preprocessing(),
    "wavelet": wavelet_preprocessing(),
    "retinex_sobel": retinex_sobel_preprocessing(),
    "homomorphic_highpass": homomorphic_highpass_preprocessing()
}

# Applica ogni preprocessamento
for name, transform in preprocessamenti.items():
    img_tensor = transform(img_pil)

    # Converti da torch tensor a numpy [H, W, C]
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    # 🔧 Controlla il numero di canali
    if img_np.shape[2] == 1:  # Grayscale: [H, W, 1]
        img_np = np.repeat(img_np, 3, axis=2)  # Converti in RGB duplicando i canali

   # Salva immagine
    save_path = os.path.join(save_dir, f"{name}.png")
    plt.imsave(save_path, img_np)
    print(f"Salvata: {save_path}")

    # Mostra immagine (opzionale)
    plt.figure()
    plt.imshow(img_np)
    plt.title(f"{name} preprocessing")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
