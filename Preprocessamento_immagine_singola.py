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

def retinex_sobel_preprocessing():
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

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(apply_retinex_sobel),
        transforms.Normalize([0.5]*3, [0.5]*3)  # ← Commenta per evitare immagine scura
    ])



img_path = r"C:\Users\tomas\Desktop\universita\Magistrale\Primo Anno\Secondo Semestre\Artificial Intelligence From Engineer To Arts\Progetto\Preprocessamento_immagine\lungaca1.jpeg"
img = cv2.imread(img_path)
if img is None:
       print("Errore nel caricamento dell'immagine")
else:
    #cv2.imshow('Image', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    print(" ")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)
transform = wavelet_preprocessing()  # Puoi usare anche standard_preprocessing(), wavelet_preprocessing(), etc.
img_tensor = transform(img_pil)
print(img_tensor)
# 4. Visualizza il tensore preprocessato (in formato numpy)
img_np = img_tensor.permute(1, 2, 0).numpy()  # CxHxW → HxWxC
img_np = np.clip(img_np, 0, 1)  # Se normalizzato, porta i valori tra 0 e 1 per visualizzazione

plt.imshow(img_np)
plt.title("wavelet preprocessing")
plt.axis('off')
plt.show()