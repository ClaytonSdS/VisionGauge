import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Arch import NewDirectModel_Inference as NDM

# ================= CONFIG =================
pd.options.display.max_columns = None
pd.options.display.max_colwidth = None
pd.options.display.width = 0

# ======================================================
# FUNÇÃO: Zero-padding para tornar a imagem quadrada
# ======================================================
def pad_to_square_center(img):
    h, w = img.shape[:2]
    size = max(h, w)

    padded = np.zeros((size, size, 3), dtype=img.dtype)
    y_off = (size - h) // 2
    x_off = (size - w) // 2

    padded[y_off:y_off + h, x_off:x_off + w] = img
    return padded


# ======================================================
# LISTA DE IMAGENS
# ======================================================
paths = [
    r"dataset\distance\fig13.jpg",

]

print("Carregando regressor...")

Regressor = NDM("resnet").load_model(r"C:\Users\Clayton\Desktop\MODELS\ResNet-18_120x120.pth")

print("Modelo carregado.\n")


# ======================================================
# LOOP NAS IMAGENS
# ======================================================
for path in paths:

    print(f"\nProcessando: {path}")

    img = cv2.imread(path)

    if img is None:
        print("Erro ao ler imagem.")
        continue

    img_original = img.copy()

    # ================= PREPARAÇÃO =================
    img_square = pad_to_square_center(img)

    # Modelo espera lista de imagens
    pred = Regressor.predict([img_square])[0]
    hp = float(pred)

    # ================= CORREÇÃO DE PARALAXE =================
    n_air = 1.000293
    n_m = 1.53
    delta = 0.1   # espessura do acrílico (cm)
    d = 26        # distância objeto-lente (cm)
    M = 150

    alpha_ang = np.arctan((M - hp) / d)
    gama = np.arcsin(n_air / n_m * np.sin(alpha_ang))
    delta_h = delta * np.tan(gama)
    ht = hp - delta_h

    print(f"hp: {hp:.2f} | ht: {ht:.2f}")

    # ================= DESENHO NA IMAGEM =================
    label = f"hp:{hp:.1f} | ht:{ht:.1f}"

    # ================= MOSTRAR =================
    img_show = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.imshow(img_show)
    plt.title(label)
    plt.axis("off")
    plt.show()  # fecha janela → próxima imagem

print("\nFinalizado.")
