from ultralytics import YOLO
import cv2
import numpy as np
import torch
import pandas as pd
from Arch import NewDirectModel_Inference as NDM
import matplotlib.pyplot as plt

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
    r"dataset\validation\raw\IMG_2.jpeg",
    r"dataset\validation\raw\IMG_2_alterada.jpg",
    r"dataset\validation\raw\IMG_2_alterada_prox.jpg",
]

# ---------------- Inicialização dos modelos ----------------
print("Carregando modelos...")

Segmentation = YOLO("models/SegARC_v08/weights/best.pt")
Regressor =  NDM("resnet").load_model(r"C:\Users\Clayton\Desktop\MODELS\resnet\L2_H0\ResNet-18_120x120_RETRAINED_2.pth")

print("Modelos carregados.\n")


# ======================================================
# LOOP NAS IMAGENS
# ======================================================
for path in paths:

    print(f"\nProcessando: {path}")

    frame = cv2.imread(path)
    if frame is None:
        print("Erro ao ler imagem.")
        continue

    img_full = frame.copy()
    img_rgb = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)

    # ================= YOLO Prediction =================
    out = Segmentation.predict(img_rgb, conf=0.3, verbose=False)[0]
    boxes = out.boxes.xyxy.cpu().numpy().astype(int)

    # ================= CROP + PAD =================
    images_raw = []
    valid_boxes = []

    for xmin, ymin, xmax, ymax in boxes:
        if xmax <= xmin or ymax <= ymin:
            continue

        crop = img_full[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            continue

        crop_square = pad_to_square_center(crop)

        images_raw.append(crop_square)
        valid_boxes.append((xmin, ymin, xmax, ymax))

    # ================= PREDIÇÕES =================
    if len(images_raw) == 0:
        print("Nenhuma detecção válida.")
        continue

    preds_r1 = Regressor.predict(images_raw)
    hp = preds_r1

    # ================= CORREÇÃO DE PARALAXE =================
    n_air = 1.000293
    n_m = 1.53
    delta = 0.1   # espessura do acrílico (cm)
    d = 26        # distância objeto-lente (cm)
    M = 150       # referência

    alpha_ang = np.arctan((M - hp) / d)
    gama = np.arcsin(n_air / n_m * np.sin(alpha_ang))
    delta_h = delta * np.tan(gama)
    ht = hp - delta_h

    # ================= DESENHO =================
    for k, (xmin, ymin, xmax, ymax) in enumerate(valid_boxes):
        bw = xmax - xmin
        bh = ymax - ymin

        hp_k = float(hp[k])
        ht_k = float(ht[k])

        label = f"hp:{hp_k:.1f} | ht:{ht_k:.1f} | {bw}x{bh}"
        print(label)

        cv2.rectangle(img_full, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            img_full,
            (xmin, ymin - text_h - 10),
            (xmin + text_w, ymin),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            img_full,
            label,
            (xmin, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # ================= MOSTRAR COM MATPLOTLIB =================
    img_show = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.imshow(img_show)
    plt.title(path)
    plt.axis("off")
    plt.show()   # <- BLOQUEIA ATÉ FECHAR A JANELA

print("\nFinalizado.")
