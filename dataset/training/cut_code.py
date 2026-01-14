import os
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import tqdm 
import os
import re

folder_path = "dataset/training/hopper"
save_path = "dataset/training/cropped"
initial_number = 6745

my_current_dir = os.getcwd()
folder_path = os.path.join(my_current_dir, folder_path)
save_path = os.path.join(my_current_dir, save_path)


# Caminho da pasta
folder_path_resized = r'dataset\training\resized'
pattern = re.compile(r'image_resized_(\d+)') # Expressão regular para capturar o número do nome do arquivo
initial_number = 0  # valor padrão caso não encontre nada

for filename in os.listdir(folder_path_resized):
    match = pattern.search(filename)
    if match:
        number = int(match.group(1))
        if number > initial_number:
            initial_number = number

print(f"Maior número encontrado: {initial_number}")
initial_number += 1

target_shape = (2000, 2000) # altura, largura

def crop_bottom_center(
    img,
    crop_h=1500,
    crop_w=1500,
    horizontal_shift=0,   # negativo = esquerda, positivo = direita
    start_row=None        # None = começa no fundo (default)
):
    height, width, _ = img.shape

    # centro horizontal da imagem
    cx = width // 2

    # aplica o deslocamento horizontal
    cx = cx + horizontal_shift

    # limita para não sair da imagem
    cx = max(crop_w // 2, min(width - crop_w // 2, cx))

    # linha inicial (se None, usa o fundo padrão)
    if start_row is None:
        start_row = height - crop_h

    # garante que não passe dos limites
    start_row = max(0, min(height - crop_h, start_row))


    end_row = start_row + crop_h

    start_col = cx - crop_w // 2
    end_col = cx + crop_w // 2

    return img[start_row:end_row, start_col:end_col]


count = 1
start_at = 247

def adjust_temperature(img, i=0):
    # img vem em RGB — converter para BGR para manipular
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    b, g, r = cv2.split(bgr)

    # intensidade > 0 = esfria
    # intensidade < 0 = aquece
    b = cv2.add(b,  i)      # mais azul
    r = cv2.subtract(r, i)  # menos vermelho

    out = cv2.merge((b, g, r))
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    
    return out


for filename in tqdm.tqdm(os.listdir(folder_path)):

    img = cv2.imread(os.path.join(folder_path, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img= cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
<<<<<<< HEAD

    if count <=130:
        recorte = crop_bottom_center(
            # temp=2, contrast = 1.4, brightness=5
            adjust_image(img, temp=0, brightness=20), # [i>0 -> -Temp] | [i<0 -> +Temp]

            crop_h=2000,
            crop_w=2000,
            start_row=530,
            horizontal_shift=-100
        )

    if count > 130:
        recorte = crop_bottom_center(
            # temp=2, contrast = 1.4, brightness=5
            adjust_image(img, temp=0, brightness=20), # [i>0 -> -Temp] | [i<0 -> +Temp]

            crop_h=2000,
            crop_w=2000,
            start_row=530,
            horizontal_shift=100
        )



=======
    if count < 2000:
        recorte = crop_bottom_center(
            adjust_temperature(img, i=2), # [i>0 -> -Temp] | [i<0 -> +Temp]

            crop_h=2500,
            crop_w=2500,
            start_row=500,
            horizontal_shift=-100
        )

>>>>>>> parent of a934b52c (dataset: update images for training and testing)

    recorte = cv2.resize(recorte, (1000, 1000))

    plt.imsave(os.path.join(save_path, f"image_{initial_number}.png"), recorte)

    count += 1
    initial_number += 1

print("Finalizado o recorte das imagens.")