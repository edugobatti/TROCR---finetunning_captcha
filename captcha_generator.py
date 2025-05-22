import os
import random
import string
from captcha.image import ImageCaptcha
from PIL import Image
import cv2
import numpy as np
from skimage.morphology import remove_small_objects
from tqdm import tqdm

# Diretório de saída
CAPTCHA_DIR = "captchas"
os.makedirs(CAPTCHA_DIR, exist_ok=True)

def gerar_texto_aleatorio():
    tamanho = random.randint(3, 6)
    caracteres = string.ascii_letters + string.digits
    return ''.join(random.choices(caracteres, k=tamanho))

def aplicar_filtros(imagem_pil: Image.Image) -> Image.Image:
    # Converte PIL para NumPy (grayscale)
    img_np = np.array(imagem_pil.convert('L'))

    # Binarização
    _, thresh = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morfologia para remover ruídos pequenos
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Remove objetos pequenos
    bool_img = opening.astype(bool)
    cleaned = remove_small_objects(bool_img, min_size=50, connectivity=2)

    # Converte de volta
    cleaned_img = (cleaned * 255).astype(np.uint8)

    # Inverte a imagem (texto preto, fundo branco)
    final = cv2.bitwise_not(cleaned_img)

    # Converte para PIL antes de salvar
    return Image.fromarray(final)

def gerar_captchas(qtd: int):
    image_captcha = ImageCaptcha(width=200, height=80)

    for _ in tqdm(range(qtd)):
        texto = gerar_texto_aleatorio()
        imagem = image_captcha.generate_image(texto)

        imagem_filtrada = aplicar_filtros(imagem)
        caminho_arquivo = os.path.join(CAPTCHA_DIR, f"{texto}.png")
        imagem_filtrada.save(caminho_arquivo)

# Exemplo de uso:
gerar_captchas(50)
