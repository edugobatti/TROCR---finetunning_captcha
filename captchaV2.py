import os
import random
import string
from captcha.image import ImageCaptcha
from PIL import Image, ImageDraw
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
    img_np = np.array(imagem_pil.convert('L'))

    _, thresh = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    bool_img = opening.astype(bool)
    cleaned = remove_small_objects(bool_img, min_size=50, connectivity=2)

    cleaned_img = (cleaned * 255).astype(np.uint8)

    final = cv2.bitwise_not(cleaned_img)
    return Image.fromarray(final)

# Subclasse personalizada com apenas linhas como ruído
class LineNoiseCaptcha(ImageCaptcha):
    def __init__(self, width=200, height=80, line_count=5, **kwargs):
        super().__init__(width=width, height=height, **kwargs)
        self.line_count = line_count

    def _draw_noise_dots(self, draw, image, color):
        pass  # Desativa os pontos

    def _draw_noise_curve(self, draw, image, color):
        width, height = image.size
        for _ in range(self.line_count):
            x = random.randint(0, width)
            y1 = 0
            y2 = height
            draw.line((x, y1, x, y2), fill=color, width=1)

def gerar_captchas(qtd: int, line_count: int = 5):
    image_captcha = LineNoiseCaptcha(width=200, height=80, line_count=line_count)

    for _ in tqdm(range(qtd)):
        texto = gerar_texto_aleatorio()
        imagem = image_captcha.generate_image(texto)
        imagem_filtrada = aplicar_filtros(imagem)
        caminho_arquivo = os.path.join(CAPTCHA_DIR, f"{texto}.png")
        imagem_filtrada.save(caminho_arquivo)


gerar_captchas(50000,5)
