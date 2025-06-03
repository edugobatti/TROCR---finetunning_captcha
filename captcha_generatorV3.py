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

def adicionar_linhas_horizontais(imagem_pil):
    """Adiciona linhas horizontais diretamente na imagem"""
    draw = ImageDraw.Draw(imagem_pil)
    width, height = imagem_pil.size
    
    # Número aleatório de linhas entre 1 e 3
    num_linhas = random.randint(1, 3)
    
    for _ in range(num_linhas):
        # Posição Y aleatória
        y = random.randint(5, height - 5)
        
        # Cor da linha (cinza)
        cor = random.randint(100, 200)
        
        # Desenha linha horizontal
        draw.line([(0, y), (width, y)], fill=cor, width=1)
    
    return imagem_pil

def gerar_captchas(qtd: int):
    # Usa ImageCaptcha padrão
    image_captcha = ImageCaptcha(width=200, height=80)

    for _ in tqdm(range(qtd)):
        texto = gerar_texto_aleatorio()
        
        # Gera imagem base
        imagem = image_captcha.generate_image(texto)
        
        # Adiciona linhas horizontais
        imagem_com_linhas = adicionar_linhas_horizontais(imagem)
        
        # Aplica filtros
        imagem_filtrada = aplicar_filtros(imagem_com_linhas)
        
        # Salva
        caminho_arquivo = os.path.join(CAPTCHA_DIR, f"{texto}.png")
        imagem_filtrada.save(caminho_arquivo)

# Para debug - gera alguns exemplos sem filtro
def gerar_exemplos_debug(qtd=10):
    print("Gerando exemplos de debug...")
    debug_dir = "captchas_debug"
    os.makedirs(debug_dir, exist_ok=True)
    
    image_captcha = ImageCaptcha(width=200, height=80)
    
    for i in range(qtd):
        texto = gerar_texto_aleatorio()
        imagem = image_captcha.generate_image(texto)
        
        # Salva sem linhas
        imagem.save(os.path.join(debug_dir, f"{texto}_sem_linhas.png"))
        
        # Adiciona linhas e salva
        imagem_com_linhas = adicionar_linhas_horizontais(imagem.copy())
        imagem_com_linhas.save(os.path.join(debug_dir, f"{texto}_com_linhas.png"))


# Gera os 30000 CAPTCHAs
gerar_captchas(3)