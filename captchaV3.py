from captcha.image import ImageCaptcha
from PIL import Image, ImageDraw, ImageOps
import random
import string
import os
from tqdm import tqdm

def adicionar_ruido(imagem: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(imagem)

    # Linhas pretas finas
    for _ in range(random.randint(5, 10)):
        x1 = random.randint(0, imagem.width)
        y1 = random.randint(0, imagem.height)
        x2 = random.randint(0, imagem.width)
        y2 = random.randint(0, imagem.height)
        draw.line((x1, y1, x2, y2), fill=0, width=1)

    # Pontos pretos
    for _ in range(200):
        x = random.randint(0, imagem.width)
        y = random.randint(0, imagem.height)
        draw.point((x, y), fill=0)

    return imagem

def gerar_captchas_com_letras_pretas(qtd_imagens: int, pasta_destino: str = "captchas_teste"):
    os.makedirs(pasta_destino, exist_ok=True)
    image = ImageCaptcha(width=280, height=90)

    for _ in tqdm(range(qtd_imagens)):
        texto = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(4, 6)))
        data = image.generate(texto)
        img = Image.open(data).convert("L")  # mantém tons de cinza (letra escura)
        img = ImageOps.autocontrast(img)     # realça o preto
        img = adicionar_ruido(img)

        caminho_arquivo = os.path.join(pasta_destino, f"{texto}.png")
        img.save(caminho_arquivo)


gerar_captchas_com_letras_pretas(5)