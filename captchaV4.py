from captcha.image import ImageCaptcha
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import random
import string
import os
from tqdm import tqdm

def adicionar_ruido(imagem: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(imagem)

    largura, altura = imagem.size

    # Mais linhas pretas (finas e grossas)
    for _ in range(random.randint(10, 20)):
        x1 = random.randint(0, largura)
        y1 = random.randint(0, altura)
        x2 = random.randint(0, largura)
        y2 = random.randint(0, altura)
        width = random.choice([1, 2, 3])
        draw.line((x1, y1, x2, y2), fill=0, width=width)

    # Pontos pretos e brancos aleatórios
    for _ in range(800):
        x = random.randint(0, largura - 1)
        y = random.randint(0, altura - 1)
        color = random.choice([0, 255])  # preto ou branco
        draw.point((x, y), fill=color)

    # Formas geométricas aleatórias (círculos e retângulos)
    for _ in range(random.randint(5, 10)):
        x1 = random.randint(0, largura - 10)
        y1 = random.randint(0, altura - 10)
        x2 = x1 + random.randint(5, 20)
        y2 = y1 + random.randint(5, 20)
        if random.random() < 0.5:
            draw.ellipse([x1, y1, x2, y2], outline=0)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=0)

    # Adiciona granulado via modificação direta dos pixels
    pixels = imagem.load()
    for i in range(largura):
        for j in range(altura):
            noise = random.randint(-30, 30)
            novo_valor = max(0, min(255, pixels[i, j] + noise))
            pixels[i, j] = novo_valor

    # Filtro leve de desfoque para aumentar dificuldade
    imagem = imagem.filter(ImageFilter.GaussianBlur(radius=0.5))

    return imagem

def gerar_captchas_com_letras_pretas(qtd_imagens: int, pasta_destino: str = "trocr-camada-3"):
    os.makedirs(pasta_destino, exist_ok=True)
    image = ImageCaptcha(width=280, height=90)

    for _ in tqdm(range(qtd_imagens)):
        texto = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(4, 6)))
        data = image.generate(texto)
        img = Image.open(data).convert("L")
        img = ImageOps.autocontrast(img)
        img = adicionar_ruido(img)

        caminho_arquivo = os.path.join(pasta_destino, f"{texto}.png")
        img.save(caminho_arquivo)

gerar_captchas_com_letras_pretas(1004)
