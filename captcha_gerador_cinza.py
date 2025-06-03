import os
import random
import string
from captcha.image import ImageCaptcha

def gerar_captchas(pasta_saida: str, quantidade: int, largura=220, altura=80):
    os.makedirs(pasta_saida, exist_ok=True)

    # Inicializa gerador de CAPTCHAs
    image_captcha = ImageCaptcha(width=largura, height=altura)

    caracteres = string.ascii_lowercase + string.digits

    for _ in range(quantidade):
        comprimento = random.randint(3, 6)  # de 3 a 6 dígitos
        texto = ''.join(random.choices(caracteres, k=comprimento))

        imagem = image_captcha.generate_image(texto)
        imagem = imagem.convert("L")  # escala de cinza

        caminho = os.path.join(pasta_saida, f"{texto}.png")
        imagem.save(caminho)

    print(f"{quantidade} CAPTCHAs salvos em: {pasta_saida}")




gerar_captchas("captchas", 30000)
