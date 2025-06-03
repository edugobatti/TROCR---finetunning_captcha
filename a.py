from captcha.image import ImageCaptcha
from PIL import Image, ImageDraw, ImageFilter
import random
import string
import os

# Gera texto aleatório com 3 a 8 caracteres
def generate_random_text():
    length = random.randint(3, 8)
    characters = string.ascii_letters + string.digits  # a-zA-Z0-9
    return ''.join(random.choices(characters, k=length))

# Adiciona ruído médio à imagem
def add_medium_noise(image):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Pontos aleatórios
    for _ in range(200):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        draw.point((x, y), fill=random.choice([(0, 0, 0), (100, 100, 100), (150, 150, 150)]))

    # Linhas aleatórias
    for _ in range(5):
        start = (random.randint(0, width), random.randint(0, height))
        end = (random.randint(0, width), random.randint(0, height))
        draw.line([start, end], fill=random.choice([(50, 50, 50), (0, 0, 0)]), width=1)

    # Blur leve
    return image.filter(ImageFilter.GaussianBlur(radius=0.5))

# Gera e salva N CAPTCHAs com nomes iguais ao texto
def generate_captcha(n, save_dir="captchas"):
    os.makedirs(save_dir, exist_ok=True)
    image_captcha = ImageCaptcha(width=160, height=60)

    for i in range(n):
        text = generate_random_text()
        image = image_captcha.generate_image(text)
        image = add_medium_noise(image)
        save_path = os.path.join(save_dir, f"{text}.png")
        image.save(save_path)
        print(f"[{i+1}/{n}] CAPTCHA salvo como: {save_path}")

# Exemplo de uso
if __name__ == "__main__":
    generate_captcha(10000)
