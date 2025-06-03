import cv2
import numpy as np
from skimage.morphology import remove_small_objects
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

cleaned_img_name = 'cleaned_image.png'
# Carrega a imagem
img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Binariza a imagem
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Remove pequenos ruídos com morfologia
kernel = np.ones((2, 2), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Remove objetos pequenos (como pontos) usando a skimage
bool_img = opening.astype(bool)
cleaned = remove_small_objects(bool_img, min_size=50, connectivity=2)

# Converte de volta para imagem
cleaned_img = (cleaned * 255).astype(np.uint8)

# Inverte a imagem para deixar o texto em preto
final = cv2.bitwise_not(cleaned_img)
scale_percent = 200  # 200% do tamanho original

# Calcula o novo tamanho
width = int(final.shape[1] * scale_percent / 100)
height = int(final.shape[0] * scale_percent / 100)
dim = (width, height)
resized_final = cv2.resize(final, dim, interpolation=cv2.INTER_LINEAR)
# Salva a imagem final
cv2.imwrite(cleaned_img_name, resized_final)



# Verifica se há GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Carrega imagem
image = Image.open(cleaned_img_name).convert("RGB")

# Carrega processor e modelo e move para GPU
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed', use_fast=True)
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed',
                                                  ignore_mismatched_sizes=True).to(device)

# Processa imagem
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

# Gera texto
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Texto extraído:", generated_text)