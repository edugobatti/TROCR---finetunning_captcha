from datasets import load_dataset
import re
import os

# Nome do dataset
DATASET_NAME = "hammer888/captcha-data"

# Pasta onde o script está
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = f"{ROOT_DIR}/captchas/"  # Salvar na raiz

# Carregar dataset
print("Baixando dataset...")
dataset = load_dataset(DATASET_NAME, split="train")

# Função para extrair nome entre aspas
def extrair_nome(texto):
    match = re.search(r"'(.*?)'", texto)
    return match.group(1) if match else None

# Loop para salvar imagens
for i, item in enumerate(dataset):
    image = item["image"]  # Já é um objeto PIL.Image.Image
    texto = item["text"]

    nome_arquivo = extrair_nome(texto)
    if not nome_arquivo:
        continue

    caminho_arquivo = os.path.join(OUTPUT_DIR, f"{nome_arquivo}.png")
    image.save(caminho_arquivo, format="PNG")

    if i % 500 == 0:
        print(f"{i} imagens salvas...")

print("Download e salvamento concluídos!")
