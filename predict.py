import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Caminhos dos arquivos treinados
MODEL_DIR = "trocr_captcha_model/final_model"
PROCESSOR_DIR = "trocr_captcha_model/final_processor"

# Caminho da imagem para inferência
IMAGE_PATH = "cleaned_image.png"

def predict(image_path):
    # Seleciona o dispositivo (GPU ou CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # Carrega modelo e processor fine-tuned
    processor = TrOCRProcessor.from_pretrained(PROCESSOR_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(device)

    # Configurações necessárias
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id

    # Carrega a imagem
    image = Image.open(image_path).convert("RGB")

    # Prepara a imagem
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Geração do texto
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"\nTexto previsto para {image_path}: {generated_text}")

if __name__ == "__main__":
    predict(IMAGE_PATH)
