import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, TrainingArguments, Trainer
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CaptchaDataset(Dataset):
    def __init__(self, image_paths, labels, processor, max_length=16):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Carregar imagem
        image = Image.open(self.image_paths[idx]).convert("RGB")
        
        # Processar imagem
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
        # Processar labels
        encoding = self.processor.tokenizer(
            self.labels[idx],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        labels = encoding.input_ids.squeeze(0)
        
        # Substituir padding token id por -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

def prepare_dataset(captcha_folder):
    """Prepara o dataset"""
    image_paths = []
    labels = []
    
    if not os.path.exists(captcha_folder):
        raise FileNotFoundError(f"Pasta {captcha_folder} não encontrada!")
    
    logger.info(f"Carregando imagens de {captcha_folder}...")
    
    for filename in os.listdir(captcha_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(captcha_folder, filename)
            label = os.path.splitext(filename)[0]
            
            try:
                Image.open(image_path).verify()
                image_paths.append(image_path)
                labels.append(label)
            except:
                logger.warning(f"Imagem inválida: {filename}")
    
    logger.info(f"Total de imagens: {len(image_paths)}")
    return image_paths, labels

def main():
    # Configurações
    CAPTCHA_FOLDER = "captchas"
    MODEL_NAME = "microsoft/trocr-small-printed"
    OUTPUT_DIR = "./trocr-finetuned-captcha"
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 30
    
    # Criar diretório
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando: {device}")
    
    # Carregar modelo
    logger.info("Carregando modelo...")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.to(device)
    
    # Configurar modelo
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    
    # Preparar dados
    image_paths, labels = prepare_dataset(CAPTCHA_FOLDER)
    
    if len(image_paths) < 2:
        raise ValueError("Precisa de pelo menos 2 imagens!")
    
    # Dividir dados
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    
    logger.info(f"Treino: {len(train_paths)}, Validação: {len(val_paths)}")
    
    # Criar datasets
    train_dataset = CaptchaDataset(train_paths, train_labels, processor)
    eval_dataset = CaptchaDataset(val_paths, val_labels, processor)
    
    # Função de collate customizada
    def collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return {'pixel_values': pixel_values, 'labels': labels}
    
    # Criar dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Otimizador
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Treinamento manual simples
    logger.info("Iniciando treinamento...")
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Média Loss: {avg_loss:.4f}")
        
        # Validação simples
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()
        
        val_loss = val_loss / len(eval_loader)
        logger.info(f"Validação Loss: {val_loss:.4f}")
        model.train()
    
    # Salvar modelo
    logger.info("Salvando modelo...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    # Testar
    logger.info("\nTestando modelo:")
    model.eval()
    
    for i in range(min(5, len(val_paths))):
        image = Image.open(val_paths[i]).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        logger.info(f"Arquivo: {os.path.basename(val_paths[i])}")
        logger.info(f"Real: {val_labels[i]}")
        logger.info(f"Predição: {generated_text}")
        logger.info("-" * 40)
    
    logger.info("Treinamento concluído!")

def inference(image_path, model_path="./trocr-finetuned-captcha"):
    """Usar o modelo treinado"""
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

if __name__ == "__main__":
    main()