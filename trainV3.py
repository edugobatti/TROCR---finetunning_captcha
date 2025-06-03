import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import time

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleCaptchaDataset(Dataset):
    def __init__(self, image_paths, labels, processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Carregar imagem
            image = Image.open(self.image_paths[idx]).convert("RGB")
            
            # Pré-processamento simples
            image = image.convert('L')  # Grayscale
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)  # Aumentar contraste
            image = image.convert('RGB')  # Voltar para RGB
            
            # Processar para o modelo
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
            
            # Processar label
            label_text = self.labels[idx].upper()
            encoding = self.processor.tokenizer(
                label_text,
                padding="max_length",
                max_length=16,
                truncation=True,
                return_tensors="pt"
            )
            
            labels = encoding.input_ids.squeeze(0)
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
            return {
                "pixel_values": pixel_values,
                "labels": labels
            }
        except Exception as e:
            logger.error(f"Erro ao processar {self.image_paths[idx]}: {e}")
            # Retornar dados vazios em caso de erro
            return {
                "pixel_values": torch.zeros((3, 384, 384)),
                "labels": torch.ones(16, dtype=torch.long) * -100
            }

def validate_and_load_data(captcha_folder):
    """Carrega e valida os dados"""
    if not os.path.exists(captcha_folder):
        raise FileNotFoundError(f"Pasta {captcha_folder} não encontrada!")
    
    image_paths = []
    labels = []
    
    logger.info(f"Carregando dados de {captcha_folder}...")
    
    files = os.listdir(captcha_folder)
    total_files = len(files)
    
    for i, filename in enumerate(files):
        if i % 100 == 0:
            logger.info(f"Processando arquivo {i}/{total_files}...")
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(captcha_folder, filename)
            label = os.path.splitext(filename)[0]
            
            # Validação básica
            if len(label) >= 1 and len(label) <= 20:
                try:
                    # Verificar se a imagem pode ser aberta
                    with Image.open(image_path) as img:
                        if img.size[0] > 10 and img.size[1] > 10:
                            image_paths.append(image_path)
                            labels.append(label)
                except:
                    logger.warning(f"Não foi possível abrir: {filename}")
    
    logger.info(f"Total de imagens válidas: {len(image_paths)}")
    
    # Mostrar exemplos
    logger.info("\nExemplos de dados:")
    for i in range(min(5, len(labels))):
        logger.info(f"  {i+1}. {os.path.basename(image_paths[i])} → '{labels[i]}'")
    
    return image_paths, labels

def main():
    # Configurações simples
    CAPTCHA_FOLDER = "captchas"
    MODEL_NAME = "microsoft/trocr-base-printed"  # Modelo base mais estável
    OUTPUT_DIR = "./trocr-captcha-model"
    BATCH_SIZE = 32  # Batch moderado
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Dispositivo: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memória disponível: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Carregar dados
    try:
        image_paths, labels = validate_and_load_data(CAPTCHA_FOLDER)
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        return
    
    if len(image_paths) < 10:
        logger.error("Poucos dados! Precisa de pelo menos 10 imagens.")
        return
    
    # Dividir dados
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    
    logger.info(f"\nDivisão dos dados:")
    logger.info(f"  Treino: {len(train_paths)} imagens")
    logger.info(f"  Validação: {len(val_paths)} imagens")
    
    # Carregar modelo e processor
    logger.info(f"\nCarregando modelo {MODEL_NAME}...")
    try:
        processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
        model.to(DEVICE)
        logger.info("Modelo carregado com sucesso!")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        return
    
    # Configurar tokens do modelo
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    
    # Adicionar caracteres especiais ao vocabulário
    special_chars = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZqwertyuioplkjhgfdsazxcvbnm')
    new_tokens = []
    for char in special_chars:
        if char not in processor.tokenizer.get_vocab():
            new_tokens.append(char)
    
    if new_tokens:
        processor.tokenizer.add_tokens(new_tokens)
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        logger.info(f"Adicionados {len(new_tokens)} novos tokens ao vocabulário")
    
    # Criar datasets
    logger.info("\nCriando datasets...")
    train_dataset = SimpleCaptchaDataset(train_paths, train_labels, processor)
    val_dataset = SimpleCaptchaDataset(val_paths, val_labels, processor)
    
    # Criar dataloaders (SEM multi-workers para evitar travamento)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # Sem multiprocessing
        pin_memory=False  # Desabilitar para evitar problemas
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0
    )
    
    # Otimizador
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Loop de treinamento
    logger.info("\nIniciando treinamento...")
    best_accuracy = 0
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # Treino
        model.train()
        total_loss = 0
        batch_count = 0
        
        logger.info(f"\nÉpoca {epoch + 1}/{NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                pixel_values = batch['pixel_values'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                # Forward pass
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # Log a cada 10 batches
                if batch_idx % 10 == 0:
                    logger.info(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
                
            except Exception as e:
                logger.error(f"Erro no batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(batch_count, 1)
        
        # Validação
        logger.info("Validando...")
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    pixel_values = batch['pixel_values'].to(DEVICE)
                    labels = batch['labels'].to(DEVICE)
                    
                    # Gerar predições
                    generated_ids = model.generate(pixel_values, max_length=16)
                    
                    # Decodificar
                    pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    
                    # Decodificar labels verdadeiros
                    labels_clean = labels.clone()
                    labels_clean[labels_clean == -100] = processor.tokenizer.pad_token_id
                    true_texts = processor.batch_decode(labels_clean, skip_special_tokens=True)
                    
                    # Calcular acertos
                    for pred, true in zip(pred_texts, true_texts):
                        total += 1
                        if pred.strip().upper() == true.strip().upper():
                            correct += 1
                
                except Exception as e:
                    logger.error(f"Erro na validação batch {batch_idx}: {e}")
                    continue
        
        accuracy = correct / max(total, 1)
        epoch_time = time.time() - epoch_start
        
        logger.info(f"\nResumo da época {epoch + 1}:")
        logger.info(f"  Loss média: {avg_loss:.4f}")
        logger.info(f"  Acurácia: {accuracy:.2%} ({correct}/{total})")
        logger.info(f"  Tempo: {epoch_time:.1f}s")
        
        # Salvar melhor modelo
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            processor.save_pretrained(OUTPUT_DIR)
            logger.info(f"  ✓ Novo melhor modelo salvo! (Acurácia: {best_accuracy:.2%})")
        
        # Mostrar exemplos de predições
        if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
            logger.info("\nExemplos de predições:")
            with torch.no_grad():
                for i in range(min(3, len(val_dataset))):
                    item = val_dataset[i]
                    pixel_values = item['pixel_values'].unsqueeze(0).to(DEVICE)
                    
                    generated_ids = model.generate(pixel_values, max_length=16)
                    pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    true_text = val_labels[i]
                    
                    status = "✓" if pred_text.upper() == true_text.upper() else "✗"
                    logger.info(f"  {status} Real: '{true_text}' → Predição: '{pred_text}'")
    
    logger.info(f"\n✅ Treinamento concluído!")
    logger.info(f"Melhor acurácia: {best_accuracy:.2%}")
    logger.info(f"Modelo salvo em: {OUTPUT_DIR}")

if __name__ == "__main__":
    # Limpar cache antes de começar
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nTreinamento interrompido pelo usuário.")
    except Exception as e:
        logger.error(f"\nErro fatal: {e}")
        import traceback
        traceback.print_exc()