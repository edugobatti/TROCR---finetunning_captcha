import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import random
import time
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# DATASET COM AUGMENTATION
# ============================================
class AntiOverfittingDataset(Dataset):
    def __init__(self, image_paths, labels, processor, mode='train', augment_prob=0.8):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.mode = mode
        self.augment_prob = augment_prob
    
    def augment_image(self, image):
        """Augmentation para prevenir overfitting"""
        if self.mode != 'train' or random.random() > self.augment_prob:
            return image
        
        # Rotação
        if random.random() < 0.7:
            angle = random.uniform(-8, 8)
            image = image.rotate(angle, fillcolor=(255, 255, 255))
        
        # Zoom
        if random.random() < 0.6:
            zoom = random.uniform(0.85, 1.15)
            w, h = image.size
            new_w, new_h = int(w * zoom), int(h * zoom)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            
            if zoom > 1:
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                image = image.crop((left, top, left + w, top + h))
            else:
                new_image = Image.new('RGB', (w, h), (255, 255, 255))
                left = (w - new_w) // 2
                top = (h - new_h) // 2
                new_image.paste(image, (left, top))
                image = new_image
        
        # Brilho/Contraste
        if random.random() < 0.7:
            brightness = ImageEnhance.Brightness(image)
            image = brightness.enhance(random.uniform(0.7, 1.3))
            
            contrast = ImageEnhance.Contrast(image)
            image = contrast.enhance(random.uniform(0.7, 1.3))
        
        # Blur
        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
        
        # Ruído
        if random.random() < 0.5:
            pixels = np.array(image)
            noise = np.random.normal(0, 8, pixels.shape)
            pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(pixels)
        
        return image
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        
        # Aplicar augmentation
        if self.mode == 'train':
            image = self.augment_image(image)
        
        # Processar
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
        # Label
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

# ============================================
# FUNÇÕES DE CHECKPOINT
# ============================================
def save_checkpoint(model, optimizer, epoch, best_accuracy, train_losses, val_losses, 
                   patience_counter, checkpoint_path):
    """Salva checkpoint completo do treinamento"""
    checkpoint = {
        'epoch': epoch,
        'best_accuracy': best_accuracy,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'patience_counter': patience_counter,
        'optimizer_state_dict': optimizer.state_dict(),
        'random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
    }
    
    # Se CUDA disponível, salvar estado CUDA também
    if torch.cuda.is_available():
        checkpoint['cuda_random_state'] = torch.cuda.get_rng_state()
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"✓ Checkpoint saved at epoch {epoch+1}")

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Carrega checkpoint e retorna estado do treinamento"""
    if not os.path.exists(checkpoint_path):
        return None
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Restaurar estados aleatórios
    random.setstate(checkpoint['random_state'])
    np.random.set_state(checkpoint['numpy_random_state'])
    torch.set_rng_state(checkpoint['torch_random_state'])
    
    if torch.cuda.is_available() and 'cuda_random_state' in checkpoint:
        torch.cuda.set_rng_state(checkpoint['cuda_random_state'])
    
    # Restaurar optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'best_accuracy': checkpoint['best_accuracy'],
        'train_losses': checkpoint['train_losses'],
        'val_losses': checkpoint['val_losses'],
        'patience_counter': checkpoint['patience_counter']
    }

# ============================================
# FUNÇÃO PRINCIPAL
# ============================================
def train_anti_overfitting():
    # Configurações
    CAPTCHA_FOLDER = "captchas"
    MODEL_NAME = "microsoft/trocr-base-printed"
    OUTPUT_DIR = "./trocr-anti-overfit-final-overbaseV2-check"
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.1
    NUM_EPOCHS = 50
    PATIENCE = 10
    LABEL_SMOOTHING = 0.1
    DROPOUT = 0.3
    AUGMENT_PROB = 0.8
    SAVE_EVERY = 1  # Salvar checkpoint a cada N epochs
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Criar diretórios
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    logger.info("="*60)
    logger.info("TREINAMENTO ANTI-OVERFITTING COM CHECKPOINT")
    logger.info("="*60)
    logger.info(f"Device: {device}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Weight decay: {WEIGHT_DECAY}")
    logger.info(f"Dropout: {DROPOUT}")
    logger.info(f"Augmentation: {AUGMENT_PROB*100}%")
    logger.info(f"Checkpoint dir: {CHECKPOINT_DIR}")
    
    # Carregar dados
    image_paths = []
    labels = []
    
    for filename in os.listdir(CAPTCHA_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(CAPTCHA_FOLDER, filename)
            label = os.path.splitext(filename)[0]
            
            if 1 <= len(label) <= 20:
                image_paths.append(image_path)
                labels.append(label)
    
    logger.info(f"\nTotal images: {len(image_paths)}")
    
    # Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    
    logger.info(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Verificar se existe checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
    model_checkpoint_path = os.path.join(CHECKPOINT_DIR, "model_checkpoint")
    
    if os.path.exists(model_checkpoint_path) and os.path.exists(checkpoint_path):
        logger.info("\n✓ Found existing checkpoint! Loading...")
        # Carregar modelo do checkpoint
        processor = TrOCRProcessor.from_pretrained(model_checkpoint_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint_path)
    else:
        logger.info(f"\nLoading {MODEL_NAME}...")
        processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    
    # IMPORTANTE: Configurar tokens especiais
    if processor.tokenizer.cls_token_id is None:
        processor.tokenizer.cls_token_id = processor.tokenizer.bos_token_id
    if processor.tokenizer.sep_token_id is None:
        processor.tokenizer.sep_token_id = processor.tokenizer.eos_token_id
    
    # Adicionar novos tokens
    all_chars = set(''.join(labels))
    new_tokens = [c for c in all_chars if c not in processor.tokenizer.get_vocab()]
    
    if new_tokens:
        processor.tokenizer.add_tokens(new_tokens)
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        logger.info(f"Added {len(new_tokens)} new tokens")
    
    # Configurar modelo completamente
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id or processor.tokenizer.eos_token_id
    model.config.bos_token_id = processor.tokenizer.bos_token_id
    model.config.use_cache = True
    
    # Adicionar dropout ao decoder
    if hasattr(model.decoder, 'model') and hasattr(model.decoder.model, 'decoder'):
        for layer in model.decoder.model.decoder.layers:
            layer.dropout = DROPOUT
            if hasattr(layer, 'self_attn'):
                layer.self_attn.dropout = DROPOUT
            if hasattr(layer, 'encoder_attn'):
                layer.encoder_attn.dropout = DROPOUT
        logger.info(f"✓ Dropout {DROPOUT} added to decoder")
    
    model.to(device)
    
    # Gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Datasets
    train_dataset = AntiOverfittingDataset(
        train_paths, train_labels, processor, 
        mode='train', augment_prob=AUGMENT_PROB
    )
    
    val_dataset = AntiOverfittingDataset(
        val_paths, val_labels, processor, 
        mode='val', augment_prob=0
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0
    )
    
    # Optimizer com weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Loss com label smoothing
    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing
            self.confidence = 1.0 - smoothing
        
        def forward(self, logits, targets):
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            # Ignorar padding
            mask = targets != -100
            if mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device)
            
            logits = logits[mask]
            targets = targets[mask]
            
            # Label smoothing
            log_probs = torch.log_softmax(logits, dim=-1)
            nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
            smooth_loss = -log_probs.mean(dim=-1)
            
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
    
    criterion = LabelSmoothingCrossEntropy(LABEL_SMOOTHING)
    
    # Carregar checkpoint se existir
    checkpoint_info = load_checkpoint(checkpoint_path, model, optimizer, device)
    
    if checkpoint_info:
        start_epoch = checkpoint_info['epoch'] + 1
        best_accuracy = checkpoint_info['best_accuracy']
        train_losses = checkpoint_info['train_losses']
        val_losses = checkpoint_info['val_losses']
        patience_counter = checkpoint_info['patience_counter']
        logger.info(f"✓ Resuming from epoch {start_epoch}")
        logger.info(f"✓ Best accuracy so far: {best_accuracy:.2%}")
    else:
        start_epoch = 0
        best_accuracy = 0
        patience_counter = 0
        train_losses = []
        val_losses = []
        logger.info("✓ Starting fresh training")
    
    # Training
    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start = time.time()
        
        # Train
        model.train()
        train_loss = 0
        
        # Progress bar para treino
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training", 
                         leave=False, position=0)
        
        for batch_idx, batch in enumerate(train_pbar):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward - garantir que requer gradiente
            pixel_values.requires_grad = False  # Input não precisa de gradiente
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = criterion(outputs.logits, labels)
            
            # Verificar se loss requer gradiente
            if not loss.requires_grad:
                logger.error("Loss doesn't require gradient!")
                continue
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Atualizar progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                                   'avg_loss': f'{train_loss/(batch_idx+1):.4f}'})
        
        train_pbar.close()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        # Progress bar para validação
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation", 
                       leave=False, position=0)
        
        with torch.no_grad():
            for batch in val_pbar:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                # Loss
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                
                # Generate - método simplificado
                try:
                    # Método 1: Usar generate padrão
                    generated_ids = model.generate(
                        pixel_values,
                        max_new_tokens=16,
                        num_beams=4
                    )
                except:
                    try:
                        # Método 2: Especificar decoder_start_token_id
                        generated_ids = model.generate(
                            inputs=pixel_values,
                            decoder_start_token_id=model.config.decoder_start_token_id,
                            max_length=16,
                            num_beams=4
                        )
                    except:
                        # Método 3: Greedy search simples
                        generated_ids = model.generate(
                            pixel_values,
                            max_length=16,
                            num_beams=1,
                            do_sample=False
                        )
                
                # Decode
                pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                labels_clean = labels.clone()
                labels_clean[labels_clean == -100] = processor.tokenizer.pad_token_id
                true_texts = processor.batch_decode(labels_clean, skip_special_tokens=True)
                
                # Calculate accuracy
                for pred, true in zip(pred_texts, true_texts):
                    pred = pred.strip().upper()
                    true = true.strip().upper()
                    
                    total += 1
                    if pred == true:
                        correct += 1
                
                # Atualizar progress bar
                current_acc = correct / total if total > 0 else 0
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                     'accuracy': f'{current_acc:.2%}'})
        
        val_pbar.close()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        accuracy = correct / total
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} ({epoch_time:.1f}s)")
        logger.info(f"Loss - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
        logger.info(f"Gap (Val-Train): {avg_val_loss - avg_train_loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            
            model.save_pretrained(OUTPUT_DIR)
            processor.save_pretrained(OUTPUT_DIR)
            
            logger.info(f"✅ New best model saved! Accuracy: {best_accuracy:.2%}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"\nEarly stopping after {PATIENCE} epochs without improvement!")
                break
        
        # Salvar checkpoint a cada SAVE_EVERY epochs
        if (epoch + 1) % SAVE_EVERY == 0:
            # Salvar modelo checkpoint
            model.save_pretrained(model_checkpoint_path)
            processor.save_pretrained(model_checkpoint_path)
            
            # Salvar estado do treinamento
            save_checkpoint(
                model, optimizer, epoch, best_accuracy, 
                train_losses, val_losses, patience_counter, 
                checkpoint_path
            )
        
        # Examples every 5 epochs
        if (epoch + 1) % 5 == 0:
            logger.info("\nExample predictions:")
            model.eval()
            with torch.no_grad():
                for i in range(min(5, len(val_dataset))):
                    item = val_dataset[i]
                    pixel_values = item['pixel_values'].unsqueeze(0).to(device)
                    
                    try:
                        generated_ids = model.generate(pixel_values, max_new_tokens=16)
                    except:
                        generated_ids = model.generate(
                            inputs=pixel_values,
                            decoder_start_token_id=model.config.decoder_start_token_id,
                            max_length=16
                        )
                    
                    pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    true_text = val_labels[i]
                    
                    status = "✓" if pred_text.upper() == true_text.upper() else "✗"
                    logger.info(f"  {status} True: '{true_text}' → Pred: '{pred_text}'")
    
    # Final analysis
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Best accuracy: {best_accuracy:.2%}")
    logger.info(f"Model saved at: {OUTPUT_DIR}")
    
    # Limpar checkpoints após conclusão
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info("✓ Checkpoint files cleaned up")
    
    if len(train_losses) > 0 and len(val_losses) > 0:
        final_gap = val_losses[-1] - train_losses[-1]
        logger.info(f"\nOverfitting Analysis:")
        logger.info(f"Final gap (Val-Train): {final_gap:.4f}")
        
        if final_gap < 0.5:
            logger.info("✅ Excellent! Minimal overfitting.")
        elif final_gap < 1.0:
            logger.info("✓ Good! Controlled overfitting.")
        elif final_gap < 2.0:
            logger.info("⚠️ Warning! Moderate overfitting.")
        else:
            logger.info("❌ Significant overfitting detected!")

if __name__ == "__main__":
    try:
        train_anti_overfitting()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info("✓ Progress saved - you can resume training by running the script again")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()