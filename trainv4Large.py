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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.nn.functional as F
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# DATASET COM AUGMENTATION MELHORADA
# ============================================
class AntiOverfittingDataset(Dataset):
    def __init__(self, image_paths, labels, processor, mode='train', augment_prob=0.5):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.mode = mode
        self.augment_prob = augment_prob
        
        # Cache desabilitado para datasets grandes
        self.cache = {}
        self.use_cache = False  # Desabilitado para economizar memória
    
    def augment_image(self, image):
        """Augmentation mais suave para prevenir overfitting sem destruir features"""
        if self.mode != 'train' or random.random() > self.augment_prob:
            return image
        
        # Lista de augmentations aplicadas
        augmentations_applied = []
        
        # Rotação suave
        if random.random() < 0.5:
            angle = random.uniform(-5, 5)  # Reduzido de -8,8
            image = image.rotate(angle, fillcolor=(255, 255, 255))
            augmentations_applied.append('rotation')
        
        # Zoom mais conservador
        if random.random() < 0.4:
            zoom = random.uniform(0.9, 1.1)  # Reduzido de 0.85,1.15
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
            augmentations_applied.append('zoom')
        
        # Brilho/Contraste mais suave
        if random.random() < 0.5:
            brightness = ImageEnhance.Brightness(image)
            image = brightness.enhance(random.uniform(0.85, 1.15))  # Reduzido
            
            contrast = ImageEnhance.Contrast(image)
            image = contrast.enhance(random.uniform(0.85, 1.15))  # Reduzido
            augmentations_applied.append('brightness/contrast')
        
        # Blur muito leve
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))  # Reduzido
            augmentations_applied.append('blur')
        
        # Ruído reduzido
        if random.random() < 0.3:
            pixels = np.array(image)
            noise = np.random.normal(0, 5, pixels.shape)  # Reduzido de 8
            pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(pixels)
            augmentations_applied.append('noise')
        
        return image
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Usar cache se disponível
        if self.use_cache and idx in self.cache:
            image = self.cache[idx].copy()
        else:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            if self.use_cache:
                self.cache[idx] = image.copy()
        
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
            "labels": labels,
            "text": label_text  # Para debug
        }

# ============================================
# MODELO COM REGULARIZAÇÃO ADICIONAL
# ============================================
class RegularizedTrOCR(nn.Module):
    def __init__(self, base_model, dropout_rate=0.1):
        super().__init__()
        self.model = base_model
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, pixel_values, labels=None):
        # Forward através do modelo base
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        
        # Aplicar dropout adicional nos logits durante treino
        if self.training and outputs.logits is not None:
            outputs.logits = self.dropout(outputs.logits)
        
        return outputs
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

# ============================================
# LOSS FUNCTION MELHORADA
# ============================================
class ImprovedLabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean', ignore_index=-100):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
        self.ignore_index = ignore_index
        
    def forward(self, logits, targets):
        # Reshape
        batch_size = logits.size(0)
        seq_len = logits.size(1)
        vocab_size = logits.size(-1)
        
        logits = logits.view(-1, vocab_size)
        targets = targets.view(-1)
        
        # Máscara para ignorar padding
        mask = targets != self.ignore_index
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Aplicar máscara
        logits_masked = logits[mask]
        targets_masked = targets[mask]
        
        # Verificar se há targets fora do vocabulário
        if targets_masked.max() >= vocab_size:
            logger.warning(f"Target token {targets_masked.max()} exceeds vocab size {vocab_size}")
            targets_masked = torch.clamp(targets_masked, 0, vocab_size - 1)
        
        # Calcular log probabilities com estabilidade numérica
        log_probs = F.log_softmax(logits_masked, dim=-1)
        
        # Se smoothing muito pequeno, usar CE normal para estabilidade
        if self.smoothing < 0.01:
            loss = F.nll_loss(log_probs, targets_masked, reduction='none')
        else:
            # Loss com label smoothing
            n_classes = log_probs.size(-1)
            
            # One-hot encoding dos targets
            one_hot = torch.zeros_like(log_probs).scatter(1, targets_masked.unsqueeze(1), 1)
            
            # Smooth labels
            smooth_labels = one_hot * self.confidence + (1 - one_hot) * self.smoothing / (n_classes - 1)
            
            # Cross entropy com smooth labels
            loss = -(smooth_labels * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ============================================
# FUNÇÃO DE TREINAMENTO MELHORADA
# ============================================
def train_anti_overfitting():
    # Configurações otimizadas para dataset grande
    CAPTCHA_FOLDER = "captchas"
    MODEL_NAME = "microsoft/trocr-large-printed"
    OUTPUT_DIR = "./trocr-anti-overfit-improved-large-100k"
    
    # Hiperparâmetros ajustados para dataset grande
    BATCH_SIZE = 16  # Reduzido para maior estabilidade inicial
    ACCUMULATION_STEPS = 4  # Aumentado para manter batch efetivo de 64
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * ACCUMULATION_STEPS  # 64
    LEARNING_RATE = 2e-5  # Reduzido ainda mais
    MIN_LR = 5e-7  # Learning rate mínimo
    WEIGHT_DECAY = 0.001  # Reduzido
    NUM_EPOCHS = 30  # Reduzido pois dataset maior converge mais rápido
    PATIENCE = 5  # Ajustado
    LABEL_SMOOTHING = 0.1  # Reduzido drasticamente
    DROPOUT = 0.1  # Reduzido para início
    AUGMENT_PROB = 0.5  # Reduzido para início mais estável
    GRADIENT_CLIP_VAL = 0.5  # Reduzido para mais controle
    WARMUP_EPOCHS = 3  # Aumentado para início ainda mais gradual
    VALIDATION_BATCHES = 50  # Validar apenas parte do dataset para economizar tempo
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("="*60)
    logger.info("TREINAMENTO ANTI-OVERFITTING - VERSÃO MELHORADA COM TQDM")
    logger.info("="*60)
    logger.info(f"Device: {device}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Batch size: {BATCH_SIZE} (Effective: {EFFECTIVE_BATCH_SIZE})")
    logger.info(f"Accumulation steps: {ACCUMULATION_STEPS}")
    logger.info(f"Learning rate: {LEARNING_RATE} -> {MIN_LR}")
    logger.info(f"Weight decay: {WEIGHT_DECAY}")
    logger.info(f"Dropout: {DROPOUT}")
    logger.info(f"Label smoothing: {LABEL_SMOOTHING}")
    logger.info(f"Augmentation: {AUGMENT_PROB*100}%")
    logger.info(f"Gradient clipping: {GRADIENT_CLIP_VAL}")
    
    # Carregar dados com barra de progresso
    logger.info("\nLoading dataset...")
    image_paths = []
    labels = []
    
    files = [f for f in os.listdir(CAPTCHA_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in tqdm(files, desc="Loading images"):
        image_path = os.path.join(CAPTCHA_FOLDER, filename)
        label = os.path.splitext(filename)[0]
        
        if 1 <= len(label) <= 20:
            image_paths.append(image_path)
            labels.append(label)
    
    logger.info(f"\nTotal images: {len(image_paths)}")
    
    # Split estratificado
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.15, random_state=42  # Reduzido para 15% validação
    )
    
    logger.info(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Carregar modelo e processor
    logger.info(f"\nLoading {MODEL_NAME}...")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    base_model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    
    # Configurar tokens especiais
    if processor.tokenizer.cls_token_id is None:
        processor.tokenizer.cls_token_id = processor.tokenizer.bos_token_id
    if processor.tokenizer.sep_token_id is None:
        processor.tokenizer.sep_token_id = processor.tokenizer.eos_token_id
    
    # Adicionar novos tokens
    all_chars = set(''.join(labels))
    new_tokens = [c for c in all_chars if c not in processor.tokenizer.get_vocab()]
    
    if new_tokens:
        # Salvar embeddings antigas antes de redimensionar
        old_embeddings = base_model.decoder.model.shared.weight.data.clone()
        
        processor.tokenizer.add_tokens(new_tokens)
        base_model.decoder.resize_token_embeddings(len(processor.tokenizer))
        
        # Inicializar novos embeddings com valores pequenos aleatórios
        new_embeddings = base_model.decoder.model.shared.weight.data
        new_embeddings[:old_embeddings.size(0)] = old_embeddings
        # Inicialização Xavier/Glorot para novos tokens
        nn.init.xavier_uniform_(new_embeddings[old_embeddings.size(0):])
        
        logger.info(f"Added {len(new_tokens)} new tokens with proper initialization")
    
    # Configurar modelo
    base_model.config.decoder_start_token_id = processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id
    base_model.config.pad_token_id = processor.tokenizer.pad_token_id
    base_model.config.eos_token_id = processor.tokenizer.sep_token_id or processor.tokenizer.eos_token_id
    base_model.config.bos_token_id = processor.tokenizer.bos_token_id
    base_model.config.use_cache = True
    base_model.config.max_length = 20  # Definir max_length explicitamente
    
    # Configurar para melhor inicialização
    base_model.config.tie_word_embeddings = True  # Compartilhar embeddings
    base_model.config.decoder.tie_word_embeddings = True
    
    # Aplicar dropout ao decoder
    if hasattr(base_model.decoder, 'model') and hasattr(base_model.decoder.model, 'decoder'):
        for layer in base_model.decoder.model.decoder.layers:
            layer.dropout = DROPOUT
            if hasattr(layer, 'self_attn'):
                layer.self_attn.dropout = DROPOUT
            if hasattr(layer, 'encoder_attn'):
                layer.encoder_attn.dropout = DROPOUT
    
    # Adicionar debug para verificar o modelo
    logger.info("\nModel configuration check:")
    logger.info(f"Vocab size: {len(processor.tokenizer)}")
    logger.info(f"Decoder vocab size: {base_model.decoder.config.vocab_size}")
    logger.info(f"Max length: {base_model.config.max_length}")
    logger.info(f"Pad token id: {processor.tokenizer.pad_token_id}")
    logger.info(f"BOS token id: {processor.tokenizer.bos_token_id}")
    logger.info(f"EOS token id: {processor.tokenizer.eos_token_id}")
    
    # Verificar um batch de exemplo antes de começar
    logger.info("\nChecking sample batch...")
    sample_dataset = AntiOverfittingDataset(
        train_paths[:10], train_labels[:10], processor, 
        mode='train', augment_prob=0
    )
    sample_loader = DataLoader(sample_dataset, batch_size=2, shuffle=False)
    sample_batch = next(iter(sample_loader))
    
    logger.info(f"Sample labels shape: {sample_batch['labels'].shape}")
    logger.info(f"Sample labels max value: {sample_batch['labels'].max().item()}")
    logger.info(f"Sample labels min value (excluding -100): {sample_batch['labels'][sample_batch['labels'] != -100].min().item()}")
    logger.info(f"Sample text: {sample_batch['text']}")
    
    # Criar modelo com regularização
    model = RegularizedTrOCR(base_model, dropout_rate=DROPOUT)
    model.to(device)
    
    # Gradient checkpointing para economizar memória
    model.model.gradient_checkpointing_enable()
    
    # Datasets
    train_dataset = AntiOverfittingDataset(
        train_paths, train_labels, processor, 
        mode='train', augment_prob=AUGMENT_PROB
    )
    
    val_dataset = AntiOverfittingDataset(
        val_paths, val_labels, processor, 
        mode='val', augment_prob=0
    )
    
    # Dataloaders com num_workers otimizado
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Aumentado para dataset grande
        pin_memory=True,
        drop_last=True,
        persistent_workers=True  # Mantém workers vivos
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # Schedulers
    total_steps = len(train_loader) * NUM_EPOCHS // ACCUMULATION_STEPS
    warmup_steps = len(train_loader) * WARMUP_EPOCHS // ACCUMULATION_STEPS
    
    # Cosine annealing com warmup
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(MIN_LR / LEARNING_RATE, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss function
    criterion = ImprovedLabelSmoothingLoss(LABEL_SMOOTHING)
    
    # Training variables
    best_accuracy = 0
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    accuracies = []
    global_step = 0
    
    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    
    # Barra de progresso principal para epochs
    epoch_pbar = tqdm(range(NUM_EPOCHS), desc="Training Progress", position=0)
    
    for epoch in epoch_pbar:
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        # Barra de progresso para batches de treino
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} - Training", position=1, leave=False)
        
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_pbar):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            
            # Calculate loss
            loss = criterion(outputs.logits, labels)
            
            # NÃO adicionar regularização L2 no início
            # Só adicionar após algumas epochs quando o loss estiver estável
            if epoch > 5:
                l2_lambda = 1e-5
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters() if p.requires_grad)
                loss = loss + l2_lambda * l2_norm
            
            # Normalize loss for gradient accumulation
            loss = loss / ACCUMULATION_STEPS
            
            # Backward pass
            loss.backward()
            
            # Update weights every ACCUMULATION_STEPS
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            train_loss += loss.item() * ACCUMULATION_STEPS
            train_steps += 1
            
            # Atualizar barra de progresso
            current_lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({
                'loss': f'{loss.item() * ACCUMULATION_STEPS:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        correct = 0
        total = 0
        
        predictions_sample = []
        
        # Limitar validação para economizar tempo
        max_val_batches = min(len(val_loader), VALIDATION_BATCHES)
        
        # Barra de progresso para validação
        val_pbar = tqdm(enumerate(val_loader), 
                       total=max_val_batches,
                       desc=f"Epoch {epoch+1} - Validation", 
                       position=1, 
                       leave=False)
        
        with torch.no_grad():
            for batch_idx, batch in val_pbar:
                if batch_idx >= max_val_batches:
                    break
                    
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                true_texts_batch = batch['text']
                
                # Calculate validation loss
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                val_steps += 1
                
                # Generate predictions
                try:
                    generated_ids = model.generate(
                        pixel_values,
                        max_new_tokens=16,
                        num_beams=2,  # Reduzido para velocidade
                        early_stopping=True,
                        no_repeat_ngram_size=0,
                        decoder_start_token_id=model.model.config.decoder_start_token_id
                    )
                except Exception as e:
                    logger.warning(f"Generation error: {e}")
                    generated_ids = model.generate(
                        pixel_values,
                        max_length=16,
                        num_beams=1
                    )
                
                # Decode predictions
                pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Calculate accuracy
                for pred, true in zip(pred_texts, true_texts_batch):
                    pred_clean = pred.strip().upper()
                    true_clean = true.strip().upper()
                    
                    total += 1
                    if pred_clean == true_clean:
                        correct += 1
                    
                    # Coletar amostras para log
                    if len(predictions_sample) < 5 and batch_idx == 0:
                        predictions_sample.append((true_clean, pred_clean))
                
                # Atualizar barra de progresso
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct/total:.2%}' if total > 0 else '0%'
                })
        
        avg_val_loss = val_loss / val_steps
        val_losses.append(avg_val_loss)
        
        accuracy = correct / total if total > 0 else 0
        accuracies.append(accuracy)
        
        epoch_time = time.time() - epoch_start
        
        # Atualizar barra de progresso principal
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'val_loss': f'{avg_val_loss:.4f}',
            'acc': f'{accuracy:.2%}',
            'best_acc': f'{best_accuracy:.2%}'
        })
        
        # Log epoch results
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} ({epoch_time:.1f}s)")
        logger.info(f"Loss - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
        logger.info(f"Gap (Val-Train): {avg_val_loss - avg_train_loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Log sample predictions
        if predictions_sample:
            logger.info("\nSample predictions:")
            for true, pred in predictions_sample:
                status = "✓" if pred == true else "✗"
                logger.info(f"  {status} True: '{true}' → Pred: '{pred}'")
        
        # Save best model baseado em accuracy E val_loss
        if accuracy > best_accuracy or (accuracy == best_accuracy and avg_val_loss < best_val_loss):
            best_accuracy = accuracy
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.model.save_pretrained(OUTPUT_DIR)
            processor.save_pretrained(OUTPUT_DIR)
            
            # Salvar checkpoint completo
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'accuracies': accuracies
            }
            torch.save(checkpoint, os.path.join(OUTPUT_DIR, 'checkpoint.pth'))
            
            logger.info(f"✅ New best model saved! Accuracy: {best_accuracy:.2%}, Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"Patience counter: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                logger.info(f"\n⚠️ Early stopping triggered after {PATIENCE} epochs without improvement!")
                epoch_pbar.close()
                break
        
        # Análise de overfitting
        gap = avg_val_loss - avg_train_loss
        if gap > 0.5:
            logger.info("⚠️ Warning: Significant gap detected. Consider reducing learning rate or increasing regularization.")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Best accuracy: {best_accuracy:.2%}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved at: {OUTPUT_DIR}")
    
    # Análise final de overfitting
    if len(train_losses) > 0 and len(val_losses) > 0:
        final_gap = val_losses[-1] - train_losses[-1]
        avg_gap = np.mean([val - train for val, train in zip(val_losses[-5:], train_losses[-5:])])
        
        logger.info(f"\nOverfitting Analysis:")
        logger.info(f"Final gap (Val-Train): {final_gap:.4f}")
        logger.info(f"Average gap (last 5 epochs): {avg_gap:.4f}")
        
        if avg_gap < 0.2:
            logger.info("✅ Excellent! Minimal overfitting.")
        elif avg_gap < 0.5:
            logger.info("✓ Good! Controlled overfitting.")
        elif avg_gap < 1.0:
            logger.info("⚠️ Warning! Moderate overfitting.")
        else:
            logger.info("❌ Significant overfitting detected!")
        
        # Estatísticas de accuracy
        if len(accuracies) > 5:
            recent_acc = accuracies[-5:]
            acc_std = np.std(recent_acc)
            logger.info(f"\nAccuracy stability (last 5 epochs):")
            logger.info(f"Mean: {np.mean(recent_acc):.2%}")
            logger.info(f"Std Dev: {acc_std:.2%}")
            if acc_std > 0.05:
                logger.info("⚠️ High variance in accuracy - training might be unstable")

if __name__ == "__main__":
    try:
        train_anti_overfitting()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()