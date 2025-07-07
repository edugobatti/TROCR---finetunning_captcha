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
import glob

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
def save_checkpoint(model, optimizer, processor, epoch, best_accuracy, train_losses, val_losses, 
                   patience_counter, output_dir):
    """Salvar checkpoint completo"""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Salvar modelo
    model_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}")
    model.save_pretrained(model_path)
    processor.save_pretrained(model_path)
    
    # Salvar estado do treinamento
    checkpoint_data = {
        'epoch': epoch,
        'best_accuracy': best_accuracy,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'patience_counter': patience_counter,
        'optimizer_state_dict': optimizer.state_dict(),
        'random_state': random.getstate(),
        'torch_random_state': torch.get_rng_state().tolist(),
        'numpy_random_state': np.random.get_state()
    }
    
    # Salvar estado atual
    checkpoint_path = os.path.join(checkpoint_dir, f"training_state_epoch_{epoch}.json")
    with open(checkpoint_path, 'w') as f:
        json.dump({k: v for k, v in checkpoint_data.items() 
                  if k not in ['optimizer_state_dict', 'random_state', 'torch_random_state', 'numpy_random_state']}, f, indent=2)
    
    # Salvar estados especiais separadamente
    torch.save({
        'optimizer_state_dict': checkpoint_data['optimizer_state_dict'],
        'random_state': checkpoint_data['random_state'],
        'torch_random_state': checkpoint_data['torch_random_state'],
        'numpy_random_state': checkpoint_data['numpy_random_state']
    }, os.path.join(checkpoint_dir, f"training_states_epoch_{epoch}.pth"))
    
    logger.info(f"✅ Checkpoint saved for epoch {epoch}")
    
    # Limpar checkpoints antigos (manter apenas os 3 mais recentes)
    cleanup_old_checkpoints(checkpoint_dir, keep_last=3)

def cleanup_old_checkpoints(checkpoint_dir, keep_last=3):
    """Remove checkpoints antigos, mantendo apenas os mais recentes"""
    try:
        # Buscar todos os checkpoints
        checkpoint_patterns = [
            "checkpoint_epoch_*",
            "training_state_epoch_*.json",
            "training_states_epoch_*.pth"
        ]
        
        for pattern in checkpoint_patterns:
            files = glob.glob(os.path.join(checkpoint_dir, pattern))
            if len(files) > keep_last:
                # Ordenar por número de epoch
                def extract_epoch(filepath):
                    try:
                        filename = os.path.basename(filepath)
                        if 'epoch_' in filename:
                            return int(filename.split('epoch_')[1].split('.')[0].split('_')[0])
                    except:
                        return 0
                    return 0
                
                files.sorted(key=extract_epoch)
                files_to_remove = files[:-keep_last]
                
                for file_path in files_to_remove:
                    try:
                        if os.path.isdir(file_path):
                            import shutil
                            shutil.rmtree(file_path)
                        else:
                            os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Could not remove old checkpoint {file_path}: {e}")
    except Exception as e:
        logger.warning(f"Error cleaning up checkpoints: {e}")

def load_latest_checkpoint(output_dir):
    """Carregar o checkpoint mais recente"""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Encontrar o checkpoint mais recente
    json_files = glob.glob(os.path.join(checkpoint_dir, "training_state_epoch_*.json"))
    if not json_files:
        return None
    
    # Pegar o epoch mais alto
    latest_epoch = -1
    latest_file = None
    
    for json_file in json_files:
        try:
            epoch = int(os.path.basename(json_file).split('epoch_')[1].split('.')[0])
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_file = json_file
        except:
            continue
    
    if latest_file is None:
        return None
    
    try:
        # Carregar dados JSON
        with open(latest_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Carregar estados especiais
        states_file = os.path.join(checkpoint_dir, f"training_states_epoch_{latest_epoch}.pth")
        if os.path.exists(states_file):
            states = torch.load(states_file, map_location='cpu')
            checkpoint_data.update(states)
        
        # Caminho do modelo
        model_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{latest_epoch}")
        if os.path.exists(model_path):
            checkpoint_data['model_path'] = model_path
            return checkpoint_data
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
    
    return None

# ============================================
# FUNÇÃO PRINCIPAL COM CHECKPOINTS
# ============================================
def train_anti_overfitting():
    # Configurações
    CAPTCHA_FOLDER = "captchas"
    MODEL_NAME = "microsoft/trocr-base-printed"
    OUTPUT_DIR = "./trocr-anti-overfit-final-overbaseV6-chars"
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.1
    NUM_EPOCHS = 30
    PATIENCE = 5
    LABEL_SMOOTHING = 0.1
    DROPOUT = 0.3
    AUGMENT_PROB = 0.8
    SIMILARITY_WEIGHT = 2.5  # Peso para erros entre caracteres similares
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("="*60)
    logger.info("TREINAMENTO ANTI-OVERFITTING")
    logger.info("="*60)
    logger.info(f"Device: {device}")
    logger.info(f"Modelo: {MODEL_NAME}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Weight decay: {WEIGHT_DECAY}")
    logger.info(f"Dropout: {DROPOUT}")
    logger.info(f"Augmentation: {AUGMENT_PROB*100}%")
    logger.info(f"Similarity penalty weight: {SIMILARITY_WEIGHT}x")
    
    # Verificar se existe checkpoint
    checkpoint = load_latest_checkpoint(OUTPUT_DIR)
    start_epoch = 0
    best_accuracy = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    if checkpoint:
        logger.info(f"\n🔄 RESUMING TRAINING FROM CHECKPOINT")
        logger.info(f"Checkpoint found: Epoch {checkpoint['epoch']}")
        logger.info(f"Best accuracy so far: {checkpoint['best_accuracy']:.2%}")
        
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
        patience_counter = checkpoint['patience_counter']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        
        # Restaurar estados aleatórios
        if 'random_state' in checkpoint:
            random.setstate(checkpoint['random_state'])
        if 'torch_random_state' in checkpoint:
            torch.set_rng_state(torch.tensor(checkpoint['torch_random_state'], dtype=torch.uint8))
        if 'numpy_random_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_random_state'])
        
        logger.info(f"Random states restored")
    else:
        logger.info(f"\n🆕 STARTING NEW TRAINING")
    
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
    
    # Split (com seed fixo para consistência)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    
    logger.info(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Carregar modelo e processor
    if checkpoint and 'model_path' in checkpoint:
        logger.info(f"Loading model from checkpoint: {checkpoint['model_path']}")
        processor = TrOCRProcessor.from_pretrained(checkpoint['model_path'])
        model = VisionEncoderDecoderModel.from_pretrained(checkpoint['model_path'])
    else:
        logger.info(f"Loading base model: {MODEL_NAME}")
        processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
        
        # IMPORTANTE: Configurar tokens especiais apenas para modelo novo
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
        logger.info(f"✓ Dropout {DROPOUT} configured")
    
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
    
    # Restaurar estado do optimizer se existir
    if checkpoint and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✓ Optimizer state restored")
        except Exception as e:
            logger.warning(f"Could not restore optimizer state: {e}")
    
    # Loss com label smoothing e ponderação por similaridade
    class WeightedLabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, processor, smoothing=0.1, similarity_weight=2.0):
            super().__init__()
            self.processor = processor
            self.smoothing = smoothing
            self.confidence = 1.0 - smoothing
            self.similarity_weight = similarity_weight
            
            # Definir grupos de caracteres similares
            self.similar_groups = [
                # Números vs letras
                ['0', 'O', 'o'],
                ['1', 'I', 'l', '|'],
                ['2', 'Z'],
                ['5', 'S', 's'],
                ['6', 'G', 'b'],
                ['8', 'B'],
                ['9', 'g', 'q'],
                
                # Letras similares
                ['I', 'l', '1', '|'],
                ['O', '0', 'o', 'Q'],
                ['C', 'c', 'G'],
                ['P', 'p', 'R'],
                ['U', 'u', 'V', 'v'],
                ['W', 'w', 'M'],
                ['n', 'h', 'r'],
                ['m', 'rn', 'nn'],
                ['cl', 'd'],
                ['ti', 'ii'],
                ['vv', 'w'],
                ['rn', 'm'],
                
                # Maiúsculas vs minúsculas
                ['A', 'a'], ['B', 'b'], ['C', 'c'], ['D', 'd'], ['E', 'e'],
                ['F', 'f'], ['G', 'g'], ['H', 'h'], ['J', 'j'], ['K', 'k'],
                ['L', 'l'], ['M', 'm'], ['N', 'n'], ['P', 'p'], ['Q', 'q'],
                ['R', 'r'], ['S', 's'], ['T', 't'], ['U', 'u'], ['V', 'v'],
                ['W', 'w'], ['X', 'x'], ['Y', 'y'], ['Z', 'z'],
            ]
            
            # Criar mapeamento de similaridade
            self.similarity_map = {}
            for group in self.similar_groups:
                for char1 in group:
                    for char2 in group:
                        if char1 != char2:
                            self.similarity_map[(char1, char2)] = True
            
            logger.info(f"✓ Similarity groups configured: {len(self.similar_groups)} groups")
            logger.info(f"✓ Similar character pairs: {len(self.similarity_map)}")
        
        def get_similarity_weight(self, predicted_text, true_text):
            """Calcular peso baseado na similaridade entre caracteres"""
            if len(predicted_text) != len(true_text):
                return 1.0  # Peso normal para erros de comprimento
            
            total_weight = 0.0
            error_count = 0
            
            for pred_char, true_char in zip(predicted_text, true_text):
                if pred_char != true_char:
                    error_count += 1
                    # Verificar se é erro entre caracteres similares
                    if (pred_char, true_char) in self.similarity_map or (true_char, pred_char) in self.similarity_map:
                        total_weight += self.similarity_weight  # Peso maior para similares
                    else:
                        total_weight += 1.0  # Peso normal
            
            return total_weight / max(error_count, 1)  # Média dos pesos
        
        def forward(self, logits, targets, pixel_values=None):
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            # Ignorar padding
            mask = targets != -100
            if mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device)
            
            logits = logits[mask]
            targets = targets[mask]
            
            # Loss básico com label smoothing
            log_probs = torch.log_softmax(logits, dim=-1)
            nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
            smooth_loss = -log_probs.mean(dim=-1)
            
            base_loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            
            # Aplicar ponderação por similaridade se possível
            if pixel_values is not None:
                try:
                    # Obter predições para calcular pesos
                    predicted_ids = torch.argmax(logits, dim=-1)
                    
                    # Reshape para sequências
                    batch_size = pixel_values.size(0)
                    seq_len = len(targets) // batch_size
                    
                    if len(targets) % batch_size == 0:
                        pred_sequences = predicted_ids.view(batch_size, seq_len)
                        true_sequences = targets.view(batch_size, seq_len)
                        loss_sequences = base_loss.view(batch_size, seq_len)
                        
                        weighted_losses = []
                        
                        for i in range(batch_size):
                            pred_seq = pred_sequences[i]
                            true_seq = true_sequences[i]
                            loss_seq = loss_sequences[i]
                            
                            # Converter para texto
                            pred_text = self.processor.tokenizer.decode(pred_seq, skip_special_tokens=True)
                            true_text = self.processor.tokenizer.decode(true_seq, skip_special_tokens=True)
                            
                            # Calcular peso
                            weight = self.get_similarity_weight(pred_text, true_text)
                            
                            # Aplicar peso
                            weighted_loss = loss_seq * weight
                            weighted_losses.append(weighted_loss.mean())
                        
                        return torch.stack(weighted_losses).mean()
                    
                except Exception as e:
                    # Em caso de erro, usar loss básico
                    pass
            
            return base_loss.mean()
    
    criterion = WeightedLabelSmoothingCrossEntropy(processor, LABEL_SMOOTHING, similarity_weight=SIMILARITY_WEIGHT)
    
    # Training
    logger.info("\n" + "="*60)
    if checkpoint:
        logger.info(f"RESUMING TRAINING FROM EPOCH {start_epoch}")
    else:
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
            loss = criterion(outputs.logits, labels, pixel_values)
            
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
                loss = criterion(outputs.logits, labels, pixel_values)
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
        
        # SALVAR CHECKPOINT A CADA ÉPOCA
        save_checkpoint(model, optimizer, processor, epoch, best_accuracy, 
                       train_losses, val_losses, patience_counter, OUTPUT_DIR)
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            
            # Salvar melhor modelo separadamente
            best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            processor.save_pretrained(best_model_dir)
            
            logger.info(f"✅ Novo modelo salvo: {best_accuracy:.2%}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"\nEarly stopping after {PATIENCE} epochs without improvement!")
                break
        
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
    logger.info("TRAINING COMPLETO")
    logger.info("="*60)
    logger.info(f"Melhor acuracia: {best_accuracy:.2%}")
    logger.info(f"Melhor modelo salvo: {os.path.join(OUTPUT_DIR, 'best_model')}")
    logger.info(f"Checkpoint salvo: {os.path.join(OUTPUT_DIR, 'checkpoints')}")
    
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
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()