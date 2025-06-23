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
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import json

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
        self.use_cache = False
    
    def set_augment_prob(self, prob):
        """Ajusta probabilidade de augmentation dinamicamente"""
        self.augment_prob = prob
    
    def augment_image(self, image):
        """Augmentation melhorada para quebrar plateau de 91%"""
        if self.mode != 'train' or random.random() > self.augment_prob:
            return image
        
        # Rotação com mais variação
        if random.random() < 0.4:
            angle = random.uniform(-5, 5)  # Aumentado de -2,2 para -5,5
            image = image.rotate(angle, fillcolor=(255, 255, 255))
        
        # Zoom com mais variação
        if random.random() < 0.3:
            zoom = random.uniform(0.95, 1.05)  # Aumentado de 0.98,1.02
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
        
        # Adicionar ruído gaussiano (NOVO)
        if random.random() < 0.2:
            img_array = np.array(image)
            noise = np.random.normal(0, 5, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)
        
        # Adicionar blur leve (NOVO)
        if random.random() < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Ajustes de brilho/contraste com mais variação
        if random.random() < 0.4:
            brightness = ImageEnhance.Brightness(image)
            image = brightness.enhance(random.uniform(0.9, 1.1))  # Aumentado de 0.95,1.05
            
            contrast = ImageEnhance.Contrast(image)
            image = contrast.enhance(random.uniform(0.9, 1.1))  # Aumentado de 0.95,1.05
        
        # Sharpness (NOVO)
        if random.random() < 0.15:
            sharpness = ImageEnhance.Sharpness(image)
            image = sharpness.enhance(random.uniform(0.8, 1.2))
        
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
            "labels": labels,
            "text": label_text,
            "label_length": len(label_text)
        }

# ============================================
# MODELO COM REGULARIZAÇÃO ADAPTATIVA
# ============================================
class RegularizedTrOCR(nn.Module):
    def __init__(self, base_model, dropout_rate=0.1):
        super().__init__()
        self.model = base_model
        self.dropout = nn.Dropout(dropout_rate)
        self.current_dropout_rate = dropout_rate
        
    def set_dropout(self, dropout_rate):
        """Ajusta dropout dinamicamente"""
        self.current_dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, pixel_values, labels=None):
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        
        # Aplicar dropout apenas durante treino
        if self.training and outputs.logits is not None and self.current_dropout_rate > 0:
            outputs.logits = self.dropout(outputs.logits)
        
        return outputs
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

# ============================================
# LOSS FUNCTION ADAPTATIVA
# ============================================
class AdaptiveLabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean', ignore_index=-100):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def set_smoothing(self, smoothing):
        """Ajusta label smoothing dinamicamente"""
        self.smoothing = smoothing
        
    def forward(self, logits, targets, return_base_loss=False):
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
        
        # Verificar targets válidos
        if targets_masked.max() >= vocab_size:
            logger.warning(f"Target token {targets_masked.max()} exceeds vocab size {vocab_size}")
            targets_masked = torch.clamp(targets_masked, 0, vocab_size - 1)
        
        # Calcular log probabilities
        log_probs = F.log_softmax(logits_masked, dim=-1)
        
        # Loss base (sem smoothing) para comparação justa
        base_loss = F.nll_loss(log_probs, targets_masked, reduction='none')
        
        if self.smoothing < 0.01 or return_base_loss:
            loss = base_loss
        else:
            # Label smoothing
            n_classes = log_probs.size(-1)
            one_hot = torch.zeros_like(log_probs).scatter(1, targets_masked.unsqueeze(1), 1)
            smooth_labels = one_hot * (1.0 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_classes - 1)
            loss = -(smooth_labels * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ============================================
# MONITORAMENTO AVANÇADO
# ============================================
class TrainingMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.best_metrics = {}
        self.error_analysis = defaultdict(list)
        
    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)
            
    def update_best(self, **kwargs):
        self.best_metrics.update(kwargs)
        
    def add_error(self, true_text, pred_text, epoch):
        """Registra erros para análise"""
        self.error_analysis[epoch].append({
            'true': true_text,
            'pred': pred_text,
            'len_diff': len(pred_text) - len(true_text),
            'error_type': self._classify_error(true_text, pred_text)
        })
    
    def _classify_error(self, true_text, pred_text):
        """Classifica tipo de erro"""
        if len(true_text) != len(pred_text):
            return 'length_mismatch'
        elif true_text.lower() == pred_text.lower():
            return 'case_error'
        else:
            # Contar caracteres diferentes
            diff_count = sum(1 for a, b in zip(true_text, pred_text) if a != b)
            if diff_count == 1:
                return 'single_char_error'
            elif diff_count <= 3:
                return 'few_chars_error'
            else:
                return 'many_chars_error'
    
    def get_error_summary(self, epoch):
        """Retorna resumo dos erros da época"""
        if epoch not in self.error_analysis:
            return {}
        
        errors = self.error_analysis[epoch]
        summary = defaultdict(int)
        for error in errors:
            summary[error['error_type']] += 1
        
        return dict(summary)
    
    def save(self, path):
        """Salva métricas em arquivo JSON"""
        data = {
            'metrics': dict(self.metrics),
            'best_metrics': self.best_metrics,
            'error_summary': {str(k): self.get_error_summary(k) for k in self.error_analysis.keys()}
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

# ============================================
# FUNÇÃO DE TREINAMENTO OTIMIZADA
# ============================================
def train_anti_overfitting():
    # Configurações otimizadas para quebrar plateau de 91.72%
    CAPTCHA_FOLDER = "captchas"
    MODEL_NAME = "microsoft/trocr-base-printed"
    OUTPUT_DIR = "./trocr-anti-overfit-improved-cosseno/"
    
    # Detectar se é continuação de treinamento
    RESUME_FROM_CHECKPOINT = "./trocr-anti-overfit-improved-cosseno/checkpoint.pth"
    RESUME_TRAINING = os.path.exists(RESUME_FROM_CHECKPOINT)
    
    # Hiperparâmetros otimizados para quebrar plateau
    BATCH_SIZE = 16  # Reduzido de 32 para mais updates
    ACCUMULATION_STEPS = 4  # Aumentado de 2
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * ACCUMULATION_STEPS  # 64
    
    # Learning rate aumentado para sair do plateau
    LEARNING_RATE = 1e-5  # Aumentado de 5e-6
    MIN_LR = 5e-7  # Aumentado de 1e-7
    WEIGHT_DECAY = 0.00001  # Reduzido drasticamente de 0.0001
    
    # Configurações de treinamento
    NUM_EPOCHS = 30  # epocas
    PATIENCE = 15  # Aumentado de 10
    GRADIENT_CLIP_VAL = 1.0
    VALIDATION_BATCHES = 5000
    
    # Regularização aumentada para quebrar plateau
    LABEL_SMOOTHING = 0.05  # Aumentado de 0.01
    DROPOUT = 0.1  # Aumentado de 0.05
    AUGMENT_PROB = 0.3  # Aumentado de 0.1
    
    # Flags de controle
    USE_ADAPTIVE_TRAINING = True
    USE_ERROR_ANALYSIS = True
    USE_L2_REGULARIZATION = False  # DESABILITADO - era um dos problemas
    USE_COSINE_SCHEDULER = True  # Usar CosineAnnealingWarmRestarts ao invés de ReduceLROnPlateau
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("="*60)
    logger.info("TREINAMENTO TROCR - VERSÃO OTIMIZADA PARA QUEBRAR PLATEAU")
    logger.info("="*60)
    logger.info(f"Device: {device}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Resume from checkpoint: {RESUME_TRAINING}")
    logger.info(f"Batch size: {BATCH_SIZE} (Effective: {EFFECTIVE_BATCH_SIZE})")
    logger.info(f"Learning rate: {LEARNING_RATE} -> {MIN_LR}")
    logger.info(f"Regularization - WD: {WEIGHT_DECAY}, LS: {LABEL_SMOOTHING}, DO: {DROPOUT}")
    logger.info(f"L2 Regularization: {'ENABLED' if USE_L2_REGULARIZATION else 'DISABLED'}")
    
    # Monitor de treinamento
    monitor = TrainingMonitor()
    
    # Carregar dados
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
    
    logger.info(f"Total images: {len(image_paths)}")
    
    # Análise do dataset
    label_lengths = defaultdict(int)
    for label in labels:
        label_lengths[len(label)] += 1
    
    logger.info("\nLabel length distribution:")
    for length in sorted(label_lengths.keys()):
        logger.info(f"  Length {length}: {label_lengths[length]} samples")
    
    # Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.15, random_state=42
    )
    
    logger.info(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Carregar modelo e processor
    if RESUME_TRAINING:
        logger.info(f"\nLoading checkpoint from {RESUME_FROM_CHECKPOINT}...")
        checkpoint = torch.load(RESUME_FROM_CHECKPOINT, map_location=device)
        
        # Carregar de um checkpoint anterior
        base_path = os.path.dirname(RESUME_FROM_CHECKPOINT)
        processor = TrOCRProcessor.from_pretrained(base_path)
        base_model = VisionEncoderDecoderModel.from_pretrained(base_path)
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_accuracy = checkpoint.get('best_accuracy', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Resuming from epoch {start_epoch}, best accuracy: {best_accuracy:.2%}")
    else:
        logger.info(f"\nLoading {MODEL_NAME}...")
        processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
        base_model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
        start_epoch = 0
        best_accuracy = 0
        best_val_loss = float('inf')
        
        # Configurar tokens especiais
        if processor.tokenizer.cls_token_id is None:
            processor.tokenizer.cls_token_id = processor.tokenizer.bos_token_id
        if processor.tokenizer.sep_token_id is None:
            processor.tokenizer.sep_token_id = processor.tokenizer.eos_token_id
        
        # Adicionar novos tokens
        all_chars = set(''.join(labels))
        new_tokens = [c for c in all_chars if c not in processor.tokenizer.get_vocab()]
        
        if new_tokens:
            old_embeddings = base_model.decoder.model.shared.weight.data.clone()
            processor.tokenizer.add_tokens(new_tokens)
            base_model.decoder.resize_token_embeddings(len(processor.tokenizer))
            
            new_embeddings = base_model.decoder.model.shared.weight.data
            new_embeddings[:old_embeddings.size(0)] = old_embeddings
            nn.init.xavier_uniform_(new_embeddings[old_embeddings.size(0):])
            
            logger.info(f"Added {len(new_tokens)} new tokens")
    
    # Configurar modelo
    base_model.config.decoder_start_token_id = processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id
    base_model.config.pad_token_id = processor.tokenizer.pad_token_id
    base_model.config.eos_token_id = processor.tokenizer.sep_token_id or processor.tokenizer.eos_token_id
    base_model.config.use_cache = True
    base_model.config.max_length = 20
    
    # Criar modelo com regularização
    model = RegularizedTrOCR(base_model, dropout_rate=DROPOUT)
    model.to(device)
    
    # Carregar state dict se resumindo
    if RESUME_TRAINING and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Gradient checkpointing
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
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Optimizer com novo learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # Scheduler - usar CosineAnnealingWarmRestarts para sair do plateau
    if USE_COSINE_SCHEDULER:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,  # Reiniciar a cada 5 épocas
            T_mult=2,  # Dobrar o período a cada restart
            eta_min=MIN_LR
        )
        logger.info("Using CosineAnnealingWarmRestarts scheduler")
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.7,  # Reduzir menos agressivamente
            patience=5,  # Mais paciência
            min_lr=MIN_LR,
            threshold=0.001  # Adicionar threshold
        )
        logger.info("Using ReduceLROnPlateau scheduler")
    
    # Loss function
    criterion = AdaptiveLabelSmoothingLoss(LABEL_SMOOTHING)
    
    # Variáveis de treinamento
    patience_counter = 0
    global_step = 0
    
    # Ajustes dinâmicos
    current_augment_prob = AUGMENT_PROB
    current_dropout = DROPOUT
    current_label_smoothing = LABEL_SMOOTHING
    
    logger.info("\n" + "="*60)
    logger.info("STARTING OPTIMIZED TRAINING TO BREAK PLATEAU")
    logger.info("="*60)
    
    # Loop principal
    epoch_pbar = tqdm(range(start_epoch, NUM_EPOCHS), desc="Training Progress", position=0)
    
    for epoch in epoch_pbar:
        epoch_start = time.time()
        
        # Ajustes adaptativos baseados na accuracy atual
        if USE_ADAPTIVE_TRAINING and best_accuracy > 0.85:  # Reduzido de 0.90
            # Ajustes mais agressivos para quebrar plateau
            if best_accuracy > 0.95:
                current_augment_prob = 0.1
                current_dropout = 0.05
                current_label_smoothing = 0.01
            elif best_accuracy > 0.90:  # Estamos aqui (91.72%)
                current_augment_prob = 0.2  # Mais augmentation
                current_dropout = 0.1  # Mais dropout
                current_label_smoothing = 0.03  # Mais smoothing
            elif best_accuracy > 0.85:
                current_augment_prob = 0.3
                current_dropout = 0.15
                current_label_smoothing = 0.05
            
            # Aplicar ajustes
            train_dataset.set_augment_prob(current_augment_prob)
            model.set_dropout(current_dropout)
            criterion.set_smoothing(current_label_smoothing)
            
            logger.info(f"\nAdaptive settings - Aug: {current_augment_prob:.3f}, DO: {current_dropout:.3f}, LS: {current_label_smoothing:.3f}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_loss_base = 0  # Loss sem regularização para comparação justa
        train_steps = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} - Training", position=1, leave=False)
        
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_pbar):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            
            # Calculate base loss (sem smoothing)
            base_loss = criterion(outputs.logits, labels, return_base_loss=True)
            train_loss_base += base_loss.item()
            
            # Calculate training loss (com smoothing)
            loss = criterion(outputs.logits, labels)
            
            # L2 Regularization - DESABILITADA ou MUITO REDUZIDA
            if USE_L2_REGULARIZATION and best_accuracy > 0.90:
                l2_lambda = 1e-9  # EXTREMAMENTE REDUZIDO
                l2_norm = 0
                for name, param in model.named_parameters():
                    if param.requires_grad and 'decoder' in name:
                        l2_norm += param.pow(2.0).sum()
                
                l2_reg = l2_lambda * l2_norm
                loss = loss + l2_reg
                
                if batch_idx % 100 == 0:
                    logger.debug(f"L2 regularization: {l2_reg.item():.8f}")
            
            # Normalize for accumulation
            loss = loss / ACCUMULATION_STEPS
            
            # Backward
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            train_loss += loss.item() * ACCUMULATION_STEPS
            train_steps += 1
            
            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({
                'loss': f'{loss.item() * ACCUMULATION_STEPS:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        avg_train_loss = train_loss / train_steps
        avg_train_loss_base = train_loss_base / train_steps
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        correct = 0
        total = 0
        correct_by_length = defaultdict(int)
        total_by_length = defaultdict(int)
        
        predictions_sample = []
        errors_sample = []
        
        max_val_batches = min(len(val_loader), VALIDATION_BATCHES)
        
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
                label_lengths = batch['label_length']
                
                # Calculate validation loss
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = criterion(outputs.logits, labels, return_base_loss=True)  # Base loss para comparação
                val_loss += loss.item()
                val_steps += 1
                
                # Generate predictions
                generated_ids = model.generate(
                    pixel_values,
                    max_new_tokens=16,
                    num_beams=5,  # Beam search para melhor accuracy
                    early_stopping=True,
                    no_repeat_ngram_size=0,
                    decoder_start_token_id=model.model.config.decoder_start_token_id,
                    do_sample=False  # Determinístico para validação
                )
                
                # Decode predictions
                pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Calculate accuracy
                for pred, true, length in zip(pred_texts, true_texts_batch, label_lengths):
                    pred_clean = pred.strip().upper()
                    true_clean = true.strip().upper()
                    
                    total += 1
                    total_by_length[length.item()] += 1
                    
                    if pred_clean == true_clean:
                        correct += 1
                        correct_by_length[length.item()] += 1
                    else:
                        # Registrar erro para análise
                        if USE_ERROR_ANALYSIS:
                            monitor.add_error(true_clean, pred_clean, epoch)
                            if len(errors_sample) < 10:
                                errors_sample.append((true_clean, pred_clean))
                    
                    # Coletar amostras
                    if len(predictions_sample) < 10 and batch_idx < 2:
                        predictions_sample.append((true_clean, pred_clean))
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct/total:.2%}' if total > 0 else '0%'
                })
        
        avg_val_loss = val_loss / val_steps
        accuracy = correct / total if total > 0 else 0
        
        # Atualizar monitor
        monitor.update(
            epoch=epoch,
            train_loss=avg_train_loss,
            train_loss_base=avg_train_loss_base,
            val_loss=avg_val_loss,
            accuracy=accuracy,
            learning_rate=optimizer.param_groups[0]['lr']
        )
        
        # Scheduler step
        old_lr = optimizer.param_groups[0]['lr']
        if USE_COSINE_SCHEDULER:
            scheduler.step()
        else:
            scheduler.step(accuracy)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                logger.info(f"📉 ReduceLROnPlateau: Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
        
        epoch_time = time.time() - epoch_start
        
        # Log resultados
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} ({epoch_time:.1f}s)")
        logger.info(f"Loss - Train (base): {avg_train_loss_base:.4f}, Train (w/ reg): {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
        logger.info(f"Gap (Val-Train base): {avg_val_loss - avg_train_loss_base:.4f}")
        logger.info(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Accuracy por comprimento
        logger.info("\nAccuracy by label length:")
        for length in sorted(total_by_length.keys())[:5]:  # Top 5 comprimentos
            if total_by_length[length] > 0:
                acc = correct_by_length[length] / total_by_length[length]
                logger.info(f"  Length {length}: {acc:.2%} ({correct_by_length[length]}/{total_by_length[length]})")
        
        # Análise de erros
        if USE_ERROR_ANALYSIS and errors_sample:
            error_summary = monitor.get_error_summary(epoch)
            if error_summary:
                logger.info("\nError analysis:")
                for error_type, count in sorted(error_summary.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"  {error_type}: {count}")
                
                logger.info("\nError samples:")
                for i, (true, pred) in enumerate(errors_sample[:5]):
                    logger.info(f"  True: '{true}' → Pred: '{pred}'")
        
        # Sample predictions
        if predictions_sample:
            logger.info("\nSample predictions:")
            for i, (true, pred) in enumerate(predictions_sample[:5]):
                status = "✓" if pred == true else "✗"
                logger.info(f"  {status} True: '{true}' → Pred: '{pred}'")
        
        # Save best model
        if accuracy > best_accuracy or (accuracy == best_accuracy and avg_val_loss < best_val_loss):
            improvement = accuracy - best_accuracy
            best_accuracy = accuracy
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.model.save_pretrained(OUTPUT_DIR)
            processor.save_pretrained(OUTPUT_DIR)
            
            # Salvar checkpoint detalhado
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy,
                'best_val_loss': best_val_loss,
                'hyperparameters': {
                    'augment_prob': current_augment_prob,
                    'dropout': current_dropout,
                    'label_smoothing': current_label_smoothing
                }
            }
            
            # Nome do arquivo com accuracy
            checkpoint_name = f'checkpoint_epoch{epoch+1}_acc{accuracy:.4f}.pth'
            torch.save(checkpoint, os.path.join(OUTPUT_DIR, checkpoint_name))
            
            # Salvar checkpoint principal
            torch.save(checkpoint, os.path.join(OUTPUT_DIR, 'checkpoint.pth'))
            
            # Salvar métricas
            monitor.update_best(accuracy=best_accuracy, val_loss=best_val_loss, epoch=epoch)
            monitor.save(os.path.join(OUTPUT_DIR, 'training_metrics.json'))
            
            logger.info(f"✅ New best model! Accuracy: {best_accuracy:.2%} (+{improvement:.2%})")
        else:
            patience_counter += 1
            logger.info(f"Patience counter: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                logger.info(f"\n⚠️ Early stopping triggered!")
                break
        
        # Análise especial para accuracy atual (91%)
        if 0.90 < accuracy < 0.93:
            logger.info("\n🎯 Breaking plateau analysis:")
            logger.info(f"  Current accuracy: {accuracy:.2%}")
            logger.info(f"  Gap from 92%: {0.92 - accuracy:.2%}")
            logger.info(f"  Current settings - Aug: {current_augment_prob:.3f}, DO: {current_dropout:.3f}, LS: {current_label_smoothing:.3f}")
            logger.info("  Strategies being applied:")
            logger.info("     - Increased augmentation diversity")
            logger.info("     - Higher learning rate with cosine restarts")
            logger.info("     - Reduced L2 regularization")
            logger.info("     - More aggressive label smoothing")
        
        # Se passou de 93%, dar feedback
        if accuracy > 0.93:
            logger.info("\n🚀 PLATEAU BROKEN! Now optimizing for higher accuracy...")
    
    # Resumo final
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Best accuracy: {best_accuracy:.2%}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved at: {OUTPUT_DIR}")
    
    # Análise final detalhada
    if len(monitor.metrics['accuracy']) > 0:
        final_acc = monitor.metrics['accuracy'][-1]
        
        # Sugestões baseadas no resultado final
        if final_acc > 0.98:
            logger.info("\n🎉 Exceptional results! Model is production-ready.")
        elif final_acc > 0.95:
            logger.info("\n✨ Excellent results! Consider:")
            logger.info("  - Ensemble with multiple checkpoints")
            logger.info("  - Test Time Augmentation")
            logger.info("  - Post-processing for common error patterns")
        elif final_acc > 0.92:
            logger.info("\n👍 Great progress! Successfully broke the plateau. Consider:")
            logger.info("  - Continue training with current settings")
            logger.info("  - Try trocr-large model")
            logger.info("  - More training data")
        elif final_acc > 0.90:
            logger.info("\n📈 Good attempt! To break plateau, also try:")
            logger.info("  - Even higher initial learning rate (5e-5)")
            logger.info("  - Different optimizer (e.g., Lion)")
            logger.info("  - MixUp or CutMix augmentation")
            logger.info("  - Ensemble training")

if __name__ == "__main__":
    try:
        train_anti_overfitting()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()