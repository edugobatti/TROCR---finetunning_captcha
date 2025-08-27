import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import random
import time
from tqdm import tqdm
import json
import glob
import cv2
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURAÇÃO DE SERIALIZAÇÃO DO PYTORCH
# ============================================
def setup_torch_serialization():
    """Configura a serialização do PyTorch para permitir carregar objetos NumPy de forma segura"""
    try:
        version = torch.__version__.split('.')
        major, minor = int(version[0]), int(version[1])
        
        if (major > 2) or (major == 2 and minor >= 6):
            logger.info(f"Detectado PyTorch {torch.__version__}: configurando serialização segura")
            
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals([
                    "numpy._core.multiarray._reconstruct",
                    "numpy.core.multiarray._reconstruct",
                    "numpy.ndarray",
                    "numpy.dtype",
                    "numpy._globals",
                    "_codecs.encode"
                ])
                logger.info("✅ Configuração de serialização segura concluída")
    except Exception as e:
        logger.warning(f"⚠️ Erro ao configurar serialização: {e}")

# ============================================
# CARREGAMENTO E CONVERSÃO DE IMAGENS
# ============================================
def load_and_convert_image(image_path, target_mode="RGB"):
    """Carrega imagem mantendo características originais do CAPTCHA"""
    try:
        image = Image.open(image_path)
        
        # Converter modos especiais
        if image.mode == "L;16":
            img_array = np.array(image, dtype=np.uint16)
            img_array = (img_array / 256).astype(np.uint8)
            image = Image.fromarray(img_array, mode='L')
        elif image.mode in ["I;16", "I", "F"]:
            img_array = np.array(image)
            min_val, max_val = img_array.min(), img_array.max()
            if max_val > min_val:
                img_array = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                img_array = np.zeros_like(img_array, dtype=np.uint8)
            image = Image.fromarray(img_array, mode='L')
        elif image.mode == "P":
            image = image.convert("RGBA")
        elif image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3] if len(image.split()) > 3 else None)
            image = background
        
        # Converter para RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
        
    except Exception as e:
        logger.error(f"Erro ao carregar imagem {image_path}: {e}")
        raise

# ============================================
# DATASET COM AUGMENTATION AVANÇADO
# ============================================
class ComplexCaptchaDataset(Dataset):
    def __init__(self, image_paths, labels, processor, mode='train', 
                 augment_prob=0.85):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.mode = mode
        self.augment_prob = augment_prob
        
        if mode == 'train':
            self._validate_images()
    
    def _validate_images(self):
        """Valida amostras do dataset"""
        logger.info("Validando amostras do dataset...")
        sample_size = min(50, len(self.image_paths))
        problematic = []
        
        for idx in tqdm(range(sample_size), desc="Validando", leave=False):
            try:
                img = load_and_convert_image(self.image_paths[idx])
                img.close()
            except Exception as e:
                problematic.append((self.image_paths[idx], str(e)))
        
        if problematic:
            logger.warning(f"⚠️ {len(problematic)} imagens problemáticas em {sample_size} amostras")
    
    def augment_image_advanced(self, image):
        """Augmentation avançado para CAPTCHAs complexos"""
        if self.mode != 'train' or random.random() > self.augment_prob:
            return image
        
        # Distorção elástica (para CAPTCHAs distorcidos)
        if random.random() < 0.4:
            image = self.elastic_transform(image)
        
        # Rotação mais sutil
        if random.random() < 0.6:
            angle = random.uniform(-5, 5)
            image = image.rotate(angle, fillcolor=(255, 255, 255), expand=False)
        
        # Perspectiva
        if random.random() < 0.3:
            image = self.perspective_transform(image)
        
        # Zoom adaptativo
        if random.random() < 0.5:
            zoom = random.uniform(0.9, 1.1)
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
        
        # Ajustes de cor mais conservadores
        if random.random() < 0.6:
            brightness = ImageEnhance.Brightness(image)
            image = brightness.enhance(random.uniform(0.8, 1.2))
            
            contrast = ImageEnhance.Contrast(image)
            image = contrast.enhance(random.uniform(0.8, 1.2))
            
            if random.random() < 0.3:
                saturation = ImageEnhance.Color(image)
                image = saturation.enhance(random.uniform(0.8, 1.2))
        
        # Blur sutil
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
        
        # Sharpen ocasional
        if random.random() < 0.2:
            sharpness = ImageEnhance.Sharpness(image)
            image = sharpness.enhance(random.uniform(1.2, 1.5))
        
        # Ruído mais controlado
        if random.random() < 0.4:
            pixels = np.array(image)
            noise_std = random.uniform(3, 8)
            noise = np.random.normal(0, noise_std, pixels.shape)
            pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(pixels)
        
        return image
    
    def elastic_transform(self, image, alpha=20, sigma=3):
        """Aplica transformação elástica"""
        img_array = np.array(image)
        shape = img_array.shape[:2]
        
        dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        x = (x + dx).astype(np.float32)
        y = (y + dy).astype(np.float32)
        
        transformed = cv2.remap(img_array, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(transformed)
    
    def perspective_transform(self, image):
        """Aplica transformação de perspectiva"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Pontos de origem
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        # Pontos de destino com pequena distorção
        offset = 10
        pts2 = np.float32([
            [random.randint(0, offset), random.randint(0, offset)],
            [w - random.randint(0, offset), random.randint(0, offset)],
            [random.randint(0, offset), h - random.randint(0, offset)],
            [w - random.randint(0, offset), h - random.randint(0, offset)]
        ])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed = cv2.warpPerspective(img_array, matrix, (w, h), 
                                         borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(transformed)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        try:
            # Carregar imagem sem pré-processamento
            image = load_and_convert_image(self.image_paths[idx], "RGB")
            
            # Aplicar augmentation
            if self.mode == 'train':
                image = self.augment_image_advanced(image)
            
            # Processar
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
            
            # Label
            label_text = self.labels[idx].upper()
            encoding = self.processor.tokenizer(
                label_text,
                padding="max_length",
                max_length=20,  # Aumentado para CAPTCHAs mais longos
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
            logger.error(f"Erro ao processar imagem {self.image_paths[idx]}: {e}")
            next_idx = (idx + 1) % len(self.labels)
            if next_idx == idx:
                raise RuntimeError(f"Não foi possível processar nenhuma imagem")
            return self.__getitem__(next_idx)

# ============================================
# LOSS FUNCTION MELHORADA
# ============================================
class EnhancedWeightedLoss(nn.Module):
    def __init__(self, processor, smoothing=0.1, similarity_weight=2.5):
        super().__init__()
        self.processor = processor
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.similarity_weight = similarity_weight
        
        # Grupos de caracteres similares expandidos
        self.similar_groups = [
            # Números vs letras
            ['0', 'O', 'o', 'Q', 'D'],
            ['1', 'I', 'l', '|', 'i', 'j'],
            ['2', 'Z', 'z'],
            ['3', 'E'],
            ['4', 'A'],
            ['5', 'S', 's'],
            ['6', 'G', 'b', 'd'],
            ['7', 'T', 'J'],
            ['8', 'B'],
            ['9', 'g', 'q', 'p'],
            
            # Letras muito similares
            ['I', 'l', '1', '|', 'i'],
            ['O', '0', 'o', 'Q', 'D'],
            ['C', 'c', 'G', '(', '['],
            ['P', 'p', 'R'],
            ['U', 'u', 'V', 'v', 'Y'],
            ['W', 'w', 'M', 'N'],
            ['n', 'h', 'r', 'm'],
            ['m', 'rn', 'nn'],
            ['cl', 'd'],
            ['ti', 'H'],
            ['vv', 'w', 'W'],
            ['rn', 'm'],
            
            # Maiúsculas vs minúsculas
            *[[chr(i), chr(i+32)] for i in range(65, 91)],
        ]
        
        # Criar mapeamento
        self.similarity_map = {}
        for group in self.similar_groups:
            for char1 in group:
                for char2 in group:
                    if char1 != char2:
                        self.similarity_map[(char1, char2)] = True
        
        logger.info(f"✓ Loss configurado com {len(self.similarity_map)} pares similares")
    
    def get_similarity_weight(self, predicted_text, true_text):
        """Calcular peso com penalização progressiva"""
        if len(predicted_text) != len(true_text):
            return 1.5  # Penalização maior para erros de comprimento
        
        total_weight = 0.0
        error_count = 0
        
        for i, (pred_char, true_char) in enumerate(zip(predicted_text, true_text)):
            if pred_char != true_char:
                error_count += 1
                
                # Peso baseado na posição (erros no início são mais graves)
                position_weight = 1.0 + (0.2 * (1 - i/len(true_text)))
                
                # Verificar similaridade
                if (pred_char, true_char) in self.similarity_map:
                    total_weight += self.similarity_weight * position_weight
                else:
                    total_weight += 1.0 * position_weight
        
        return total_weight / max(error_count, 1)
    
    def forward(self, logits, targets, pixel_values=None):
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        
        mask = targets != -100
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        logits = logits[mask]
        targets = targets[mask]
        
        # Loss com label smoothing
        log_probs = torch.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        return loss.mean()

# ============================================
# LEARNING RATE SCHEDULER CUSTOMIZADO
# ============================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Warmup phase
            scale = self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + np.cos(np.pi * progress))
        
        for idx, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[idx]
            new_lr = self.min_lr + (base_lr - self.min_lr) * scale
            group['lr'] = new_lr
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

# ============================================
# FUNÇÃO PRINCIPAL DE RE-TREINO
# ============================================
def retrain_complex_captcha():
    # Configurar serialização
    setup_torch_serialization()
    
    # ====== CONFIGURAÇÕES AJUSTADAS PARA RE-TREINO ======
    BASE_MODEL_PATH = "./best_model"  # Modelo base treinado
    CAPTCHA_FOLDER = "./trocr-camada-3"      # Nova base complexa
    OUTPUT_DIR = "./trocr-camada-1.2"          # Novo diretório de saída
    
   # Hiperparâmetros ultra-agressivos para fine-tuning em dataset pequeno

    BATCH_SIZE = 4                      # Batch físico pequeno para overfitting controlado
    GRADIENT_ACCUMULATION = 16          # Simula batch virtual gigante para estabilidade
    LEARNING_RATE = 5e-5                # Alto para aprender rápido
    MIN_LR = 1e-8
    WEIGHT_DECAY = 0.002                # Quase sem restrição
    
    NUM_EPOCHS = 200                    # Muitas épocas para explorar
    PATIENCE = 60                       # Early stopping bem tolerante
    WARMUP_RATIO = 0.3                  # Warmup longo para não explodir no início
    
    LABEL_SMOOTHING = 0.2               # Muito smoothing para não decorar
    DROPOUT = 0.5                       # Dropout muito forte
    AUGMENT_PROB = 1.0                  # Sempre aplica augmentation
    SIMILARITY_WEIGHT = 6.0          # Penaliza fortemente erros parecidos
    
    # Otimização
    GRADIENT_ACCUMULATION = 8      # Mais acumulação para simular batch maior e estabilizar

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("="*70)
    logger.info("RE-TREINO PARA CAPTCHAS COMPLEXOS - CAMADA 2")
    logger.info("="*70)
    logger.info(f"🔧 Device: {device}")
    logger.info(f"📂 Modelo base: {BASE_MODEL_PATH}")
    logger.info(f"📂 Dataset: {CAPTCHA_FOLDER}")
    logger.info(f"📂 Output: {OUTPUT_DIR}")
    logger.info(f"⚙️ Batch size: {BATCH_SIZE} (x{GRADIENT_ACCUMULATION} accumulation)")
    logger.info(f"⚙️ Learning rate: {LEARNING_RATE} → {MIN_LR}")
    logger.info(f"⚙️ Epochs: {NUM_EPOCHS}")
    logger.info(f"⚙️ Augmentation: {AUGMENT_PROB*100:.0f}%")
    
    # Criar diretório de saída
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
    
    # ====== CARREGAR MODELO BASE ======
    logger.info("\n" + "="*70)
    logger.info("CARREGANDO MODELO BASE TREINADO")
    logger.info("="*70)
    
    try:
        processor = TrOCRProcessor.from_pretrained(BASE_MODEL_PATH)
        model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_PATH)
        logger.info("✅ Modelo base carregado com sucesso")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo base: {e}")
        logger.info("Carregando modelo padrão do HuggingFace...")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-str")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-str")
    
    # Configurar model
    model.config.use_cache = True
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    
    # Configurar dropout
    if hasattr(model.decoder, 'model') and hasattr(model.decoder.model, 'decoder'):
        for layer in model.decoder.model.decoder.layers:
            layer.dropout = DROPOUT
            if hasattr(layer, 'self_attn'):
                layer.self_attn.dropout = DROPOUT
            if hasattr(layer, 'encoder_attn'):
                layer.encoder_attn.dropout = DROPOUT
    
    model.to(device)
    model.gradient_checkpointing_enable()
    
    # ====== CARREGAR DATASET ======
    logger.info("\n" + "="*70)
    logger.info("CARREGANDO DATASET COMPLEXO")
    logger.info("="*70)
    
    image_paths = []
    labels = []
    file_stats = {'total': 0, 'valid': 0, 'invalid': 0}
    
    for filename in os.listdir(CAPTCHA_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_stats['total'] += 1
            image_path = os.path.join(CAPTCHA_FOLDER, filename)
            label = os.path.splitext(filename)[0]
            
            # Validação mais flexível para CAPTCHAs complexos
            if 1 <= len(label) <= 25:  # Permitir labels mais longos
                try:
                    # Teste de carregamento
                    test_img = load_and_convert_image(image_path, "RGB")
                    test_img.close()
                    
                    image_paths.append(image_path)
                    labels.append(label)
                    file_stats['valid'] += 1
                except Exception as e:
                    file_stats['invalid'] += 1
                    if file_stats['invalid'] <= 5:
                        logger.warning(f"Imagem inválida: {filename} - {e}")
            else:
                file_stats['invalid'] += 1
    
    logger.info(f"📊 Estatísticas do dataset:")
    logger.info(f"   Total de arquivos: {file_stats['total']}")
    logger.info(f"   Válidos: {file_stats['valid']}")
    logger.info(f"   Inválidos: {file_stats['invalid']}")
    
    if len(image_paths) == 0:
        logger.error("❌ Nenhuma imagem válida encontrada!")
        return
    
    # Adicionar novos tokens se necessário
    all_chars = set(''.join(labels))
    vocab = processor.tokenizer.get_vocab()
    new_tokens = [c for c in all_chars if c not in vocab]
    
    if new_tokens:
        processor.tokenizer.add_tokens(new_tokens)
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        logger.info(f"✅ Adicionados {len(new_tokens)} novos tokens ao vocabulário")
    
    # Split do dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.15, random_state=42
    )
    
    logger.info(f"📊 Split do dataset:")
    logger.info(f"   Treino: {len(train_paths)} imagens")
    logger.info(f"   Validação: {len(val_paths)} imagens")
    
    # ====== CRIAR DATASETS E DATALOADERS ======
    train_dataset = ComplexCaptchaDataset(
        train_paths, train_labels, processor,
        mode='train', augment_prob=AUGMENT_PROB
    )
    
    val_dataset = ComplexCaptchaDataset(
        val_paths, val_labels, processor,
        mode='val', augment_prob=0
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2,
        shuffle=False, num_workers=0
    )
    
    # ====== CONFIGURAR OTIMIZAÇÃO ======
    # Usar diferentes learning rates para encoder e decoder
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': LEARNING_RATE * 0.1},  # LR menor para encoder
        {'params': decoder_params, 'lr': LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY)
    
    # Scheduler
    total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, MIN_LR)
    
    # Loss function
    criterion = EnhancedWeightedLoss(processor, LABEL_SMOOTHING, SIMILARITY_WEIGHT)
    
    # ====== TREINAMENTO ======
    logger.info("\n" + "="*70)
    logger.info("INICIANDO RE-TREINO")
    logger.info("="*70)
    
    best_accuracy = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    accuracies = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # ====== FASE DE TREINO ======
        model.train()
        train_loss = 0
        accumulation_counter = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training")
        
        for batch_idx, batch in enumerate(train_pbar):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = criterion(outputs.logits, labels, pixel_values)
            
            # Normalizar loss pela acumulação de gradiente
            loss = loss / GRADIENT_ACCUMULATION
            loss.backward()
            
            accumulation_counter += 1
            
            # Atualizar pesos após acumulação
            if accumulation_counter % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * GRADIENT_ACCUMULATION
            
            # Atualizar progress bar
            current_lr = scheduler.get_lr()[0]
            train_pbar.set_postfix({
                'loss': f'{loss.item()*GRADIENT_ACCUMULATION:.4f}',
                'avg_loss': f'{train_loss/(batch_idx+1):.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        train_pbar.close()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # ====== FASE DE VALIDAÇÃO ======
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        char_correct = 0
        char_total = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation")
        
        with torch.no_grad():
            for batch in val_pbar:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                # Loss
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = criterion(outputs.logits, labels, pixel_values)
                val_loss += loss.item()
                
                # Generate
                generated_ids = model.generate(
                    pixel_values,
                    max_new_tokens=20,
                    num_beams=5,
                    length_penalty=0.8,
                    early_stopping=True
                )
                
                # Decode
                pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                labels_clean = labels.clone()
                labels_clean[labels_clean == -100] = processor.tokenizer.pad_token_id
                true_texts = processor.batch_decode(labels_clean, skip_special_tokens=True)
                
                # Calculate accuracies
                for pred, true in zip(pred_texts, true_texts):
                    pred = pred.strip().upper()
                    true = true.strip().upper()
                    
                    total += 1
                    if pred == true:
                        correct += 1
                    
                    # Character-level accuracy
                    for p_char, t_char in zip(pred, true):
                        char_total += 1
                        if p_char == t_char:
                            char_correct += 1
                    char_total += abs(len(pred) - len(true))  # Penalizar diferença de tamanho
                
                # Atualizar progress bar
                current_acc = correct / total if total > 0 else 0
                char_acc = char_correct / char_total if char_total > 0 else 0
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.2%}',
                    'char_acc': f'{char_acc:.2%}'
                })
        
        val_pbar.close()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        accuracy = correct / total
        char_accuracy = char_correct / char_total if char_total > 0 else 0
        accuracies.append(accuracy)
        
        epoch_time = time.time() - epoch_start
        
        # ====== LOG METRICS ======
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 Epoch {epoch+1}/{NUM_EPOCHS} ({epoch_time:.1f}s)")
        logger.info(f"📉 Loss - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
        logger.info(f"📈 Accuracy - Full: {accuracy:.2%}, Character: {char_accuracy:.2%}")
        logger.info(f"🔧 Learning Rate: {scheduler.get_lr()[0]:.2e}")
        
        # ====== SALVAR CHECKPOINT ======
        if (epoch + 1) % 5 == 0 or accuracy > best_accuracy:
            checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoints", f"checkpoint_epoch_{epoch+1}")
            model.save_pretrained(checkpoint_path)
            processor.save_pretrained(checkpoint_path)
            
            # Salvar métricas
            metrics = {
                'epoch': epoch + 1,
                'accuracy': accuracy,
                'char_accuracy': char_accuracy,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': scheduler.get_lr()[0]
            }
            
            with open(os.path.join(checkpoint_path, "metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"💾 Checkpoint salvo: {checkpoint_path}")
        
        # ====== SALVAR MELHOR MODELO ======
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            
            best_model_path = os.path.join(OUTPUT_DIR, "best_model")
            model.save_pretrained(best_model_path)
            processor.save_pretrained(best_model_path)
            
            logger.info(f"🏆 Novo melhor modelo: {best_accuracy:.2%}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"\n⏹️ Early stopping após {PATIENCE} epochs sem melhoria")
                break
        
        # ====== EXEMPLOS DE PREDIÇÃO ======
        if (epoch + 1) % 10 == 0 or accuracy > best_accuracy - 0.01:
            logger.info("\n📝 Exemplos de predições:")
            model.eval()
            with torch.no_grad():
                for i in range(min(5, len(val_dataset))):
                    item = val_dataset[i]
                    pixel_values = item['pixel_values'].unsqueeze(0).to(device)
                    
                    generated_ids = model.generate(
                        pixel_values, 
                        max_new_tokens=20,
                        num_beams=5
                    )
                    
                    pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    true_text = val_labels[i]
                    
                    status = "✅" if pred_text.upper() == true_text.upper() else "❌"
                    logger.info(f"  {status} True: '{true_text}' → Pred: '{pred_text}'")
    
    # ====== ANÁLISE FINAL ======
    logger.info("\n" + "="*70)
    logger.info("🎯 RE-TREINO COMPLETO!")
    logger.info("="*70)
    logger.info(f"📊 Melhor acurácia: {best_accuracy:.2%}")
    logger.info(f"📂 Melhor modelo salvo em: {os.path.join(OUTPUT_DIR, 'best_model')}")
    
    # Salvar histórico de treinamento
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'accuracies': accuracies,
        'best_accuracy': best_accuracy,
        'config': {
            'base_model': BASE_MODEL_PATH,
            'dataset': CAPTCHA_FOLDER,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'epochs_trained': len(train_losses),
            'augmentation_prob': AUGMENT_PROB
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"📊 Histórico salvo em: {os.path.join(OUTPUT_DIR, 'training_history.json')}")
    
    # Análise de overfitting
    if len(train_losses) > 0 and len(val_losses) > 0:
        final_gap = val_losses[-1] - train_losses[-1]
        logger.info(f"\n📈 Análise de Overfitting:")
        logger.info(f"   Gap final (Val-Train): {final_gap:.4f}")
        
        if final_gap < 0.3:
            logger.info("   ✅ Excelente! Overfitting mínimo")
        elif final_gap < 0.6:
            logger.info("   ✅ Bom! Overfitting controlado")
        elif final_gap < 1.0:
            logger.info("   ⚠️ Atenção! Overfitting moderado")
        else:
            logger.info("   ❌ Overfitting significativo detectado")

if __name__ == "__main__":
    try:
        # Instalar dependências se necessário
        try:
            import cv2
        except ImportError:
            logger.info("Instalando opencv-python...")
            os.system("pip install opencv-python")
            import cv2
        
        retrain_complex_captcha()
    except KeyboardInterrupt:
        logger.info("\n⏹️ Treinamento interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()