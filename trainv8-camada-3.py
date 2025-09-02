import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
import copy
import cv2
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
                logger.info(" Configuração de serialização segura concluída")
    except Exception as e:
        logger.warning(f" Erro ao configurar serialização: {e}")

# ============================================
# ELASTIC WEIGHT CONSOLIDATION (EWC)
# ============================================
class EWC:
    """Implementa Elastic Weight Consolidation para preservar conhecimento anterior"""
    
    def __init__(self, model, dataloader, device, fisher_estimation_sample_size=200):
        self.model = model
        self.device = device
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._estimate_fisher(dataloader, fisher_estimation_sample_size)
        
    def _estimate_fisher(self, dataloader, sample_size):
        """Estima a matriz de Fisher para importância dos parâmetros"""
        fisher = {}
        self.model.eval()
        
        sample_count = 0
        for batch in tqdm(dataloader, desc="Estimando Fisher Matrix", leave=False):
            if sample_count >= sample_size:
                break
                
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if n not in fisher:
                        fisher[n] = p.grad.data.clone().detach() ** 2
                    else:
                        fisher[n] += p.grad.data.clone().detach() ** 2
            
            sample_count += len(pixel_values)
        
        # Normalizar Fisher
        for n in fisher:
            fisher[n] /= sample_count
            
        self.model.train()
        return fisher
    
    def penalty(self, model):
        """Calcula penalidade EWC"""
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return loss

# ============================================
# KNOWLEDGE DISTILLATION
# ============================================
class KnowledgeDistillationLoss(nn.Module):
    """Loss de destilação para preservar conhecimento do modelo anterior"""
    
    def __init__(self, temperature=3.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, labels, criterion):
        """
        Combina loss de destilação com loss supervisionado
        alpha: peso para loss de destilação
        (1-alpha): peso para loss supervisionado
        """
        # Loss supervisionado (novo dataset)
        hard_loss = criterion(student_logits, labels)
        
        # Loss de destilação (preservar conhecimento)
        soft_targets = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = self.kl_div(soft_targets, soft_teacher) * (self.temperature ** 2)
        
        # Combinar losses
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, hard_loss, soft_loss

# ============================================
# REPLAY BUFFER PARA REHEARSAL
# ============================================
class ReplayBuffer:
    """Buffer para armazenar exemplos antigos e fazer rehearsal"""
    
    def __init__(self, max_size=500):
        self.max_size = max_size
        self.buffer = []
        
    def add(self, image_path, label):
        """Adiciona exemplo ao buffer"""
        if len(self.buffer) >= self.max_size:
            # Remover aleatoriamente para manter tamanho
            idx = random.randint(0, len(self.buffer) - 1)
            self.buffer.pop(idx)
        self.buffer.append((image_path, label))
    
    def sample(self, n):
        """Amostra n exemplos do buffer"""
        if len(self.buffer) == 0:
            return [], []
        n = min(n, len(self.buffer))
        samples = random.sample(self.buffer, n)
        paths, labels = zip(*samples)
        return list(paths), list(labels)
    
    def get_all(self):
        """Retorna todos os exemplos do buffer"""
        if len(self.buffer) == 0:
            return [], []
        paths, labels = zip(*self.buffer)
        return list(paths), list(labels)

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
# DATASET COM MIXUP E AUGMENTATION
# ============================================
class IncrementalCaptchaDataset(Dataset):
    """Dataset com suporte para mixup e preservação de conhecimento"""
    
    def __init__(self, image_paths, labels, processor, mode='train', 
                 augment_prob=0.7, mixup_alpha=0.2):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.mode = mode
        self.augment_prob = augment_prob
        self.mixup_alpha = mixup_alpha
        
    def mixup_data(self, x1, x2, alpha=0.2):
        """Aplica mixup entre duas imagens"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        mixed = lam * x1 + (1 - lam) * x2
        return mixed, lam
    
    def augment_image_conservative(self, image):
        """Augmentation mais conservador para não distorcer demais"""
        if self.mode != 'train' or random.random() > self.augment_prob:
            return image
        
        # Rotação muito sutil
        if random.random() < 0.3:
            angle = random.uniform(-3, 3)
            image = image.rotate(angle, fillcolor=(255, 255, 255), expand=False)
        
        # Zoom leve
        if random.random() < 0.3:
            zoom = random.uniform(0.95, 1.05)
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
        
        # Ajustes de cor conservadores
        if random.random() < 0.4:
            brightness = ImageEnhance.Brightness(image)
            image = brightness.enhance(random.uniform(0.9, 1.1))
            
            contrast = ImageEnhance.Contrast(image)
            image = contrast.enhance(random.uniform(0.9, 1.1))
        
        # Ruído muito leve
        if random.random() < 0.2:
            pixels = np.array(image)
            noise_std = random.uniform(2, 5)
            noise = np.random.normal(0, noise_std, pixels.shape)
            pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(pixels)
        
        return image
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        try:
            # Carregar imagem principal
            image = load_and_convert_image(self.image_paths[idx], "RGB")
            
            # Aplicar augmentation conservador
            if self.mode == 'train':
                image = self.augment_image_conservative(image)
            
            # Processar
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
            
            # Label
            label_text = self.labels[idx].upper()
            encoding = self.processor.tokenizer(
                label_text,
                padding="max_length",
                max_length=20,
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
            return self.__getitem__(next_idx)

# ============================================
# PROGRESSIVE NEURAL NETWORKS LITE
# ============================================
class ProgressiveAdapter(nn.Module):
    """Adaptador progressivo para adicionar capacidade sem esquecer"""
    
    def __init__(self, input_dim, adapter_dim=64):
        super().__init__()
        self.down_project = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Inicialização especial para começar com identidade
        nn.init.zeros_(self.down_project.weight)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
        
    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        x = self.layer_norm(x + residual)
        return x

# ============================================
# LOSS FUNCTION COM REGULARIZAÇÃO L2
# ============================================
class IncrementalLoss(nn.Module):
    def __init__(self, processor, smoothing=0.1, l2_lambda=0.01):
        super().__init__()
        self.processor = processor
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.l2_lambda = l2_lambda
        
    def forward(self, logits, targets, old_params=None, new_params=None):
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
        
        # Adicionar regularização L2 se parâmetros antigos fornecidos
        if old_params is not None and new_params is not None:
            l2_loss = 0
            for old_p, new_p in zip(old_params, new_params):
                l2_loss += ((new_p - old_p) ** 2).sum()
            loss = loss.mean() + self.l2_lambda * l2_loss
        else:
            loss = loss.mean()
        
        return loss

# ============================================
# FUNÇÃO PRINCIPAL DE TREINO INCREMENTAL
# ============================================
def incremental_training():
    # Configurar serialização
    setup_torch_serialization()
    
    # ====== CONFIGURAÇÕES PARA APRENDIZADO INCREMENTAL ======
    BASE_MODEL_PATH = "./trocr-incremental-9/best_model"           # Modelo base treinado
    OLD_DATASET_PATH = "./captcha-full"            # Dataset original (para replay)
    NEW_DATASET_PATH = "./img_base_captcha"      # Novo dataset complexo
    OUTPUT_DIR = "./trocr-incremental-10"         # Diretório de saída
    
    # Hiperparâmetros otimizados para preservar conhecimento
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 64
    LEARNING_RATE = 1.5e-4                       # LR bem menor para não esquecer
    MIN_LR = 1e-7
    WEIGHT_DECAY = 0.01
    
    NUM_EPOCHS = 100
    PATIENCE = 30
    WARMUP_RATIO = 0.1
    
    # Parâmetros de preservação de conhecimento
    USE_EWC = True                             # Elastic Weight Consolidation
    EWC_LAMBDA = 0.8                           # Importância do conhecimento anterior
    
    USE_DISTILLATION = True                    # Knowledge Distillation
    DISTILLATION_ALPHA = 0.6                   # Balance entre novo e antigo
    DISTILLATION_TEMPERATURE = 4.0
    
    USE_REPLAY = True                          # Experience Replay
    REPLAY_RATIO = 0.3                         # treino 40/60 40 antigas ,60 novas (%)
    REPLAY_BUFFER_SIZE = 2000
    
    LABEL_SMOOTHING = 0.02
    AUGMENT_PROB = 0.9                        # Menos augmentation
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("="*70)
    logger.info("TREINAMENTO INCREMENTAL - PRESERVANDO CONHECIMENTO")
    logger.info("="*70)
    logger.info(f" Device: {device}")
    logger.info(f" Modelo base: {BASE_MODEL_PATH}")
    logger.info(f" Dataset antigo: {OLD_DATASET_PATH}")
    logger.info(f" Dataset novo: {NEW_DATASET_PATH}")
    logger.info(f" Output: {OUTPUT_DIR}")
    logger.info(f" Batch size: {BATCH_SIZE} (x{GRADIENT_ACCUMULATION} accumulation)")
    logger.info(f" Learning rate: {LEARNING_RATE}")
    logger.info(f" EWC: {USE_EWC} (λ={EWC_LAMBDA})")
    logger.info(f" Distillation: {USE_DISTILLATION} (α={DISTILLATION_ALPHA})")
    logger.info(f" Replay: {USE_REPLAY} (ratio={REPLAY_RATIO})")
    
    # Criar diretórios
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
    
    # ====== CARREGAR MODELO BASE E CRIAR CÓPIA PARA TEACHER ======
    logger.info("\n" + "="*70)
    logger.info("CARREGANDO MODELOS")
    logger.info("="*70)
    
    try:
        processor = TrOCRProcessor.from_pretrained(BASE_MODEL_PATH)
        student_model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_PATH)
        logger.info(" Modelo base carregado com sucesso")
        
        # Criar modelo teacher (frozen)
        if USE_DISTILLATION:
            teacher_model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_PATH)
            teacher_model.to(device)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            logger.info(" Modelo teacher criado e congelado")
        else:
            teacher_model = None
            
    except Exception as e:
        logger.error(f"Erro ao carregar modelo base: {e}")
        return
    
    # Configurar student model
    student_model.config.use_cache = True
    student_model.to(device)
    
    # ====== CARREGAR DATASETS ======
    logger.info("\n" + "="*70)
    logger.info("CARREGANDO DATASETS")
    logger.info("="*70)
    
    # Carregar dataset antigo para replay
    old_image_paths = []
    old_labels = []
    
    if USE_REPLAY and os.path.exists(OLD_DATASET_PATH):
        for filename in os.listdir(OLD_DATASET_PATH):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(OLD_DATASET_PATH, filename)
                label = os.path.splitext(filename)[0]
                if 3 <= len(label) <= 10:
                    old_image_paths.append(image_path)
                    old_labels.append(label)
        
        logger.info(f"Dataset antigo: {len(old_image_paths)} imagens")
    
    # Carregar novo dataset
    new_image_paths = []
    new_labels = []
    
    for filename in os.listdir(NEW_DATASET_PATH):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(NEW_DATASET_PATH, filename)
            label = os.path.splitext(filename)[0]
            if 1 <= len(label) <= 25:
                new_image_paths.append(image_path)
                new_labels.append(label)
    
    logger.info(f"Dataset novo: {len(new_image_paths)} imagens")
    
    # Criar replay buffer
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    
    # Adicionar amostras antigas ao buffer
    if USE_REPLAY and len(old_image_paths) > 0:
        # Amostrar subset do dataset antigo
        n_old_samples = min(len(old_image_paths), int(len(new_image_paths) * REPLAY_RATIO))
        old_indices = random.sample(range(len(old_image_paths)), n_old_samples)
        
        for idx in old_indices:
            replay_buffer.add(old_image_paths[idx], old_labels[idx])
        
        logger.info(f" Replay buffer: {len(replay_buffer.buffer)} exemplos antigos")
    
    # Combinar datasets
    if USE_REPLAY and len(replay_buffer.buffer) > 0:
        replay_paths, replay_labels = replay_buffer.get_all()
        combined_paths = new_image_paths + replay_paths
        combined_labels = new_labels + replay_labels
        logger.info(f"Dataset combinado: {len(combined_paths)} imagens total")
    else:
        combined_paths = new_image_paths
        combined_labels = new_labels
    
    # Adicionar novos tokens se necessário
    all_chars = set(''.join(combined_labels))
    vocab = processor.tokenizer.get_vocab()
    new_tokens = [c for c in all_chars if c not in vocab]
    
    if new_tokens:
        processor.tokenizer.add_tokens(new_tokens)
        student_model.decoder.resize_token_embeddings(len(processor.tokenizer))
        if teacher_model:
            teacher_model.decoder.resize_token_embeddings(len(processor.tokenizer))
        logger.info(f" Adicionados {len(new_tokens)} novos tokens")
    
    # Split do dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        combined_paths, combined_labels, test_size=0.15, random_state=42
    )
    
    logger.info(f"Split final:")
    logger.info(f"   Treino: {len(train_paths)} imagens")
    logger.info(f"   Validação: {len(val_paths)} imagens")
    
    # ====== CALCULAR EWC SE NECESSÁRIO ======
    ewc = None
    if USE_EWC and len(old_image_paths) > 0:
        logger.info("\n Calculando Fisher Information Matrix para EWC...")
        
        # Criar dataset temporário com dados antigos
        ewc_dataset = IncrementalCaptchaDataset(
            old_image_paths[:200], old_labels[:200], processor, mode='val'
        )
        ewc_loader = DataLoader(ewc_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        ewc = EWC(student_model, ewc_loader, device)
        logger.info(" EWC configurado")
    
    # ====== CRIAR DATASETS E DATALOADERS ======
    train_dataset = IncrementalCaptchaDataset(
        train_paths, train_labels, processor,
        mode='train', augment_prob=AUGMENT_PROB
    )
    
    val_dataset = IncrementalCaptchaDataset(
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
    # Diferentes learning rates para diferentes partes
    encoder_params = []
    decoder_params = []
    
    for name, param in student_model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': LEARNING_RATE * 0.1},  # Encoder aprende mais devagar
        {'params': decoder_params, 'lr': LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY)
    
    # Loss functions
    base_criterion = IncrementalLoss(processor, LABEL_SMOOTHING)
    
    if USE_DISTILLATION:
        kd_criterion = KnowledgeDistillationLoss(
            temperature=DISTILLATION_TEMPERATURE,
            alpha=DISTILLATION_ALPHA
        )
    
    # ====== TREINAMENTO ======
    logger.info("\n" + "="*70)
    logger.info("INICIANDO TREINAMENTO INCREMENTAL")
    logger.info("="*70)
    
    best_accuracy = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    accuracies = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # ====== FASE DE TREINO ======
        student_model.train()
        train_loss = 0
        ewc_loss_total = 0
        kd_loss_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training")
        
        for batch_idx, batch in enumerate(train_pbar):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward no student
            outputs = student_model(pixel_values=pixel_values, labels=labels)
            
            # Loss base
            loss = base_criterion(outputs.logits, labels)
            
            # Knowledge Distillation
            if USE_DISTILLATION and teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(pixel_values=pixel_values, labels=labels)
                
                kd_loss, hard_loss, soft_loss = kd_criterion(
                    outputs.logits, teacher_outputs.logits, labels, base_criterion
                )
                loss = kd_loss
                kd_loss_total += soft_loss.item()
            
            # EWC penalty
            if USE_EWC and ewc is not None:
                ewc_penalty = ewc.penalty(student_model)
                loss = loss + EWC_LAMBDA * ewc_penalty
                ewc_loss_total += ewc_penalty.item()
            
            # Backward
            loss = loss / GRADIENT_ACCUMULATION
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * GRADIENT_ACCUMULATION
            
            # Update progress bar
            postfix = {'loss': f'{loss.item()*GRADIENT_ACCUMULATION:.4f}'}
            if USE_EWC and ewc_loss_total > 0:
                postfix['ewc'] = f'{ewc_loss_total/(batch_idx+1):.4f}'
            if USE_DISTILLATION and kd_loss_total > 0:
                postfix['kd'] = f'{kd_loss_total/(batch_idx+1):.4f}'
            train_pbar.set_postfix(postfix)
        
        train_pbar.close()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # ====== FASE DE VALIDAÇÃO ======
        student_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        # Separar métricas para dados novos e antigos
        new_correct = 0
        new_total = 0
        old_correct = 0
        old_total = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation")
        
        with torch.no_grad():
            for batch in val_pbar:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward
                outputs = student_model(pixel_values=pixel_values, labels=labels)
                loss = base_criterion(outputs.logits, labels)
                val_loss += loss.item()
                
                # Generate predictions
                generated_ids = student_model.generate(
                    pixel_values,
                    max_new_tokens=20,
                    num_beams=3,
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
                    
                    # Separar métricas (aproximado)
                    if len(true) <= 6:  # Assumir que CAPTCHAs curtos são antigos
                        old_total += 1
                        if pred == true:
                            old_correct += 1
                    else:
                        new_total += 1
                        if pred == true:
                            new_correct += 1
                
                current_acc = correct / total if total > 0 else 0
                val_pbar.set_postfix({'acc': f'{current_acc:.2%}'})
        
        val_pbar.close()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        accuracy = correct / total
        old_accuracy = old_correct / old_total if old_total > 0 else 0
        new_accuracy = new_correct / new_total if new_total > 0 else 0
        accuracies.append(accuracy)
        
        epoch_time = time.time() - epoch_start
        
        # ====== LOG METRICS ======
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} ({epoch_time:.1f}s)")
        logger.info(f" Loss - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
        logger.info(f" Accuracy Total: {accuracy:.2%}")
        logger.info(f"   └─ Dados Antigos: {old_accuracy:.2%} ({old_total} exemplos)")
        logger.info(f"   └─ Dados Novos: {new_accuracy:.2%} ({new_total} exemplos)")
        
        if USE_EWC and ewc_loss_total > 0:
            logger.info(f" EWC Penalty médio: {ewc_loss_total/len(train_loader):.4f}")
        if USE_DISTILLATION and kd_loss_total > 0:
            logger.info(f" KD Loss médio: {kd_loss_total/len(train_loader):.4f}")
        
        # ====== SALVAR CHECKPOINT ======
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            
            best_model_path = os.path.join(OUTPUT_DIR, "best_model")
            student_model.save_pretrained(best_model_path)
            processor.save_pretrained(best_model_path)
            
            # Salvar métricas
            metrics = {
                'epoch': epoch + 1,
                'accuracy_total': accuracy,
                'accuracy_old': old_accuracy,
                'accuracy_new': new_accuracy,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': {
                    'use_ewc': USE_EWC,
                    'use_distillation': USE_DISTILLATION,
                    'use_replay': USE_REPLAY,
                    'ewc_lambda': EWC_LAMBDA if USE_EWC else None,
                    'distillation_alpha': DISTILLATION_ALPHA if USE_DISTILLATION else None,
                    'replay_ratio': REPLAY_RATIO if USE_REPLAY else None
                }
            }
            
            with open(os.path.join(best_model_path, "metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"-----> Novo melhor modelo: {best_accuracy:.2%}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"\n Early stopping após {PATIENCE} epochs sem melhoria")
                break
    
    # ====== ANÁLISE FINAL ======
    logger.info("\n" + "="*70)
    logger.info(" TREINAMENTO INCREMENTAL COMPLETO!")
    logger.info("="*70)
    logger.info(f"Melhor acurácia total: {best_accuracy:.2%}")
    logger.info(f" Modelo salvo em: {os.path.join(OUTPUT_DIR, 'best_model')}")
    
    # Análise de catastrophic forgetting
    if old_total > 0 and new_total > 0:
        logger.info(f"\n Análise de Preservação de Conhecimento:")
        logger.info(f"   Performance em dados antigos: {old_accuracy:.2%}")
        logger.info(f"   Performance em dados novos: {new_accuracy:.2%}")
        
        if old_accuracy > 0.7:
            logger.info("    Excelente preservação do conhecimento anterior!")
        elif old_accuracy > 0.5:
            logger.info("    Alguma perda de conhecimento anterior detectada")
        else:
            logger.info("   Catastrophic forgetting significativo")

if __name__ == "__main__":
    try:       
        incremental_training()
    except KeyboardInterrupt:
        logger.info("\n Treinamento interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro: {e}")
        import traceback
        traceback.print_exc()