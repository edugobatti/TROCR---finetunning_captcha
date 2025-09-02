import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import OneCycleLR
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
                logger.info("✓ Configuração de serialização segura concluída")
    except Exception as e:
        logger.warning(f"⚠ Erro ao configurar serialização: {e}")

# ============================================
# METRICS TRACKER MELHORADO
# ============================================
class MetricsTracker:
    """Rastreia métricas separadas para dados antigos e novos"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {
            'old_data': {'correct': 0, 'total': 0, 'losses': []},
            'new_data': {'correct': 0, 'total': 0, 'losses': []},
            'mixed_data': {'correct': 0, 'total': 0, 'losses': []}
        }
    
    def update(self, pred, true, is_old_data, loss=None):
        """Atualiza métricas baseado na origem dos dados"""
        category = 'old_data' if is_old_data else 'new_data'
        self.metrics[category]['total'] += 1
        
        if pred.strip().upper() == true.strip().upper():
            self.metrics[category]['correct'] += 1
        
        if loss is not None:
            self.metrics[category]['losses'].append(loss)
    
    def get_accuracies(self):
        """Retorna acurácias separadas"""
        results = {}
        for category, data in self.metrics.items():
            if data['total'] > 0:
                results[f"{category}_acc"] = data['correct'] / data['total']
                if data['losses']:
                    results[f"{category}_loss"] = np.mean(data['losses'])
            else:
                results[f"{category}_acc"] = 0.0
                results[f"{category}_loss"] = 0.0
        return results

# ============================================
# ELASTIC WEIGHT CONSOLIDATION (EWC) MELHORADO
# ============================================
class EWC:
    """Implementa Elastic Weight Consolidation para preservar conhecimento anterior"""
    
    def __init__(self, model, dataloader, device, fisher_estimation_sample_size=1000):
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
            # Adicionar pequeno epsilon para estabilidade
            fisher[n] += 1e-8
            
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
# KNOWLEDGE DISTILLATION MELHORADO
# ============================================
class KnowledgeDistillationLoss(nn.Module):
    """Loss de destilação para preservar conhecimento do modelo anterior"""
    
    def __init__(self, temperature=6.0, alpha=0.45):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, labels, criterion):
        """
        Combina loss de destilação com loss supervisionado
        alpha: peso para loss de destilação (45%)
        (1-alpha): peso para loss supervisionado (55%)
        """
        # Loss supervisionado (novo dataset) - 55% de importância
        hard_loss = criterion(student_logits, labels)
        
        # Loss de destilação (preservar conhecimento) - 45% de importância
        soft_targets = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = self.kl_div(soft_targets, soft_teacher) * (self.temperature ** 2)
        
        # Combinar losses
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, hard_loss, soft_loss

# ============================================
# REPLAY BUFFER MELHORADO
# ============================================
class ReplayBuffer:
    """Buffer para armazenar exemplos antigos e fazer rehearsal"""
    
    def __init__(self, max_size=3000):
        self.max_size = max_size
        self.buffer = []
        
    def add(self, image_path, label, priority=1.0):
        """Adiciona exemplo ao buffer com prioridade"""
        if len(self.buffer) >= self.max_size:
            # Remover item com menor prioridade
            min_idx = min(range(len(self.buffer)), 
                         key=lambda i: self.buffer[i][2])
            self.buffer.pop(min_idx)
        self.buffer.append((image_path, label, priority))
    
    def sample(self, n):
        """Amostra n exemplos do buffer com weighted sampling"""
        if len(self.buffer) == 0:
            return [], []
        
        n = min(n, len(self.buffer))
        
        # Weighted sampling baseado em prioridade
        items = self.buffer
        weights = [item[2] for item in items]
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        indices = np.random.choice(len(items), size=n, replace=False, p=weights)
        samples = [items[i] for i in indices]
        
        paths = [s[0] for s in samples]
        labels = [s[1] for s in samples]
        return paths, labels
    
    def get_all(self):
        """Retorna todos os exemplos do buffer"""
        if len(self.buffer) == 0:
            return [], []
        paths = [item[0] for item in self.buffer]
        labels = [item[1] for item in self.buffer]
        return paths, labels
    
    def update_priorities(self, accuracies):
        """Atualiza prioridades baseado em performance"""
        # Dar mais prioridade para exemplos difíceis
        pass  # Implementar se necessário

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
# DATASET COM MIXUP E AUGMENTATION MELHORADO
# ============================================
class IncrementalCaptchaDataset(Dataset):
    """Dataset com suporte para mixup e preservação de conhecimento"""
    
    def __init__(self, image_paths, labels, processor, is_old_data=None,
                 mode='train', augment_prob=0.95, mixup_alpha=0.3, use_mixup=True):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.mode = mode
        self.augment_prob = augment_prob
        self.mixup_alpha = mixup_alpha
        self.use_mixup = use_mixup
        
        # Rastrear origem dos dados
        if is_old_data is None:
            self.is_old_data = [False] * len(labels)
        else:
            self.is_old_data = is_old_data
    
    def augment_image_conservative(self, image):
        """Augmentation mais conservador para não distorcer demais"""
        if self.mode != 'train' or random.random() > self.augment_prob:
            return image
        
        # Aplicar múltiplas augmentations com probabilidades
        augmentations_applied = []
        
        # Rotação sutil
        if random.random() < 0.4:
            angle = random.uniform(-5, 5)
            image = image.rotate(angle, fillcolor=(255, 255, 255), expand=False)
            augmentations_applied.append('rotation')
        
        # Zoom leve
        if random.random() < 0.4:
            zoom = random.uniform(0.92, 1.08)
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
        
        # Ajustes de cor
        if random.random() < 0.5:
            # Brightness
            if random.random() < 0.7:
                brightness = ImageEnhance.Brightness(image)
                image = brightness.enhance(random.uniform(0.85, 1.15))
            
            # Contrast
            if random.random() < 0.7:
                contrast = ImageEnhance.Contrast(image)
                image = contrast.enhance(random.uniform(0.85, 1.15))
            
            # Sharpness
            if random.random() < 0.3:
                sharpness = ImageEnhance.Sharpness(image)
                image = sharpness.enhance(random.uniform(0.8, 1.2))
            
            augmentations_applied.append('color')
        
        # Ruído gaussiano leve
        if random.random() < 0.3:
            pixels = np.array(image)
            noise_std = random.uniform(3, 8)
            noise = np.random.normal(0, noise_std, pixels.shape)
            pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(pixels)
            augmentations_applied.append('noise')
        
        # Blur muito leve
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))
            augmentations_applied.append('blur')
        
        return image
    
    def apply_mixup(self, image1, image2, label1, label2):
        """Aplica mixup entre duas imagens"""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Garantir que as imagens tenham o mesmo tamanho
        if image1.size != image2.size:
            # Redimensionar image2 para o tamanho de image1
            image2 = image2.resize(image1.size, Image.LANCZOS)
        
        # Mixup nas imagens
        mixed_image = Image.blend(image1, image2, 1 - lam)
        
        # Para seq2seq, usar o label dominante
        mixed_label = label1 if lam > 0.5 else label2
        
        return mixed_image, mixed_label, lam
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        try:
            # Carregar imagem principal
            image = load_and_convert_image(self.image_paths[idx], "RGB")
            label_text = self.labels[idx].upper()
            
            # Aplicar mixup ocasionalmente (20% das vezes no treino)
            mixup_applied = False
            if (self.mode == 'train' and self.use_mixup and 
                random.random() < 0.2 and len(self.image_paths) > 1):
                
                try:
                    # Escolher segunda imagem aleatória
                    idx2 = random.randint(0, len(self.image_paths) - 1)
                    while idx2 == idx:
                        idx2 = random.randint(0, len(self.image_paths) - 1)
                    
                    image2 = load_and_convert_image(self.image_paths[idx2], "RGB")
                    label2_text = self.labels[idx2].upper()
                    
                    # Aplicar mixup
                    image, label_text, lam = self.apply_mixup(
                        image, image2, label_text, label2_text
                    )
                    mixup_applied = True
                except Exception as e:
                    # Se mixup falhar, continuar com a imagem original sem mixup
                    # A imagem NÃO é descartada, apenas não recebe mixup
                    logger.debug(f"Mixup falhou para {self.image_paths[idx]}, usando imagem original: {e}")
                    pass
            
            # Aplicar augmentation
            if self.mode == 'train':
                image = self.augment_image_conservative(image)
            
            # Processar imagem
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
            
            # Processar label
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
                "labels": labels,
                "is_old": self.is_old_data[idx]
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar imagem {self.image_paths[idx]}: {e}")
            next_idx = (idx + 1) % len(self.labels)
            return self.__getitem__(next_idx)

# ============================================
# LOSS FUNCTION COM REGULARIZAÇÃO L2
# ============================================
class IncrementalLoss(nn.Module):
    def __init__(self, processor, smoothing=0.03, l2_lambda=0.005):
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
# GRADIENT CLIPPING ADAPTATIVO
# ============================================
def adaptive_gradient_clipping(model, epoch, max_norm=1.0):
    """Clipping adaptativo baseado na época"""
    # Adaptar max_norm baseado na época
    if epoch < 10:
        adaptive_max_norm = max_norm * 2.0  # Mais permissivo no início
    elif epoch < 30:
        adaptive_max_norm = max_norm * 1.5
    elif epoch < 50:
        adaptive_max_norm = max_norm * 1.2
    else:
        adaptive_max_norm = max_norm
    
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 
        adaptive_max_norm
    )
    
    # Converter para float se for tensor
    if isinstance(total_norm, torch.Tensor):
        total_norm = total_norm.item()
    
    return total_norm, adaptive_max_norm

# ============================================
# FUNÇÃO PRINCIPAL DE TREINO INCREMENTAL
# ============================================
def incremental_training():
    # Configurar serialização
    setup_torch_serialization()
    
    # ====== CONFIGURAÇÕES OTIMIZADAS PARA MÁXIMO APRENDIZADO ======
    BASE_MODEL_PATH = "./trocr-incremental-9/best_model"   # Modelo base treinado
    OLD_DATASET_PATH = "./img_base_captcha"                    # Dataset original (para replay)
    NEW_DATASET_PATH = "./captcha-full"                # Novo dataset complexo
    OUTPUT_DIR = "./trocr-incremental-12"                  # Diretório de saída
    
    # Hiperparâmetros otimizados
    BATCH_SIZE = 4                          # Aumentado de 1 para 4
    GRADIENT_ACCUMULATION = 16              # Reduzido de 64 para 16
    LEARNING_RATE = 2.5e-4                  # Aumentado para aprendizado mais agressivo
    MIN_LR = 5e-7
    WEIGHT_DECAY = 0.005                    # Reduzido para permitir mais adaptação
    
    NUM_EPOCHS = 100                        # Aumentado para garantir convergência
    PATIENCE = 20                           # Aumentado para dar mais chance
    WARMUP_STEPS = 500                      # Warmup em steps ao invés de ratio
    
    # Parâmetros de preservação de conhecimento otimizados
    USE_EWC = True                          # Elastic Weight Consolidation
    EWC_LAMBDA = 0.4                        # Reduzido para ser menos conservador
    FISHER_SAMPLES = 1000                   # Aumentado para melhor estimativa
    
    USE_DISTILLATION = True                 # Knowledge Distillation
    DISTILLATION_ALPHA = 0.5               # 45% preservação, 55% novo
    DISTILLATION_TEMPERATURE = 6.0          # Aumentado para mais softening
    
    USE_REPLAY = True                       # Experience Replay
    REPLAY_RATIO = 0.35                     # 35% dados antigos, 65% dados novos
    REPLAY_BUFFER_SIZE = 3000               # Buffer aumentado
    
    # Regularização e augmentation
    LABEL_SMOOTHING = 0.03                  # Aumentado para melhor generalização
    AUGMENT_PROB = 0.95                     # Augmentation muito agressivo
    USE_MIXUP = True                        # Ativar mixup
    MIXUP_ALPHA = 0.3                       # Alpha para mixup
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("="*70)
    logger.info("🚀 TREINAMENTO INCREMENTAL OTIMIZADO - MÁXIMO APRENDIZADO")
    logger.info("="*70)
    logger.info(f"📍 Device: {device}")
    logger.info(f"📁 Modelo base: {BASE_MODEL_PATH}")
    logger.info(f"📁 Dataset antigo: {OLD_DATASET_PATH}")
    logger.info(f"📁 Dataset novo: {NEW_DATASET_PATH}")
    logger.info(f"📁 Output: {OUTPUT_DIR}")
    logger.info(f"⚙️  Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    logger.info(f"📈 Learning rate: {LEARNING_RATE}")
    logger.info(f"🔧 EWC: {USE_EWC} (λ={EWC_LAMBDA}, samples={FISHER_SAMPLES})")
    logger.info(f"🔧 Distillation: {USE_DISTILLATION} (α={DISTILLATION_ALPHA}, T={DISTILLATION_TEMPERATURE})")
    logger.info(f"🔧 Replay: {USE_REPLAY} (ratio={REPLAY_RATIO}, buffer={REPLAY_BUFFER_SIZE})")
    logger.info(f"🔧 Mixup: {USE_MIXUP} (α={MIXUP_ALPHA})")
    
    # Criar diretórios
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
    
    # ====== CARREGAR MODELO BASE E CRIAR CÓPIA PARA TEACHER ======
    logger.info("\n" + "="*70)
    logger.info("📚 CARREGANDO MODELOS")
    logger.info("="*70)
    
    try:
        processor = TrOCRProcessor.from_pretrained(BASE_MODEL_PATH)
        student_model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_PATH)
        logger.info("✓ Modelo base carregado com sucesso")
        
        # Criar modelo teacher (frozen)
        if USE_DISTILLATION:
            teacher_model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_PATH)
            teacher_model.to(device)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            logger.info("✓ Modelo teacher criado e congelado")
        else:
            teacher_model = None
            
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo base: {e}")
        return
    
    # Configurar student model
    student_model.config.use_cache = True
    student_model.to(device)
    
    # ====== CARREGAR DATASETS ======
    logger.info("\n" + "="*70)
    logger.info("📊 CARREGANDO DATASETS")
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
        
        logger.info(f"📂 Dataset antigo: {len(old_image_paths)} imagens")
    
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
    
    logger.info(f"📂 Dataset novo: {len(new_image_paths)} imagens")
    
    # Criar replay buffer
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    
    # Adicionar amostras antigas ao buffer com prioridade
    if USE_REPLAY and len(old_image_paths) > 0:
        # Calcular quantidade de exemplos antigos
        n_old_samples = min(len(old_image_paths), int(len(new_image_paths) * REPLAY_RATIO))
        
        # Amostrar com estratificação se possível
        old_indices = random.sample(range(len(old_image_paths)), n_old_samples)
        
        for idx in old_indices:
            # Dar prioridade maior para exemplos mais difíceis
            priority = random.uniform(0.8, 1.2)
            replay_buffer.add(old_image_paths[idx], old_labels[idx], priority)
        
        logger.info(f"💾 Replay buffer: {len(replay_buffer.buffer)} exemplos antigos")
    
    # Combinar datasets e rastrear origem
    if USE_REPLAY and len(replay_buffer.buffer) > 0:
        replay_paths, replay_labels = replay_buffer.get_all()
        
        combined_paths = new_image_paths + replay_paths
        combined_labels = new_labels + replay_labels
        
        # Rastrear origem dos dados
        is_old_data = [False] * len(new_image_paths) + [True] * len(replay_paths)
        
        logger.info(f"📊 Dataset combinado: {len(combined_paths)} imagens total")
        logger.info(f"   → {len(new_image_paths)} novas ({(len(new_image_paths)/len(combined_paths)*100):.1f}%)")
        logger.info(f"   → {len(replay_paths)} antigas ({(len(replay_paths)/len(combined_paths)*100):.1f}%)")
    else:
        combined_paths = new_image_paths
        combined_labels = new_labels
        is_old_data = [False] * len(combined_paths)
    
    # Adicionar novos tokens se necessário
    all_chars = set(''.join(combined_labels))
    vocab = processor.tokenizer.get_vocab()
    new_tokens = [c for c in all_chars if c not in vocab]
    
    if new_tokens:
        processor.tokenizer.add_tokens(new_tokens)
        student_model.decoder.resize_token_embeddings(len(processor.tokenizer))
        if teacher_model:
            teacher_model.decoder.resize_token_embeddings(len(processor.tokenizer))
        logger.info(f"✓ Adicionados {len(new_tokens)} novos tokens: {new_tokens}")
    
    # Split do dataset mantendo 15% para validação
    train_paths, val_paths, train_labels, val_labels, train_is_old, val_is_old = train_test_split(
        combined_paths, combined_labels, is_old_data,
        test_size=0.15,  # 15% para validação conforme solicitado
        random_state=42
    )
    
    logger.info(f"\n📊 Split final:")
    logger.info(f"   → Treino: {len(train_paths)} imagens (85%)")
    logger.info(f"   → Validação: {len(val_paths)} imagens (15%)")
    
    # ====== CALCULAR EWC SE NECESSÁRIO ======
    ewc = None
    if USE_EWC and len(old_image_paths) > 0:
        logger.info("\n⚖️ Calculando Fisher Information Matrix para EWC...")
        
        # Usar mais amostras para melhor estimativa
        n_fisher_samples = min(FISHER_SAMPLES, len(old_image_paths))
        fisher_indices = random.sample(range(len(old_image_paths)), n_fisher_samples)
        
        ewc_paths = [old_image_paths[i] for i in fisher_indices]
        ewc_labels = [old_labels[i] for i in fisher_indices]
        
        ewc_dataset = IncrementalCaptchaDataset(
            ewc_paths, ewc_labels, processor, 
            is_old_data=[True] * len(ewc_paths),
            mode='val', augment_prob=0
        )
        
        ewc_loader = DataLoader(
            ewc_dataset, 
            batch_size=BATCH_SIZE * 2,  # Batch maior para Fisher
            shuffle=False,
            num_workers=0
        )
        
        ewc = EWC(student_model, ewc_loader, device, fisher_estimation_sample_size=n_fisher_samples)
        logger.info(f"✓ EWC configurado com {n_fisher_samples} amostras")
    
    # ====== CRIAR DATASETS E DATALOADERS ======
    train_dataset = IncrementalCaptchaDataset(
        train_paths, train_labels, processor,
        is_old_data=train_is_old,
        mode='train', 
        augment_prob=AUGMENT_PROB,
        mixup_alpha=MIXUP_ALPHA,
        use_mixup=USE_MIXUP
    )
    
    val_dataset = IncrementalCaptchaDataset(
        val_paths, val_labels, processor,
        is_old_data=val_is_old,
        mode='val', 
        augment_prob=0,
        use_mixup=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=0, 
        drop_last=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE * 2,
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    # ====== CONFIGURAR OTIMIZAÇÃO ======
    # Diferentes learning rates para diferentes partes do modelo
    encoder_params = []
    decoder_embed_params = []
    decoder_layer_params = []
    
    for name, param in student_model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        elif 'decoder' in name and 'embed' in name:
            decoder_embed_params.append(param)
        else:
            decoder_layer_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': LEARNING_RATE * 0.1, 'weight_decay': WEIGHT_DECAY * 2},
        {'params': decoder_embed_params, 'lr': LEARNING_RATE * 0.5, 'weight_decay': WEIGHT_DECAY},
        {'params': decoder_layer_params, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY * 0.5}
    ], betas=(0.9, 0.999), eps=1e-8)
    
    # Calcular total de steps para scheduler
    total_training_steps = len(train_loader) * NUM_EPOCHS
    
    # Learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[LEARNING_RATE * 0.1, LEARNING_RATE * 0.5, LEARNING_RATE],
        total_steps=total_training_steps,
        pct_start=0.05,  # 5% warmup
        anneal_strategy='cos',
        cycle_momentum=False,
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # Loss functions
    base_criterion = IncrementalLoss(processor, LABEL_SMOOTHING, l2_lambda=WEIGHT_DECAY)
    
    if USE_DISTILLATION:
        kd_criterion = KnowledgeDistillationLoss(
            temperature=DISTILLATION_TEMPERATURE,
            alpha=DISTILLATION_ALPHA
        )
    
    # Metrics tracker
    metrics_tracker = MetricsTracker()
    
    # ====== TREINAMENTO ======
    logger.info("\n" + "="*70)
    logger.info("🎯 INICIANDO TREINAMENTO INCREMENTAL OTIMIZADO")
    logger.info("="*70)
    
    best_accuracy = 0
    best_old_accuracy = 0
    best_new_accuracy = 0
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
        grad_norms = []
        
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
            
            # Backward com gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION
            loss.backward()
            
            # Update weights a cada GRADIENT_ACCUMULATION steps
            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                # Gradient clipping adaptativo
                grad_norm, max_norm = adaptive_gradient_clipping(student_model, epoch)
                grad_norms.append(grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * GRADIENT_ACCUMULATION
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            postfix = {
                'loss': f'{loss.item()*GRADIENT_ACCUMULATION:.4f}',
                'lr': f'{current_lr:.2e}'
            }
            if USE_EWC and ewc_loss_total > 0:
                postfix['ewc'] = f'{ewc_loss_total/(batch_idx+1):.4f}'
            if USE_DISTILLATION and kd_loss_total > 0:
                postfix['kd'] = f'{kd_loss_total/(batch_idx+1):.4f}'
            if grad_norms:
                # Garantir que grad_norms são valores float, não tensors
                grad_norms_values = [g.item() if isinstance(g, torch.Tensor) else g for g in grad_norms]
                postfix['grad'] = f'{np.mean(grad_norms_values):.2f}'
            train_pbar.set_postfix(postfix)
        
        train_pbar.close()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # ====== FASE DE VALIDAÇÃO ======
        student_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        # Resetar tracker de métricas
        metrics_tracker.reset()
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation")
        
        with torch.no_grad():
            for batch in val_pbar:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                is_old = batch['is_old']
                
                # Forward
                outputs = student_model(pixel_values=pixel_values, labels=labels)
                loss = base_criterion(outputs.logits, labels)
                val_loss += loss.item()
                
                # Generate predictions
                generated_ids = student_model.generate(
                    pixel_values,
                    max_new_tokens=20,
                    num_beams=3,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
                
                # Decode
                pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                labels_clean = labels.clone()
                labels_clean[labels_clean == -100] = processor.tokenizer.pad_token_id
                true_texts = processor.batch_decode(labels_clean, skip_special_tokens=True)
                
                # Calculate accuracies com tracking
                for pred, true, old in zip(pred_texts, true_texts, is_old):
                    pred = pred.strip().upper()
                    true = true.strip().upper()
                    
                    total += 1
                    if pred == true:
                        correct += 1
                    
                    # Atualizar métricas separadas
                    metrics_tracker.update(pred, true, old.item(), loss.item())
                
                current_acc = correct / total if total > 0 else 0
                val_pbar.set_postfix({'acc': f'{current_acc:.2%}'})
        
        val_pbar.close()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Obter métricas separadas
        metrics = metrics_tracker.get_accuracies()
        accuracy = correct / total
        old_accuracy = metrics.get('old_data_acc', 0)
        new_accuracy = metrics.get('new_data_acc', 0)
        accuracies.append(accuracy)
        
        epoch_time = time.time() - epoch_start
        
        # ====== LOG METRICS ======
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 Epoch {epoch+1}/{NUM_EPOCHS} ({epoch_time:.1f}s)")
        logger.info(f"📉 Loss - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
        logger.info(f"🎯 Accuracy Total: {accuracy:.2%}")
        logger.info(f"   └─ Dados Antigos: {old_accuracy:.2%}")
        logger.info(f"   └─ Dados Novos: {new_accuracy:.2%}")
        logger.info(f"📈 Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        
        if USE_EWC and ewc_loss_total > 0:
            logger.info(f"⚖️ EWC Penalty médio: {ewc_loss_total/len(train_loader):.4f}")
        if USE_DISTILLATION and kd_loss_total > 0:
            logger.info(f"🔄 KD Loss médio: {kd_loss_total/len(train_loader):.4f}")
        if grad_norms:
            grad_norms_values = [g.item() if isinstance(g, torch.Tensor) else g for g in grad_norms]
            logger.info(f"📊 Gradient Norm médio: {np.mean(grad_norms_values):.2f}")
        
        # ====== SALVAR CHECKPOINT ======
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_old_accuracy = old_accuracy
            best_new_accuracy = new_accuracy
            patience_counter = 0
            
            best_model_path = os.path.join(OUTPUT_DIR, "best_model")
            student_model.save_pretrained(best_model_path)
            processor.save_pretrained(best_model_path)
            
            # Salvar métricas detalhadas
            metrics_dict = {
                'epoch': epoch + 1,
                'accuracy_total': accuracy,
                'accuracy_old': old_accuracy,
                'accuracy_new': new_accuracy,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': scheduler.get_last_lr()[0],
                'config': {
                    'batch_size': BATCH_SIZE,
                    'gradient_accumulation': GRADIENT_ACCUMULATION,
                    'effective_batch_size': BATCH_SIZE * GRADIENT_ACCUMULATION,
                    'learning_rate': LEARNING_RATE,
                    'weight_decay': WEIGHT_DECAY,
                    'use_ewc': USE_EWC,
                    'ewc_lambda': EWC_LAMBDA if USE_EWC else None,
                    'fisher_samples': FISHER_SAMPLES if USE_EWC else None,
                    'use_distillation': USE_DISTILLATION,
                    'distillation_alpha': DISTILLATION_ALPHA if USE_DISTILLATION else None,
                    'distillation_temperature': DISTILLATION_TEMPERATURE if USE_DISTILLATION else None,
                    'use_replay': USE_REPLAY,
                    'replay_ratio': REPLAY_RATIO if USE_REPLAY else None,
                    'replay_buffer_size': REPLAY_BUFFER_SIZE if USE_REPLAY else None,
                    'use_mixup': USE_MIXUP,
                    'mixup_alpha': MIXUP_ALPHA if USE_MIXUP else None,
                    'label_smoothing': LABEL_SMOOTHING,
                    'augment_prob': AUGMENT_PROB
                }
            }
            
            with open(os.path.join(best_model_path, "metrics.json"), 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            
            logger.info(f"🏆 Novo melhor modelo salvo: {best_accuracy:.2%}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"\n⛔ Early stopping após {PATIENCE} epochs sem melhoria")
                break
        
        # Salvar checkpoint periódico
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoints", f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'accuracies': accuracies
            }, checkpoint_path)
            logger.info(f"💾 Checkpoint salvo: {checkpoint_path}")
    
    # ====== ANÁLISE FINAL ======
    logger.info("\n" + "="*70)
    logger.info("🎉 TREINAMENTO INCREMENTAL COMPLETO!")
    logger.info("="*70)
    logger.info(f"🏆 Melhor acurácia total: {best_accuracy:.2%}")
    logger.info(f"   └─ Em dados antigos: {best_old_accuracy:.2%}")
    logger.info(f"   └─ Em dados novos: {best_new_accuracy:.2%}")
    logger.info(f"📁 Modelo salvo em: {os.path.join(OUTPUT_DIR, 'best_model')}")
    
    # Análise de catastrophic forgetting
    if best_old_accuracy > 0 and best_new_accuracy > 0:
        logger.info(f"\n📊 Análise de Preservação de Conhecimento:")
        
        preservation_score = best_old_accuracy
        adaptation_score = best_new_accuracy
        overall_score = (preservation_score + adaptation_score) / 2
        
        logger.info(f"   Preservação: {preservation_score:.2%}")
        logger.info(f"   Adaptação: {adaptation_score:.2%}")
        logger.info(f"   Score geral: {overall_score:.2%}")
        
        if preservation_score > 0.75:
            logger.info("   ✅ Excelente preservação do conhecimento anterior!")
        elif preservation_score > 0.60:
            logger.info("   ⚠️ Preservação moderada do conhecimento anterior")
        else:
            logger.info("   ❌ Catastrophic forgetting significativo detectado")
        
        if adaptation_score > 0.85:
            logger.info("   ✅ Excelente adaptação aos novos dados!")
        elif adaptation_score > 0.70:
            logger.info("   ⚠️ Adaptação moderada aos novos dados")
        else:
            logger.info("   ❌ Dificuldade em aprender novos padrões")
    
    # Salvar relatório final
    final_report = {
        'training_completed': True,
        'total_epochs': epoch + 1,
        'best_accuracy': best_accuracy,
        'best_old_accuracy': best_old_accuracy,
        'best_new_accuracy': best_new_accuracy,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'training_time': sum([epoch_time for epoch_time in range(epoch + 1)]),
        'model_path': os.path.join(OUTPUT_DIR, 'best_model')
    }
    
    with open(os.path.join(OUTPUT_DIR, "training_report.json"), 'w') as f:
        json.dump(final_report, f, indent=2)
    
    logger.info(f"\n📄 Relatório final salvo em: {os.path.join(OUTPUT_DIR, 'training_report.json')}")

if __name__ == "__main__":
    try:
        incremental_training()
    except KeyboardInterrupt:
        logger.info("\n⛔ Treinamento interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()