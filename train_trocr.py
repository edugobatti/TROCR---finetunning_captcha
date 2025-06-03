import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from PIL import Image
import evaluate
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Dataset personalizado
class CaptchaDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.images = []
        self.texts = []

        for filename in os.listdir(root_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(root_dir, filename))
                self.texts.append(os.path.splitext(filename)[0])

        print(f"Dataset carregado com {len(self.images)} imagens")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        text = self.texts[idx]
    
        # Faz o processamento conjunto da imagem e texto
        encoding = self.processor(
            images=image,
            text=text,
            padding="max_length",
            truncation=True,
            max_length=20,
            return_tensors="pt"
        )
    
        # input_ids pode estar dentro de encoding["labels"], dependendo da versão do transformers
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"].squeeze(0) if "labels" in encoding else encoding["input_ids"].squeeze(0)
    
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

# Função de métrica
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = evaluate.load("cer").compute(predictions=pred_str, references=label_str)
    wer = evaluate.load("wer").compute(predictions=pred_str, references=label_str)

    return {"cer": cer, "wer": wer}

# Função de collate para o DataLoader
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

if __name__ == "__main__":
    captcha_dir = "captchas"
    output_dir = "trocr_captcha_model"
    epochs = 10
    batch_size = 16
    learning_rate = 2e-5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Usando dispositivo: {device}")

    os.makedirs(output_dir, exist_ok=True)

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")  # Pode usar "large" se tiver recursos
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed").to(device)
    # Adiciona isso para evitar o erro
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id

    

    dataset = CaptchaDataset(captcha_dir, processor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        eval_steps=1000,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        predict_with_generate=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=200,
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )

    print("Iniciando treinamento...")
    trainer.train()

    model.save_pretrained(f"{output_dir}/final_model")
    processor.save_pretrained(f"{output_dir}/final_processor")
    print("Treinamento finalizado.")

    # Teste rápido (opcional)
    print("\nTestando o modelo em algumas imagens:")
    test_images = [
        os.path.join(captcha_dir, f)
        for f in os.listdir(captcha_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ][:5]

    correct = 0
    for img_path in test_images:
        image = Image.open(img_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        expected_text = os.path.splitext(os.path.basename(img_path))[0]

        print(f"Imagem: {os.path.basename(img_path)}")
        print(f"Esperado: {expected_text} | Predito: {generated_text}")
        print(f"Correto: {'✓' if expected_text == generated_text else '✗'}")
        print("-" * 50)
        if expected_text == generated_text:
            correct += 1

    if test_images:
        print(f"Acurácia nos testes: {correct/len(test_images):.2%}")

    print("\nExecute: tensorboard --logdir=trocr_captcha_model/logs para visualizar os logs.")
