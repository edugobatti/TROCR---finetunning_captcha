import os
from collections import Counter
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import time

class CaptchaSafetensorsPredictor:
    def __init__(self, model_path="trocr-anti-overfit-final-100k"):
        """
        Inicializa o preditor carregando o modelo da pasta especificada
        """
        print(f"🔄 Carregando modelo de: {model_path}")
        
        # Verifica se a pasta existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pasta do modelo não encontrada: {model_path}")
        
        # Carrega o processador e modelo
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        
        print("✅ Modelo carregado com sucesso!")
    
    def predict(self, image_path):
        """
        Faz a predição de um captcha
        """
        start_time = time.time()
        
        # Carrega e processa a imagem
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        
        # Gera a predição
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        time_taken = time.time() - start_time
        return generated_text, time_taken
    
    def predict_with_tta(self, image_path, num_predictions=5):
        """
        Faz múltiplas predições e retorna o resultado mais comum (Test Time Augmentation)
        """
        predictions = []
        
        for i in range(num_predictions):
            result, _ = self.predict(image_path)
            predictions.append(result)
            print(f"   Predição {i+1}: {result}")
        
        # Conta as ocorrências
        counter = Counter(predictions)
        most_common = counter.most_common(1)[0]
        
        tta_result = most_common[0]
        confidence = most_common[1] / num_predictions
        
        return tta_result, confidence, counter

def find_first_image(directory="."):
    """
    Procura a primeira imagem válida (png, jpg, jpeg) no diretório especificado.
    """
    for file in os.listdir(directory):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            return os.path.join(directory, file)
    return None

def test_captcha(image_path=None):
    """
    Função principal para testar com fallback para imagens da pasta
    """
    print("="*60)
    print("🔍 TESTE DE CAPTCHA (MODELO SAFETENSORS)")
    print("="*60)
    
    # Se nenhum caminho foi fornecido, tentar encontrar uma imagem na pasta
    if image_path is None or not os.path.exists(image_path):
        print(f"\n⚠️  Imagem '{image_path}' não encontrada ou não especificada.")
        image_path = find_first_image()
        if image_path:
            print(f"📷 Usando imagem encontrada: {image_path}")
        else:
            print("❌ Nenhuma imagem encontrada na pasta atual!")
            return None
    
    try:
        # Inicializar preditor com o caminho do modelo
        predictor = CaptchaSafetensorsPredictor(model_path="trocr-anti-overfit-final-100k")
        
        print(f"\n📷 Testando: {image_path}")
        print("-"*60)
        
        # Predição simples
        print("\n1️⃣ PREDIÇÃO SIMPLES:")
        result, time_taken = predictor.predict(image_path)
        print(f"   Resultado: {result}")
        print(f"   Tempo: {time_taken:.3f}s")
        
        # TTA
        print("\n2️⃣ PREDIÇÃO COM TTA:")
        tta_result, confidence, counter = predictor.predict_with_tta(image_path)
        
        print(f"\n📊 RESULTADO FINAL:")
        print(f"   🎯 Captcha: {tta_result}")
        print(f"   📈 Confiança: {confidence:.1%}")
        print(f"   🗳️  Votações: {dict(counter)}")
        print("="*60)
        
        return tta_result
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return None

# Exemplo de uso
if __name__ == "__main__":
    # Pode passar o caminho da imagem ou deixar None para buscar automaticamente
    resultado = test_captcha('./captchas_teste/0HNXS.png')  # ou test_captcha("caminho/para/imagem.png")
    print(resultado)