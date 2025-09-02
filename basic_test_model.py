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
        print(f" Carregando modelo de: {model_path}")
        
        # Verifica se a pasta existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pasta do modelo não encontrada: {model_path}")
        
        # Carrega o processador e modelo
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        
        print(" Modelo carregado com sucesso!")
    
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
        
        # Conta as ocorrências
        counter = Counter(predictions)
        most_common = counter.most_common(1)[0]
        
        tta_result = most_common[0]
        confidence = most_common[1] / num_predictions
        
        return tta_result, confidence, counter

def get_expected_text_from_filename(filename):
    """
    Extrai o texto esperado do nome do arquivo
    Exemplo: 'ZVKGÇ.PNG' -> 'ZVKGÇ'
    """
    # Remove a extensão do arquivo
    name_without_ext = os.path.splitext(filename)[0]
    return name_without_ext

def test_multiple_captchas(folder_path="captchas_teste", use_tta=True):
    """
    Testa múltiplos captchas em uma pasta e compara com os nomes dos arquivos
    """
    print("="*70)
    print("🔍 TESTE DE MÚLTIPLOS CAPTCHAS")
    print("="*70)
    
    # Verifica se a pasta existe
    if not os.path.exists(folder_path):
        print(f"Pasta '{folder_path}' não encontrada!")
        return
    
    # Lista todas as imagens na pasta
    image_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    image_files = [f for f in os.listdir(folder_path) if f.endswith(image_extensions)]
    
    if not image_files:
        print(f"Nenhuma imagem encontrada na pasta '{folder_path}'!")
        return
    
    print(f"\n📁 Pasta: {folder_path}")
    print(f" Total de imagens encontradas: {len(image_files)}")
    print(f" Modo: {'Com TTA (5 predições por imagem)' if use_tta else 'Predição simples'}")
    print("-"*70)
    
    try:
        # Inicializar preditor
        predictor = CaptchaSafetensorsPredictor(model_path="trocr-anti-overfit-final-100k")
        
        # Contadores
        total_correct = 0
        total_images = len(image_files)
        results = []
        
        # Processar cada imagem
        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(folder_path, image_file)
            expected_text = get_expected_text_from_filename(image_file)
            
            print(f"\n[{idx}/{total_images}] Processando: {image_file}")
            
            # Fazer predição
            if use_tta:
                predicted_text, confidence, counter = predictor.predict_with_tta(image_path)
                print(f"   Votações: {dict(counter)}")
                print(f"   Confiança: {confidence:.1%}")
            else:
                predicted_text, time_taken = predictor.predict(image_path)
                confidence = 1.0
            
            # Comparar resultado
            is_correct = predicted_text.upper() == expected_text.upper()
            
            # Armazenar resultado
            results.append({
                'arquivo': image_file,
                'esperado': expected_text,
                'predito': predicted_text,
                'correto': is_correct,
                'confianca': confidence
            })
            
            # Atualizar contador
            if is_correct:
                total_correct += 1
            
            # Mostrar resultado
            print(f"   Esperado: {expected_text}")
            print(f"   Predito:  {predicted_text}")
            print(f"   Status:   {' CORRETO' if is_correct else 'INCORRETO'}")
        
        # Estatísticas finais
        accuracy = (total_correct / total_images) * 100
        
        print("\n" + "="*70)
        print("RESULTADO FINAL")
        print("="*70)
        print(f" Acertos: {total_correct}/{total_images}")
        print(f" Acurácia: {accuracy:.2f}%")
        
        # Mostrar erros
        errors = [r for r in results if not r['correto']]
        if errors:
            print(f"\nErros ({len(errors)} total):")
            for err in errors:
                print(f"   {err['arquivo']}: esperado '{err['esperado']}' → obtido '{err['predito']}'")
        
        # Mostrar estatísticas de confiança (se usando TTA)
        if use_tta:
            avg_confidence = sum(r['confianca'] for r in results) / len(results)
            print(f"\nConfiança média: {avg_confidence:.1%}")
            
            # Separar por níveis de confiança
            high_conf = [r for r in results if r['confianca'] >= 0.8]
            med_conf = [r for r in results if 0.6 <= r['confianca'] < 0.8]
            low_conf = [r for r in results if r['confianca'] < 0.6]
            
            print(f"\n Distribuição de confiança:")
            print(f"   Alta (≥80%): {len(high_conf)} imagens")
            print(f"   Média (60-79%): {len(med_conf)} imagens")
            print(f"   Baixa (<60%): {len(low_conf)} imagens")
        
        print("="*70)
        
        return results
        
    except Exception as e:
        print(f"\nErro: {e}")
        import traceback
        traceback.print_exc()
        return None

# Função para testar uma única imagem (mantida para compatibilidade)
def test_captcha(image_path=None):
    """
    Função para testar um único captcha
    """
    if image_path is None:
        print("Caminho da imagem não especificado!")
        return None
    
    try:
        predictor = CaptchaSafetensorsPredictor(model_path="trocr-anti-overfit-final-100k")
        
        print(f"\n Testando: {image_path}")
        result, time_taken = predictor.predict(image_path)
        print(f"Resultado: {result}")
        print(f"Tempo: {time_taken:.3f}s")
        
        return result
        
    except Exception as e:
        print(f"\nErro: {e}")
        import traceback
        traceback.print_exc()
        return None

# Exemplo de uso
if __name__ == "__main__":
    # Testar múltiplos captchas da pasta 'captchas_teste'
    results = test_multiple_captchas(folder_path="captchas_teste", use_tta=True)
    
    # Para testar sem TTA (mais rápido):
    # results = test_multiple_captchas(folder_path="captchas_teste", use_tta=False)