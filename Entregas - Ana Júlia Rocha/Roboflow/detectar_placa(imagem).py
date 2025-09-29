import cv2
import time
from ultralytics import YOLO

# Caminhos dos arquivos
MODELO_PLACAS_PATH = 'model/best.pt'
IMAGE_TESTE_PATH = 'image/foto01.png'
IMAGEM_SAIDA_PATH = 'image/predicao_imagem_final.jpg'

#Carregamento do modelo
try:
    model = YOLO(MODELO_PLACAS_PATH)
    print("Modelo carregado com sucesso.")
    print(f"Classes carregadas: {model.names}")
except Exception as e:
    print(f"ERRO ao carregar modelo: {e}")
    exit()

#Carregar a imagem 
img = cv2.imread(IMAGE_TESTE_PATH)
if img is None:
    print(f"ERRO: Imagem não encontrada em {IMAGE_TESTE_PATH}")
    exit()

#Predição
inicio = time.time()
results = model(img, conf=0.3, iou=0.4, verbose=False)[0]

nomes = results.names
detections_count = 0

#Verificação
if not results.boxes:
    print("Nenhuma detecção encontrada.")
else:
    for box in results.boxes:
        # Coordenadas da bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Classe, Nome e Confiança
        cls = int(box.cls.item())
        nomeClasse = nomes.get(cls, "Desconhecido")
        conf = float(box.conf.item())

        # Mostrar todas as detecções com confiança > 0.3
        if conf < 0.3:
            continue

        detections_count += 1

        # Texto e cor da caixa
        texto = f'{nomeClasse} ({conf:.2f})'
        cor = (0, 255, 0) if cls == 0 else (255, 0, 0)

        # Desenha na imagem
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
        cv2.putText(img, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

#Resultados
tempo_inferencia = time.time() - inicio
print(f"Tempo de inferência: {tempo_inferencia:.4f} segundos")
print(f"Detecções válidas encontradas: {detections_count}")

#Salva a imagem
cv2.imwrite(IMAGEM_SAIDA_PATH, img)
print(f"Imagem com predições salva em: {IMAGEM_SAIDA_PATH}")

# Exibe a imagem 
cv2.imshow('Resultado YOLO', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
