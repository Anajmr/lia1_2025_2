import cv2
import time
from ultralytics import YOLO

MODELO_PLACAS_PATH = 'model/best.pt'
VIDEO_PATH = 'video/Video1.mp4'
VIDEO_SAIDA_PATH = 'video/predicao_saida.mp4'

#Carregar modelo
try:
    model = YOLO(MODELO_PLACAS_PATH)
    print("Modelo carregado com sucesso.")
except Exception as e:
    print(f"ERRO ao carregar modelo: {e}")
    exit()

#Abrir vídeo
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERRO: Não foi possível abrir {VIDEO_PATH}")
    exit()

#Informações do vídeo original
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_original = cap.get(cv2.CAP_PROP_FPS)

#Acelerar vídeo
fps_final = fps_original * 2

#Salva  o video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_SAIDA_PATH, fourcc, fps_final, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo.")
        break

    # Redimensiona antes da inferência
    frame_resized = cv2.resize(frame, (640, 640))

    # Predição
    results = model(frame_resized, conf=0.1, iou=0.4, verbose=False)[0]

    # Ajustar detecções para o frame original
    h_orig, w_orig = frame.shape[:2]
    h_res, w_res = frame_resized.shape[:2]
    scale_x, scale_y = w_orig / w_res, h_orig / h_res

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

        cls = int(box.cls.item())
        nomeClasse = model.names.get(cls, "Desconhecido")
        conf = float(box.conf.item())

        cor = (0, 255, 0) if cls == 0 else (255, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)

        # Texto: classe + porcentagem
        texto = f"{nomeClasse} {conf*100:.1f}%"
        cv2.putText(frame, texto, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

    # Grava frame no vídeo de saída
    out.write(frame)

    # Exibir vídeo acelerado
    cv2.imshow("YOLO - Vídeo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Vídeo salvo em: {VIDEO_SAIDA_PATH}")
