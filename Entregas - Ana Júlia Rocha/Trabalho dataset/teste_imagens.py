# apresentacao_interativa.py (Versão com sugestão de lixeira)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

print("CARREGANDO MODELO TREINADO")

MODEL_FILENAME = 'model_najukaggle.h5'
CLASS_NAMES_FILENAME = 'class_names.txt'
PASTA_IMAGENS = 'imagens apresentação' 
IMG_HEIGHT = 128
IMG_WIDTH = 128

lixeira_info = {
    "paper": ("Lixeira Azul (Papel)", (255, 128, 0)),
    "cardboard": ("Lixeira Azul (Papelao)", (255, 128, 0)),
    "plastic": ("Lixeira Vermelha (Plastico)", (0, 0, 255)),
    "metal": ("Lixeira Amarela (Metal)", (0, 255, 255)),
    "trash": ("Lixeira Cinza (Nao Reciclavel)", (128, 128, 128)),
    "battery": ("Coleta Especial (Pilhas/Baterias)", (0, 0, 0)),
    "glass": ("Lixeira Verde (Vidro)", (0, 255, 0)),
    "biological": ("Lixeira Marrom (Organico)", (20, 70, 140)),
    "clothes": ("Ponto de Doacao/Coleta (Texteis)", (255, 0, 255)),
    "shoes": ("Ponto de Doacao/Coleta (Calcados)", (255, 0, 255))
}

# Carrega o modelo treinado
try:
    model_najukaggle = tf.keras.models.load_model(MODEL_FILENAME)
except Exception as e:
    print(f"\n[ERRO] Não foi possível carregar o modelo '{MODEL_FILENAME}'.")
    exit()

# Carrega os nomes das classes
class_names = []
try:
    with open(CLASS_NAMES_FILENAME, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
except Exception as e:
    print(f"\n[ERRO] Não foi possível carregar '{CLASS_NAMES_FILENAME}'.")
    exit()

print("MODELO CARREGADO")

#PASTA IMAGENS TESTE
try:
    lista_imagens = os.listdir(PASTA_IMAGENS)
    lista_imagens = [f for f in lista_imagens if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not lista_imagens:
        raise FileNotFoundError
except FileNotFoundError:
    print(f"\n[ERRO] A pasta '{PASTA_IMAGENS}' não foi encontrada ou está vazia.")
    exit()


while True:
    print("    CLASSIFICADOR INTERATIVO DE LIXO")
    print("Imagens disponíveis para teste:")
    
    for i, nome_imagem in enumerate(lista_imagens):
        print(f"  [{i+1}] {nome_imagem}")
        
    print("\n  [s] Sair do programa")

    escolha = input("\nDigite o número da imagem que deseja classificar: ")

    if escolha.lower() == 's':
        break

    try:
        index_escolhido = int(escolha) - 1
        nome_arquivo_escolhido = lista_imagens[index_escolhido]
        caminho_imagem = os.path.join(PASTA_IMAGENS, nome_arquivo_escolhido)

        # Prepara a imagem
        img = tf.keras.utils.load_img(caminho_imagem, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model_najukaggle.predict(img_array, verbose=0)
        
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = 100 * np.max(predictions[0])

        info_lixeira, _ = lixeira_info.get(predicted_class, ("Descarte nao especificado", (255, 255, 255)))

        # --- 3. EXIBE O RESULTADO ATUALIZADO ---
        print("\n------------------ RESULTADO ------------------")
        print(f"  Imagem Analisada: {nome_arquivo_escolhido}")
        print(f"  >> Classe Prevista: {predicted_class}")
        print(f"  >> Confiança: {confidence:.2f}%")
        print(f"  >> Descarte Sugerido: {info_lixeira}") 
        print("---------------------------------------------")

        # Mostra a imagem em uma janela pop-up
        plt.imshow(img)
        titulo = f"Previsto: {predicted_class} ({confidence:.2f}%)\nDescarte: {info_lixeira}"
        plt.title(titulo)
        plt.axis("off")
        plt.show()

    except (ValueError, IndexError):
        print("\n[AVISO] Escolha inválida. Por favor, digite um número da lista.")
    except Exception as e:
        print(f"\n[ERRO] Não foi possível processar a imagem: {e}")

print("\nPrograma finalizado.")