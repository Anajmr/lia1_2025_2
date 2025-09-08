
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

DATA_DIR = 'C:/LIA VSCODE/archive/garbage-dataset'
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

print("\n Carregando e dividindo o dataset")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
NUM_CLASSES = len(class_names)
print(f"Classes encontradas ({NUM_CLASSES}): {class_names}")

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
])

model_najukaggle = models.Sequential()

model_najukaggle.add(layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)))

# Aumento de Dados
model_najukaggle.add(data_augmentation)

model_najukaggle.add(layers.Rescaling(1./255))

model_najukaggle.add(layers.Conv2D(32, (3, 3), activation='relu'))
model_najukaggle.add(layers.MaxPooling2D((2, 2)))
model_najukaggle.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_najukaggle.add(layers.MaxPooling2D((2, 2)))
model_najukaggle.add(layers.Conv2D(64, (3, 3), activation='relu'))

model_najukaggle.add(layers.Flatten())
model_najukaggle.add(layers.Dense(128, activation='relu')) 
model_najukaggle.add(layers.Dropout(0.5)) 
model_najukaggle.add(layers.Dense(NUM_CLASSES, activation='softmax'))

model_najukaggle.summary()


#COMPILAÇÃO E TREINAMENTO

model_najukaggle.compile(optimizer='adam', 
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

print("\nTREINAMENTO")
history = model_najukaggle.fit(
    train_dataset,
    epochs=70,
    validation_data=validation_dataset
)
print("\nFINALIZADO")

print("\nDESEMPENHO TREINAMENTO")


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Acurácia de Treino')
plt.plot(epochs_range, val_acc, label='Acurácia de Validação')
plt.legend(loc='lower right')
plt.title('Acurácia de Treino e Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perda de Treino')
plt.plot(epochs_range, val_loss, label='Perda de Validação')
plt.legend(loc='upper right')
plt.title('Perda de Treino e Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')

plt.tight_layout()
plt.show()

#TESTE E MATRIZ DE CONFUSÃO
loss, acc = model_najukaggle.evaluate(validation_dataset, verbose=2) 
print(f"\nAcurácia com dados de validação: {acc:.4f}")

print("\nMATRIZ DE CONFUSÃO")
y_pred_classes = []
y_true = []

for images, labels in validation_dataset:
    predictions = model_najukaggle.predict(images, verbose=0)
    y_pred_classes.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title(f'Matriz de Confusão - Acurácia: {acc:.2f}')
plt.xlabel('Rótulo Previsto')
plt.ylabel('Rótulo Verdadeiro')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


model_najukaggle.save('model_najukaggle.h5') 
print("\nModelo salvo como 'model_najukaggle.h5'")

with open('class_names.txt', 'w') as f:
    for item in class_names:
        f.write("%s\n" % item)
print("Nomes das classes salvos em 'class_names.txt'")