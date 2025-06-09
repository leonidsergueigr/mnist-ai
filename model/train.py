import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split


tf.random.set_seed(42)
np.random.seed(42)

print("Version de TensorFlow:", tf.__version__)

print("\n=== CHARGEMENT DES DONNÉES ===")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(f"Forme des données d'entraînement: {x_train.shape}")
print(f"Forme des étiquettes d'entraînement: {y_train.shape}")
print(f"Forme des données de test: {x_test.shape}")
print(f"Forme des étiquettes de test: {y_test.shape}")

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f'Chiffre: {y_train[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()

print("\n=== PRÉPROCESSING ===")
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], 28*28)
print(f"Nouvelle forme après redimensionnement: {x_train.shape}")

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(f"Forme des étiquettes après one-hot: {y_train.shape}")

print("Création du set de validation...")
x_train_split, x_val, y_train_split, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42, 
    stratify=np.argmax(y_train, axis=1)
)
print(f"Données d'entraînement: {x_train_split.shape}")
print(f"Données de validation: {x_val.shape}")

print("\n=== CRÉATION DU MODÈLE OPTIMISÉ ===")
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(10, activation='softmax')
])

print("\n=== ARCHITECTURE DU MODÈLE ===")
model.summary()
print(f"Nombre total de paramètres: {model.count_params():,}")

print("\n=== COMPILATION ===")
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n=== CONFIGURATION DES CALLBACKS ===")
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model_temp.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("\n=== ENTRAÎNEMENT ===")
history = model.fit(
    x_train_split, y_train_split,
    batch_size=128,
    epochs=100,  # Plus d'époques avec early stopping
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=1
)

print("\n=== ÉVALUATION ===")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Précision sur les données de test: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Perte sur les données de test: {test_loss:.4f}")


val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
print(f"Précision sur les données de validation: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

print("\n=== VISUALISATION ===")
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Précision entraînement', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Précision validation', linewidth=2)
plt.title('Évolution de la précision')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Perte entraînement', linewidth=2)
plt.plot(history.history['val_loss'], label='Perte validation', linewidth=2)
plt.title('Évolution de la perte')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
if 'lr' in history.history:
    plt.plot(history.history['lr'], linewidth=2)
    plt.title('Évolution du Learning Rate')
    plt.xlabel('Époque')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
else:
    plt.text(0.5, 0.5, 'Learning Rate\nnon disponible', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Learning Rate')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


print("\n=== TEST SUR DES EXEMPLES ===")
num_examples = 10
predictions = model.predict(x_test[:num_examples])
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test[:num_examples], axis=1)

plt.figure(figsize=(15, 6))
for i in range(num_examples):
    plt.subplot(2, 5, i+1)
    img = x_test[i].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    
    confidence = np.max(predictions[i]) * 100
    is_correct = predicted_classes[i] == true_classes[i]
    color = 'green' if is_correct else 'red'
    
    plt.title(f'Prédit: {predicted_classes[i]}\nVrai: {true_classes[i]}\nConfiance: {confidence:.1f}%', 
              color=color, fontsize=9)
    plt.axis('off')

plt.suptitle('Test sur des exemples (Vert=Correct, Rouge=Erreur)', fontsize=12)
plt.tight_layout()
plt.show()

print("\n=== ANALYSE DE PERFORMANCE ===")
all_predictions = model.predict(x_test)
all_predicted_classes = np.argmax(all_predictions, axis=1)
all_true_classes = np.argmax(y_test, axis=1)

print("Performance par chiffre:")
print("Chiffre | Précision | Nb d'exemples | Erreurs")
print("-" * 50)
total_correct = 0
total_examples = 0

for digit in range(10):
    indices_digit = np.where(all_true_classes == digit)[0]
    
    if len(indices_digit) > 0:
        pred_digit = all_predicted_classes[indices_digit]
        
        correct = np.sum(pred_digit == digit)
        incorrect = len(indices_digit) - correct
        
        precision = correct / len(indices_digit)
        
        print(f"   {digit}    | {precision*100:6.1f}%   |      {len(indices_digit)}     |   {incorrect}")
        
        total_correct += correct
        total_examples += len(indices_digit)

overall_accuracy = total_correct / total_examples
print("-" * 50)
print(f"Global  | {overall_accuracy*100:6.1f}%   |     {total_examples}     |   {total_examples - total_correct}")


print("\n=== SAUVEGARDE ===")


model.save('mnist_model.h5')
print("✅ Modèle optimisé sauvegardé sous 'mnist_model.h5'")

if os.path.exists('best_model_temp.h5'):
    os.remove('best_model_temp.h5')


print("\n" + "="*60)
print("RÉSUMÉ DU PROJET - MODÈLE OPTIMISÉ")
print("="*60)
print(f"• Dataset utilisé: MNIST (chiffres manuscrits)")
print(f"• Nombre d'exemples d'entraînement: {len(x_train_split):,}")
print(f"• Nombre d'exemples de validation: {len(x_val):,}")
print(f"• Nombre d'exemples de test: {len(x_test):,}")
print(f"• Architecture: Réseau dense profond (512->256->128->64->10)")
print(f"• Techniques d'optimisation: BatchNorm, Dropout, EarlyStopping, ReduceLROnPlateau")
print(f"• Nombre de paramètres: {model.count_params():,}")
print(f"• Précision finale sur test: {test_accuracy*100:.2f}%")
print(f"• Précision finale sur validation: {val_accuracy*100:.2f}%")
print(f"• Nombre d'époques effectuées: {len(history.history['accuracy'])}")

best_val_acc = max(history.history['val_accuracy'])
best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
print(f"• Meilleure précision validation: {best_val_acc*100:.2f}% (époque {best_epoch})")
print("="*60)
print("✅ Modèle prêt pour l'utilisation avec Flask!")