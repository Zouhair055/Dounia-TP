import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Charger les données prétraitées
data = np.load('preprocessed_data.npz')
x = data['x']
xt = data['xt']
y = data['y']
yt = data['yt']

# Reshape des données pour correspondre aux dimensions des images (28x28 pixels)
x_reshaped = x.reshape(-1, 28, 28, 1)
xt_reshaped = xt.reshape(-1, 28, 28, 1)

# Encoder les labels pour la classification multiclasse
encoder = LabelEncoder()
yb_encoded = encoder.fit_transform(y)
ytb_encoded = encoder.transform(yt)

yb3 = to_categorical(yb_encoded, num_classes=3)
ytb3 = to_categorical(ytb_encoded, num_classes=3)

# Créer le modèle convolutif
model_lenet = Sequential([
    Conv2D(6, (5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(16, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(3, activation='softmax')
])

# Compiler le modèle
model_lenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model_lenet.fit(x_reshaped, yb3, epochs=50, batch_size=100)

# Évaluer le modèle
loss, accuracy = model_lenet.evaluate(xt_reshaped, ytb3)
print(f"LeNet Model Loss: {loss}, Accuracy: {accuracy}")
