import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Charger les données prétraitées
data = np.load('preprocessed_data.npz')
x = data['x']
xt = data['xt']
y = data['y']
yt = data['yt']

# Choisir un nombre (par exemple, 1)
chosen_number = 1
yb = (y == chosen_number).astype(int)
ytb = (yt == chosen_number).astype(int)

# Créer le modèle avec une couche de dropout
model_dropout = Sequential([
    Dense(1, activation='sigmoid', input_shape=(x.shape[1],)),
    Dropout(0.5)
])

# Compiler le modèle
model_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# Entraîner le modèle
model_dropout.fit(x, yb, epochs=50, batch_size=100)

# Évaluer le modèle
loss, accuracy = model_dropout.evaluate(xt, ytb)
print(f"Dropout Model Loss: {loss}, Accuracy: {accuracy}")
