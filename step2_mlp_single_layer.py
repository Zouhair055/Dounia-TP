import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

# Créer le modèle
model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(x.shape[1],))
])

# Compiler le modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# Afficher le résumé du modèle
model.summary()

# Entraîner le modèle
model.fit(x, yb, epochs=50, batch_size=100)

# Évaluer le modèle
loss, accuracy = model.evaluate(xt, ytb)
print(f"Binary Classification Loss: {loss}, Accuracy: {accuracy}")
