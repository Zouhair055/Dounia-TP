import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

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

# Créer le modèle avec régularisation L2
model_l2 = Sequential([
    Dense(1, activation='sigmoid', input_shape=(x.shape[1],), kernel_regularizer=l2(0.01))
])

# Compiler le modèle
model_l2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# Entraîner le modèle
model_l2.fit(x, yb, epochs=50, batch_size=100)

# Évaluer le modèle
loss, accuracy = model_l2.evaluate(xt, ytb)
print(f"L2 Regularized Model Loss: {loss}, Accuracy: {accuracy}")
