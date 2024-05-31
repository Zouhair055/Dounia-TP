import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Charger les données prétraitées
data = np.load('preprocessed_data.npz')
x = data['x']
xt = data['xt']
y = data['y']
yt = data['yt']

# Encoder les labels pour la classification multiclasse
encoder = LabelEncoder()
yb_encoded = encoder.fit_transform(y)
ytb_encoded = encoder.transform(yt)

yb3 = to_categorical(yb_encoded, num_classes=3)
ytb3 = to_categorical(ytb_encoded, num_classes=3)

# Créer le modèle avec 3 outputs
model_multiclass = Sequential([
    Dense(3, activation='softmax', input_shape=(x.shape[1],))
])

# Compiler le modèle
model_multiclass.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model_multiclass.fit(x, yb3, epochs=50, batch_size=100)

# Évaluer le modèle
loss, accuracy = model_multiclass.evaluate(xt, ytb3)
print(f"Multiclass Classification Loss: {loss}, Accuracy: {accuracy}")
