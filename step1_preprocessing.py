import numpy as np

# Charger les données
data = np.load('digits.npz')
x = data['x']
xt = data['xt']

# Normaliser les données
x = x / 255.0
xt = xt / 255.0

# Sauvegarder les données prétraitées pour les étapes suivantes
np.savez('preprocessed_data.npz', x=x, xt=xt, y=data['y'], yt=data['yt'])
print("Données chargées et normalisées.")
