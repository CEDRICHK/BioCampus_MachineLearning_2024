

!pip install tensorflow==2.10
!pip install n2v
!pip install scikit-image
!nvidia-smi

reticulate::use_python("/home/cedric/anaconda3/envs/N2V/bin/python3.9")


import tensorflow as tf
import n2v
print(tf.__version__)
print(n2v.__version__)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# Import all dependencies
from n2v.models import N2VConfig, N2V
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
from skimage import io, util
import urllib
import os
import zipfile
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Create DataGenerator object
datagen = N2V_DataGenerator()
# Load images from the BBBC006 dataset
input_folder = '/home/cedric/Documents/BioCampus_MDL_2024/exercises/python/N2V/images/noisy-imgs/'

# Utiliser load_imgs_from_directory pour charger les images
images = datagen.load_imgs_from_directory(directory=input_folder, filter='*.tif', dims='YX')

print(f'Loaded {len(images)} images from dataset.')

# Afficher quelques images pour vérification
for i in range(min(3, len(images))):
    plt.figure()
    plt.imshow(np.squeeze(images[i]), cmap='gray')
    plt.title(f'Image: {i}')
    plt.axis('off')
    plt.show()

# Vérifiez les dimensions des images
print(f"Example image shape: {images[0].shape}")

# Extraire des patchs des images en utilisant generate_patches_from_list
patch_size = 64
patches = datagen.generate_patches_from_list(images, shape=(patch_size, patch_size), augment=True, shuffle=True)

# Vérifiez la forme des patchs générés
if patches is None:
    raise ValueError("Patch generation failed, patches is None.")
else:
    print(f"Generated patches shape: {patches.shape}")

# Diviser en ensembles d'entraînement et de validation
train_val_split = int(patches.shape[0] * 0.8)
X = patches[:train_val_split]
X_val = patches[train_val_split:]

print(f'Training patches shape: {X.shape}')
print(f'Validation patches shape: {X_val.shape}')

# Afficher deux patchs pour vérification
plt.figure(figsize=(14,7))
plt.subplot(1,2,1)
plt.imshow(np.squeeze(X[0]), cmap='gray')
plt.title('Training Patch')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(np.squeeze(X_val[0]), cmap='gray')
plt.title('Validation Patch')
plt.axis('off')
plt.show()

# Configurer le modèle N2V
train_batch = 32  # Augmenter la taille des lots
config = N2VConfig(X, 
                   unet_kern_size=3, 
                   unet_n_first=16,  # Augmenter le nombre de filtres initiaux
                   unet_n_depth=3,  # Augmenter la profondeur du réseau
                   train_steps_per_epoch=int(X.shape[0] / train_batch), 
                   train_epochs=50,  # Augmenter le nombre d'époques
                   train_loss='mse', 
                   batch_norm=True, 
                   train_batch_size=train_batch, 
                   n2v_perc_pix=0.198, 
                   n2v_patch_shape=(patch_size, patch_size), 
                   n2v_manipulator='uniform_withCP', 
                   n2v_neighborhood_radius=5, 
                   single_net_per_channel=False,
                   train_learning_rate=0.001,  # Définir le taux d'apprentissage initial
                   train_reduce_lr_on_plateau=True,  # Réduire le taux d'apprentissage en cas de plateau
                   train_reduce_lr_factor=0.5,  # Facteur de réduction du taux d'apprentissage
                   train_reduce_lr_patience=5)  # Patience avant réduction du taux d'apprentissage

# Vérifier les paramètres stockés dans l'objet config
print(vars(config))

# Définir le nom du modèle et le répertoire
model_name = 'n2v_noisy'
basedir = 'models'
model = N2V(config, model_name, basedir=basedir)

# Entraîner le modèle
history = model.train(X, X_val)

# Tracer l'historique de l'entraînement
print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history, ['loss', 'val_loss'])

# Dénouer des images en utilisant le modèle entraîné
model = N2V(config=None, name=model_name, basedir=basedir)

# Charger une image de test du dataset BBBC006
test_image = images[0]

# Dénouer l'image (prédiction)
pred = model.predict(test_image, axes='SYXC')

# Afficher les résultats
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.imshow(np.squeeze(test_image), cmap='gray')
plt.title('Input')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(np.squeeze(pred), cmap='gray')
plt.title('Prediction')
plt.axis('off')
plt.show()
