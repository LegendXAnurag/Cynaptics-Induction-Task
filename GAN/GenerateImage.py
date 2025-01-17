import tensorflow as tf
tf.keras.backend.clear_session()
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
latent_dim=128
n_samples=49
from tensorflow.keras.models import load_model

#Change the model path correspondingly
g_model = load_model(f'D:\dcgan\final\disc_Final-Epoch-199_.h5')
d_model = load_model(f'D:\dcgan\final\gen_Final-Epoch-199_.h5')

# The models are compiled again as a warning was showing that saved models wont run without compiling again. (optimizer states are not saved)
g_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
d_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input
def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    y = -np.ones((n_samples, 1))
    return X, y
n_samples=49
X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
X = (X - X.min()) / (X.max() - X.min())
square = int(np.sqrt(n_samples))
fig, axes = plt.subplots(square, square, figsize=(4,4), dpi=650)
for i in range(n_samples):
    ax = axes.flatten()[i]
    ax.axis('off')  
    ax.imshow(X[i])
filename1 = 'test.png'
plt.subplots_adjust(wspace=0, hspace=0) 
plt.savefig(filename1, bbox_inches='tight', dpi=650)  
plt.close(fig)
