import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

latent_dim = 128
BATCH = 32
images_to_load= 8000
epochs = 200

dataset = keras.preprocessing.image_dataset_from_directory(
    directory=r"D:/dcgan/celeba_hq/train/female", label_mode=None, image_size=(128,128), batch_size=BATCH,
    shuffle=True, seed=None, validation_split=None,
).map(lambda x: x/255.0).take(images_to_load // BATCH)
#dataset link https://www.kaggle.com/datasets/lamsimon/celebahq  (3 GB)

discriminator = keras.Sequential(
    [
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(256, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(512, kernel_size=4, strides=2, padding="same"), 
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)
print(discriminator.summary())

# Generator
generator = keras.Sequential(
    [
        layers.Input(shape=(latent_dim,)),
        layers.Dense(16 * 16 * 256),
        layers.Reshape((16, 16, 256)),
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
    ],
    name="generator",
)
print(generator.summary())



opt_gen = keras.optimizers.Adam(0.0002, beta_1=0.5, beta_2=0.999,epsilon=1e-8)
opt_disc = keras.optimizers.Adam(0.0002, beta_1=0.5, beta_2=0.999,epsilon=1e-8)
loss_fn = keras.losses.BinaryCrossentropy()

def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input
def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    y = -np.ones((n_samples, 1))
    return X, y
def summarize_performance(epoch,g_model,d_model, latent_dim, n_samples=49):
    name = 'Final-Epoch-%s_' % (epoch)

    
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    X = (X - X.min()) / (X.max() - X.min())
    square = int(n_samples**0.5)
    fig, axes = plt.subplots(square, square, figsize=(4,4), dpi=600)

    for i in range(n_samples):
        ax = axes.flatten()[i]
        ax.axis('off') 
        ax.imshow(X[i]) 
    filename1 = '128x128_%s.png' % (name)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(filename1, bbox_inches='tight', dpi=600)
    plt.close(fig)
    
    filename2 = 'gen_%s.h5' % (name)
    g_model.save(filename2)
    filename3 = 'disc_%s.h5' % (name)
    d_model.save(filename3)
    print('>Saved: %s, %s and %s' % (filename1, filename2,filename3))

for epoch in range(epochs):
    for idx, (real) in enumerate(tqdm(dataset)):
        batch_size = real.shape[0]
        with tf.GradientTape() as gen_tape:
            random_latent_vectors = tf.random.normal(shape = (batch_size, latent_dim))
            fake = generator(random_latent_vectors)
        with tf.GradientTape() as disc_tape:
            loss_disc_real = loss_fn(tf.ones((batch_size, 1)), discriminator(real))
            loss_disc_fake = loss_fn(tf.zeros((batch_size, 1)), discriminator(fake))
            loss_disc = (loss_disc_real + loss_disc_fake)/2

        grads = disc_tape.gradient(loss_disc, discriminator.trainable_weights)
        opt_disc.apply_gradients(
            zip(grads, discriminator.trainable_weights)
        )
        with tf.GradientTape() as gen_tape:
            fake = generator(random_latent_vectors)
            output = discriminator(fake)
            loss_gen = loss_fn(tf.ones(batch_size, 1), output)

        grads = gen_tape.gradient(loss_gen, generator.trainable_weights)
        opt_gen.apply_gradients(zip(grads, generator.trainable_weights))
    summarize_performance(epoch,generator,discriminator,latent_dim)
    tqdm.write('Image Generated for Epoch-%d' %(epoch))
