# G-GAN for Face Frontalization - PC Optimized Version
# Specs: RTX 3050, 8GB RAM, TensorFlow GPU

import os
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm

# Enable mixed precision for RTX 3050 (if supported)
mixed_precision.set_global_policy('mixed_float16')

# Check GPU
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Config
IMG_HEIGHT, IMG_WIDTH = 64, 64
BATCH_SIZE = 4
EPOCHS = 200

# Load and preprocess dataset (resize to 64x64)
def load_image_pair(profile_path, frontal_path):
    profile_img = Image.open(profile_path).resize((IMG_WIDTH, IMG_HEIGHT)).convert('RGB')
    frontal_img = Image.open(frontal_path).resize((IMG_WIDTH, IMG_HEIGHT)).convert('RGB')
    profile_img = ((np.array(profile_img).astype(np.float32)) / 127.5) - 1.0
    frontal_img = ((np.array(frontal_img).astype(np.float32)) / 127.5) - 1.0
    return profile_img, frontal_img

def load_dataset_pairs(base_path):
    profile_images = []
    frontal_images = []
    person_dirs = glob(os.path.join(base_path, '*'))
    for person_dir in tqdm(person_dirs):
        profile_imgs = sorted(glob(os.path.join(person_dir, 'profile', '*.jpg')))
        frontal_imgs = sorted(glob(os.path.join(person_dir, 'frontal', '*.jpg')))
        min_len = min(len(profile_imgs), len(frontal_imgs))
        for i in range(min_len):
            prof, fron = load_image_pair(profile_imgs[i], frontal_imgs[i])
            profile_images.append(prof)
            frontal_images.append(fron)
    return np.array(profile_images), np.array(frontal_images)

# Load data (adjust this path to your local data folder)
profile_imgs, frontal_imgs = load_dataset_pairs('./dataset/cfp-dataset/Data/Images')

dataset = tf.data.Dataset.from_tensor_slices((profile_imgs, frontal_imgs))
dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Generator with dropout
def build_generator(input_shape=(64, 64, 3)):
    inputs = Input(shape=input_shape)
    x = inputs
    for i, filters in enumerate([16, 32, 64, 128, 256, 512, 1024]):
        x = Conv2D(filters, kernel_size=4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        if i >= 3:
            x = Dropout(0.3)(x)

    for i, filters in enumerate([512, 256, 128, 64, 32, 16]):
        x = Conv2DTranspose(filters, kernel_size=4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        if i < 3:
            x = Dropout(0.3)(x)

    x = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh', dtype='float32')(x)
    return Model(inputs, x, name="PC_GGAN_Generator")

# Discriminator

def build_discriminator(input_shape=(64, 64, 3)):
    inputs = Input(shape=input_shape)
    x = inputs
    for filters in [16, 32, 64, 128, 256, 512]:
        x = Conv2D(filters, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid', dtype='float32')(x)
    return Model(inputs, x, name="PC_GGAN_Discriminator")

# Custom loss

def combined_gan_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    l1 = tf.reduce_mean(tf.square(y_true - y_pred))
    l2 = tf.reduce_mean(tf.abs(y_true - y_pred))
    l3 = tf.reduce_mean(tf.square(y_true - y_pred))
    return 0.001 * l1 + 1.0 * l2 + 1.0 * l3

# Build models
with tf.device('/GPU:0'):
    generator = build_generator()
    discriminator = build_discriminator()
    g_optimizer = Adam(0.0002, beta_1=0.5)
    d_optimizer = Adam(0.0002, beta_1=0.5)
    bce = tf.keras.losses.BinaryCrossentropy()

# Training step
@tf.function
def train_step(profile_images, real_frontal_images):
    with tf.GradientTape(persistent=True) as tape:
        generated_frontal = generator(profile_images, training=True)
        real_output = discriminator(real_frontal_images, training=True)
        fake_output = discriminator(generated_frontal, training=True)
        d_loss_real = bce(tf.ones_like(real_output), real_output)
        d_loss_fake = bce(tf.zeros_like(fake_output), fake_output)
        d_loss = d_loss_real + d_loss_fake
        g_loss_gan = bce(tf.ones_like(fake_output), fake_output)
        g_loss_custom = combined_gan_loss(real_frontal_images, generated_frontal)
        g_total_loss = g_loss_gan + g_loss_custom
    g_grad = tape.gradient(g_total_loss, generator.trainable_variables)
    d_grad = tape.gradient(d_loss, discriminator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grad, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(d_grad, discriminator.trainable_variables))
    return d_loss, g_total_loss

# Training loop
best_g_loss = float('inf')
def train(dataset, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for profile_batch, frontal_batch in dataset:
            d_loss, g_loss = train_step(profile_batch, frontal_batch)
        print(f"D Loss: {d_loss.numpy():.4f} | G Loss: {g_loss.numpy():.4f}")

        # Save best generator
        global best_g_loss
        if g_loss < best_g_loss:
            best_g_loss = g_loss
            generator.save('best_generator_pc.h5')
            print("✅ Saved best generator.")

# Start training
train(dataset, EPOCHS)
