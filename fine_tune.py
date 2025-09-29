import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, ReLU, Conv2DTranspose, concatenate, Activation
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt

# DATA LOADING FOR CFPW DATASET


def load_images(base_path, size=(128, 128)):
    """
    Loads and pre-processes paired images from the CFPW dataset structure.
    """
    profile_images = []
    frontal_images = []
    person_dirs = sorted(glob.glob(os.path.join(base_path, '*')))
    
    print(f"--- Scanning {len(person_dirs)} person directories in: {base_path} ---")

    for person_dir in tqdm(person_dirs):
        profile_paths = sorted(glob.glob(os.path.join(person_dir, 'profile', '*.jpg')))
        frontal_paths = sorted(glob.glob(os.path.join(person_dir, 'frontal', '*.jpg')))
        min_len = min(len(profile_paths), len(frontal_paths))
        
        for i in range(min_len):
            try:
                profile_img = Image.open(profile_paths[i]).resize(size).convert('RGB')
                frontal_img = Image.open(frontal_paths[i]).resize(size).convert('RGB')
                
                profile_array = ((np.array(profile_img, dtype=np.float32) / 127.5) - 1.0)
                frontal_array = ((np.array(frontal_img, dtype=np.float32) / 127.5) - 1.0)
                profile_images.append(profile_array)
                frontal_images.append(frontal_array)
            except Exception as e:
                print(f"\nSkipping corrupt file pair due to error: {e}")

    return [np.array(profile_images), np.array(frontal_images)]

def discriminator_loss(real_output, fake_output):
    real_loss = binary_cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output, generated_images, real_images):
    adversarial_loss = binary_cross_entropy(tf.ones_like(fake_output), fake_output)
    l1_loss = tf.reduce_mean(tf.abs(real_images - generated_images))
    return adversarial_loss + (L1_LOSS_LAMBDA * l1_loss)
