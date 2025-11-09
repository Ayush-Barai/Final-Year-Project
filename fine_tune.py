# ==============================================================================
# 1. SETUP AND IMPORTS
# ==============================================================================
import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, ReLU, Conv2DTranspose, concatenate, Activation
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt

# ==============================================================================
# 2. DATA LOADING FOR CFPW DATASET
# ==============================================================================

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

# ==============================================================================
# 3. MODEL ARCHITECTURE (Must be identical to pre-training)
# ==============================================================================

def define_discriminator(image_shape):
    """Defines the PatchGAN discriminator model."""
    init = tf.random_normal_initializer(0., 0.02)
    in_src_image = Input(shape=image_shape, name='source_image')
    in_target_image = Input(shape=image_shape, name='target_image')
    merged = concatenate([in_src_image, in_target_image])

    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)

    model = Model([in_src_image, in_target_image], d, name="Discriminator")
    return model

def define_generator(image_shape=(128, 128, 3)):
    """Defines the U-Net generator model."""
    def define_encoder_block(layer_in, n_filters, batchnorm=True):
        init = tf.random_normal_initializer(0., 0.02)
        g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
        if batchnorm: g = BatchNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)
        return g

    def decoder_block(layer_in, skip_in, n_filters, dropout=True):
        init = tf.random_normal_initializer(0., 0.02)
        g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
        g = BatchNormalization()(g)
        if dropout: g = tf.keras.layers.Dropout(0.5)(g)
        g = concatenate([g, skip_in])
        g = ReLU()(g)
        return g

    in_image = Input(shape=image_shape)
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    b = Conv2D(512, (4,4), strides=(2,2), padding='same')(e6)
    b = ReLU()(b)
    d1 = decoder_block(b, e6, 512)
    d2 = decoder_block(d1, e5, 512)
    d3 = decoder_block(d2, e4, 512)
    d4 = decoder_block(d3, e3, 256, dropout=False)
    d5 = decoder_block(d4, e2, 128, dropout=False)
    d6 = decoder_block(d5, e1, 64, dropout=False)
    g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same')(d6)
    out_image = Activation('tanh')(g)

    model = Model(in_image, out_image, name="Generator")
    return model

# ==============================================================================
# 4. FINE-TUNING SETUP
# ==============================================================================

# --- Use a LOWER learning rate for fine-tuning ---
# This prevents the model from forgetting its well-learned features from Multi-PIE.
generator_optimizer = tf.keras.optimizers.Adam(2e-5, beta_1=0.5) # 1/10th of original
discriminator_optimizer = tf.keras.optimizers.Adam(2e-5, beta_1=0.5) # 1/10th of original

binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
L1_LOSS_LAMBDA = 10

def discriminator_loss(real_output, fake_output):
    real_loss = binary_cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output, generated_images, real_images):
    adversarial_loss = binary_cross_entropy(tf.ones_like(fake_output), fake_output)
    l1_loss = tf.reduce_mean(tf.abs(real_images - generated_images))
    return adversarial_loss + (L1_LOSS_LAMBDA * l1_loss)

@tf.function
def train_step(g_model, d_model, source_images, target_images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = g_model(source_images, training=True)
        real_output = d_model([source_images, target_images], training=True)
        fake_output = d_model([source_images, generated_images], training=True)
        gen_loss = generator_loss(fake_output, generated_images, target_images)
        disc_loss = discriminator_loss(real_output, fake_output)

    generator_gradients = gen_tape.gradient(gen_loss, g_model.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, d_model.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, g_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, d_model.trainable_variables))
    return disc_loss, gen_loss

def plot_results(g_model, src_images, n_samples=3):
    ix = np.random.randint(0, len(src_images), n_samples)
    X_in, X_out_fake = src_images[ix], g_model.predict(src_images[ix], verbose=0)
    X_in, X_out_fake = (X_in + 1) / 2.0, (X_out_fake + 1) / 2.0

    plt.figure(figsize=(8, n_samples * 4))
    for i in range(n_samples):
        plt.subplot(n_samples, 2, 1 + i*2); plt.axis('off'); plt.imshow(X_in[i]); plt.title('Source (Profile)')
        plt.subplot(n_samples, 2, 2 + i*2); plt.axis('off'); plt.imshow(X_out_fake[i]); plt.title('Generated (Frontal)')
    plt.show()

def train_finetune(g_model, d_model, dataset, n_epochs=50, n_batch=8):
    """Fine-tuning loop that saves the best fine-tuned model."""
    src_images, tar_images = dataset
    models_dir = '/content/drive/MyDrive/New_Model_4/FaceFrontalization_Models'
    best_g_loss = float('inf')

    print(f"\n--- Starting Fine-Tuning ---\nEpochs: {n_epochs}, Batch Size: {n_batch}")

    for epoch in range(n_epochs):
        epoch_g_loss_avg = []
        tf_dataset = tf.data.Dataset.from_tensor_slices((src_images, tar_images)).shuffle(len(src_images)).batch(n_batch)

        for step, (source_batch, target_batch) in enumerate(tqdm(tf_dataset, desc=f"Epoch {epoch + 1}/{n_epochs}")):
            d_loss, g_loss = train_step(g_model, d_model, source_batch, target_batch)
            epoch_g_loss_avg.append(g_loss)

        avg_loss = np.mean(epoch_g_loss_avg)
        print(f"End of Epoch {epoch + 1}: Avg. Discriminator Loss={d_loss:.4f}, Avg. Generator Loss={avg_loss:.4f}")

        if avg_loss < best_g_loss:
            best_g_loss = avg_loss
            model_path = os.path.join(models_dir, 'best_finetuned_model.h5') # Save to a new file
            g_model.save(model_path)
            print(f"** New best fine-tuned model saved! Loss improved to {best_g_loss:.4f}. **")

        if (epoch + 1) % 5 == 0:
            plot_results(g_model, src_images, n_samples=2)

# ==============================================================================
# 5. EXECUTION
# ==============================================================================

if __name__ == '__main__':
    # --- Define Constants ---
    IMAGE_SIZE = (128, 128)
    BATCH_SIZE = 8
    EPOCHS_FINETUNE = 50 # Fine-tuning often requires fewer epochs
    DATA_PATH_CFPW = '/content/dataset/cfp-dataset/Data/Images'

    # --- Path to your pre-trained model from the Multi-PIE training ---
    PRETRAINED_MODEL_PATH = '/content/drive/MyDrive/New_Model_4/FaceFrontalization_Models/best_generator_model.h5'

    # --- Load the new dataset for fine-tuning ---
    [src_images, tar_images] = load_images(DATA_PATH_CFPW, size=IMAGE_SIZE)

    if len(src_images) > 0:
        # --- Define model architecture ---
        image_shape = src_images.shape[1:]
        g_model = define_generator(image_shape)
        d_model = define_discriminator(image_shape)

        # --- THE CRUCIAL STEP: LOAD PRE-TRAINED WEIGHTS ---
        if os.path.exists(PRETRAINED_MODEL_PATH):
            print(f"--- Loading pre-trained weights from: {PRETRAINED_MODEL_PATH} ---")
            g_model.load_weights(PRETRAINED_MODEL_PATH)
        else:
            print(f"--- WARNING: Pre-trained model not found at {PRETRAINED_MODEL_PATH}. Training from scratch. ---")

        # --- Begin Fine-Tuning ---
        train_finetune(g_model, d_model, [src_images, tar_images], n_epochs=EPOCHS_FINETUNE, n_batch=BATCH_SIZE)
        print("\n--- Fine-Tuning Complete ---")
    else:
        print("\n--- No images were loaded from CFPW dataset. Please check the DATA_PATH. ---")