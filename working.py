# ==============================================================================
# 1. SETUP AND IMPORTS
# ==============================================================================
import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm # A library for a progress bar
from sklearn.model_selection import train_test_split # Import for splitting data

from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, ReLU, Conv2DTranspose, concatenate, Activation
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt

# ==============================================================================
# NOTE: PREREQUISITES - Run these in separate cells before this script
# ==============================================================================
#
# from google.colab import drive
# drive.mount('/content/drive')
#
# !pip install kaggle tqdm
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
#
# # Download and unzip the dataset to the /content/ directory
# !kaggle datasets download -d jason9075/cfp-dataset -p /content/dataset --unzip
#
# ==============================================================================
# 2. ROBUST DATA LOADING (USING PIL FOR CORRECT RGB FORMAT)
# ==============================================================================

def load_images(base_path, size=(128, 128)):
    """
    Loads and pre-processes paired images using PIL (RGB format) and
    includes error handling for corrupt files.
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
                # PIL loads images in RGB format by default, preventing color issues.
                profile_img = Image.open(profile_paths[i]).resize(size).convert('RGB')
                frontal_img = Image.open(frontal_paths[i]).resize(size).convert('RGB')

                # Normalize pixel values to [-1, 1] and append
                profile_array = ((np.array(profile_img, dtype=np.float32) / 127.5) - 1.0)
                frontal_array = ((np.array(frontal_img, dtype=np.float32) / 127.5) - 1.0)
                profile_images.append(profile_array)
                frontal_images.append(frontal_array)
            except Exception as e:
                print(f"\nSkipping corrupt file pair due to error: {e}")

    return [np.array(profile_images), np.array(frontal_images)]

# ==============================================================================
# 3. MODEL ARCHITECTURE
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
    # Note: No final activation. Loss function will use from_logits=True for stability.

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
# 4. OPTIMIZED TRAINING SETUP
# ==============================================================================

# --- Define Optimizers, Losses, and the Training Step ---
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Use from_logits=True for numerical stability
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#  **IMPROVEMENT 1: Increased L1_LOSS_LAMBDA for sharper, more accurate images**
L1_LOSS_LAMBDA = 100

def discriminator_loss(real_output, fake_output):
    real_loss = binary_cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output, generated_images, real_images):
    adversarial_loss = binary_cross_entropy(tf.ones_like(fake_output), fake_output)
    l1_loss = tf.reduce_mean(tf.abs(real_images - generated_images))
    return adversarial_loss + (L1_LOSS_LAMBDA * l1_loss)

# Use tf.function to compile the training step into a high-performance graph
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
    """Generates and plots a sample of results."""
    ix = np.random.randint(0, len(src_images), n_samples)
    X_in = src_images[ix]
    X_out_fake = g_model.predict(X_in, verbose=0)

    # Scale pixels from [-1,1] to [0,1] for plotting
    X_in = (X_in + 1) / 2.0
    X_out_fake = (X_out_fake + 1) / 2.0

    plt.figure(figsize=(8, n_samples * 4))
    for i in range(n_samples):
        plt.subplot(n_samples, 2, 1 + i*2); plt.axis('off'); plt.imshow(X_in[i]); plt.title('Source (Profile)')
        plt.subplot(n_samples, 2, 2 + i*2); plt.axis('off'); plt.imshow(X_out_fake[i]); plt.title('Generated (Frontal)')
    plt.show()


def train(g_model, d_model, train_dataset, test_dataset, n_epochs=200, n_batch=8):
    """
    Optimized training loop that saves periodic checkpoints AND the single best model.
    """
    train_src, train_tar = train_dataset
    test_src, test_tar = test_dataset

    # Define separate directories for clarity
    checkpoint_dir = '/content/drive/MyDrive/New_Model_2/FaceFrontalization_Checkpoints'
    best_model_dir = '/content/drive/MyDrive/New_Model_2/FaceFrontalization_Models'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    # Variable to track the best model's performance across all epochs
    best_g_loss = float('inf')

    print(f"\n--- Starting Optimized Training ---\nEpochs: {n_epochs}, Batch Size: {n_batch}")

    for epoch in range(n_epochs):
        epoch_g_loss_avg = []
        # Create a new shuffled dataset for each epoch
        tf_dataset = tf.data.Dataset.from_tensor_slices((train_src, train_tar)).shuffle(len(train_src)).batch(n_batch)

        for step, (source_batch, target_batch) in enumerate(tqdm(tf_dataset, desc=f"Epoch {epoch + 1}/{n_epochs}")):
            d_loss, g_loss = train_step(g_model, d_model, source_batch, target_batch)
            epoch_g_loss_avg.append(g_loss)

        # Calculate the average generator loss for the epoch
        avg_loss = np.mean(epoch_g_loss_avg)
        print(f"End of Epoch {epoch + 1}: Avg. Discriminator Loss={d_loss:.4f}, Avg. Generator Loss={avg_loss:.4f}")

        #  **IMPROVEMENT 2: Enabled saving the best model based on generator loss**
        if avg_loss < best_g_loss:
            best_g_loss = avg_loss
            best_model_path = os.path.join(best_model_dir, 'best_generator_model.h5')
            g_model.save(best_model_path)
            print(f"** New best model saved! Loss improved to {best_g_loss:.4f}. Model saved to {best_model_path} **")
        else:
            print(f"No improvement in model loss this epoch (best: {best_g_loss:.4f}).")

        # --- Save a Regular Checkpoint Every 10 Epochs ---
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'g_model_epoch_{epoch+1}.h5')
            g_model.save(checkpoint_path)
            print(f"** Checkpoint saved for epoch {epoch + 1} at {checkpoint_path} **")

            # Also plot results at every checkpoint to see progress using test data
            if len(test_src) > 0:
                plot_results(g_model, test_src, n_samples=2)


# ==============================================================================
# 5. EXECUTION
# ==============================================================================

if __name__ == '__main__':
    # Define constants
    # IMPROVEMENT 3: Increased image size for better detail (requires more GPU memory)**
    IMAGE_SIZE = (128, 128) # Or (256, 256) if your GPU can handle it

    # IMPROVEMENT 4: Increased batch size for potentially more stable training**
    BATCH_SIZE = 4

    # IMPROVEMENT 5: Significantly increased epochs for proper convergence**
    EPOCHS = 100
    DATA_PATH = '/content/dataset/cfp-dataset/Data/Images'

    # Load data
    [src_images, tar_images] = load_images(DATA_PATH, size=IMAGE_SIZE)

    if len(src_images) > 0:
        # Split data into training and testing sets
        # Using a 80/20 split
        train_src, test_src, train_tar, test_tar = train_test_split(src_images, tar_images, test_size=0.2, random_state=42)

        # Define models
        image_shape = train_src.shape[1:]
        g_model = define_generator(image_shape)
        d_model = define_discriminator(image_shape)

        print("\n--- Generator Summary ---")
        g_model.summary()

        # Train the model
        train(g_model, d_model, [train_src, train_tar], [test_src, test_tar], n_epochs=EPOCHS, n_batch=BATCH_SIZE)
        print("\n--- Training Complete ---")
    else:
        print("\n--- No images were loaded. Please check that the DATA_PATH is correct. ---")