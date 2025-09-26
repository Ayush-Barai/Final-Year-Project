# ==============================================================================
# 1. SETUP AND IMPORTS
# ==============================================================================
import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
# 2. EFFICIENT DATA PIPELINE
# ==============================================================================

def get_image_paths(base_path):
    """Gets paired profile and frontal image paths without loading them into memory."""
    profile_paths = []
    frontal_paths = []
    person_dirs = sorted(glob.glob(os.path.join(base_path, '*')))
    print(f"--- Scanning {len(person_dirs)} person directories in: {base_path} ---")

    for person_dir in tqdm(person_dirs):
        p_paths = sorted(glob.glob(os.path.join(person_dir, 'profile', '*.jpg')))
        f_paths = sorted(glob.glob(os.path.join(person_dir, 'frontal', '*.jpg')))
        min_len = min(len(p_paths), len(f_paths))

        profile_paths.extend(p_paths[:min_len])
        frontal_paths.extend(f_paths[:min_len])

    return profile_paths, frontal_paths

def load_and_preprocess_image(profile_path, frontal_path, size):
    """
    Reads, decodes, and preprocesses a single pair of images using TensorFlow operations.
    This function will be mapped over the dataset of file paths.
    """
    # Read files from disk
    profile_img = tf.io.read_file(profile_path)
    frontal_img = tf.io.read_file(frontal_path)

    # Decode JPEG to a tensor
    profile_img = tf.io.decode_jpeg(profile_img, channels=3)
    frontal_img = tf.io.decode_jpeg(frontal_img, channels=3)

    # Resize images
    profile_img = tf.image.resize(profile_img, size)
    frontal_img = tf.image.resize(frontal_img, size)

    # Normalize pixel values to the range [-1, 1]
    profile_img = (tf.cast(profile_img, tf.float32) / 127.5) - 1.0
    frontal_img = (tf.cast(frontal_img, tf.float32) / 127.5) - 1.0

    return profile_img, frontal_img

def create_dataset_pipeline(profile_paths, frontal_paths, batch_size, image_size):
    """Creates a tf.data.Dataset pipeline for efficient, parallelized training."""
    # Create a dataset of file path pairs
    dataset = tf.data.Dataset.from_tensor_slices((profile_paths, frontal_paths))

    # Shuffle the file paths. This is more memory-efficient than shuffling images.
    dataset = dataset.shuffle(buffer_size=len(profile_paths), reshuffle_each_iteration=True)

    # Map the preprocessing function to each path pair.
    # num_parallel_calls=tf.data.AUTOTUNE allows TF to process multiple images in parallel.
    dataset = dataset.map(lambda p, f: load_and_preprocess_image(p, f, image_size),
                          num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the data into groups of `batch_size`
    dataset = dataset.batch(batch_size)

    # Prefetch the next batch while the current one is being processed on the GPU.
    # This overlaps data preprocessing and model execution for optimal performance.
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


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

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

L1_LOSS_LAMBDA = 100

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

def plot_results(g_model, test_dataset, n_samples=3):
    """Generates and plots a sample of results from the test dataset."""
    # Take one batch from the test dataset to use for plotting
    for src_batch, _ in test_dataset.take(1):
        # Select n_samples from the batch
        if src_batch.shape[0] < n_samples:
            n_samples = src_batch.shape[0]
        X_in = src_batch[:n_samples]
        X_out_fake = g_model.predict(X_in, verbose=0)

        # Scale pixels from [-1,1] to [0,1] for plotting
        X_in = (X_in + 1) / 2.0
        X_out_fake = (X_out_fake + 1) / 2.0

        plt.figure(figsize=(8, n_samples * 4))
        for i in range(n_samples):
            plt.subplot(n_samples, 2, 1 + i*2); plt.axis('off'); plt.imshow(X_in[i]); plt.title('Source (Profile)')
            plt.subplot(n_samples, 2, 2 + i*2); plt.axis('off'); plt.imshow(X_out_fake[i]); plt.title('Generated (Frontal)')
        plt.show()
        break # Only use the first batch

def train(g_model, d_model, train_tf_dataset, test_tf_dataset, n_epochs=200):
    """Optimized training loop that accepts tf.data.Dataset objects."""
    checkpoint_dir = '/content/drive/MyDrive/New_Model_3/FaceFrontalization_Checkpoints'
    best_model_dir = '/content/drive/MyDrive/New_Model_3/FaceFrontalization_Models'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    best_g_loss = float('inf')

    print(f"\n--- Starting Optimized Training ---\nEpochs: {n_epochs}")

    for epoch in range(n_epochs):
        epoch_g_loss_avg = tf.keras.metrics.Mean()
        epoch_d_loss_avg = tf.keras.metrics.Mean()

        # The dataset is already shuffled and batched
        for source_batch, target_batch in tqdm(train_tf_dataset, desc=f"Epoch {epoch + 1}/{n_epochs}"):
            d_loss, g_loss = train_step(g_model, d_model, source_batch, target_batch)
            epoch_d_loss_avg.update_state(d_loss)
            epoch_g_loss_avg.update_state(g_loss)

        avg_g_loss = epoch_g_loss_avg.result()
        avg_d_loss = epoch_d_loss_avg.result()
        print(f"End of Epoch {epoch + 1}: Avg. Discriminator Loss={avg_d_loss:.4f}, Avg. Generator Loss={avg_g_loss:.4f}")

        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            best_model_path = os.path.join(best_model_dir, 'best_generator_model.h5')
            g_model.save(best_model_path)
            print(f"** New best model saved! Loss improved to {best_g_loss:.4f}. Model saved to {best_model_path} **")
        else:
            print(f"No improvement in model loss this epoch (best: {best_g_loss:.4f}).")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'g_model_epoch_{epoch+1}.h5')
            g_model.save(checkpoint_path)
            print(f"** Checkpoint saved for epoch {epoch + 1} at {checkpoint_path} **")

            # Plot results using the test dataset
            plot_results(g_model, test_tf_dataset, n_samples=2)

# ==============================================================================
# 5. EXECUTION
# ==============================================================================

if __name__ == '__main__':
    # RECOMMENDATION 1: Start with a smaller batch size if you encounter GPU memory errors.
    # 64 might be too large for a 15GB GPU. Start with 16 or 32 and increase if possible.
    BATCH_SIZE = 32

    # RECOMMENDATION 2: Increase Epochs significantly for GAN convergence.
    # 10 epochs is not enough. 100-200 is a more realistic starting point.
    EPOCHS = 100

    IMAGE_SIZE = (128, 128)
    DATA_PATH = '/content/dataSet'

    # STEP 1: Get all file paths (very low memory usage)
    src_paths, tar_paths = get_image_paths(DATA_PATH)

    if len(src_paths) > 0:
        # STEP 2: Split the *paths* into training and testing sets, not the images
        train_src_paths, test_src_paths, train_tar_paths, test_tar_paths = train_test_split(
            src_paths, tar_paths, test_size=0.2, random_state=42
        )
        print(f"Data split: {len(train_src_paths)} training pairs, {len(test_src_paths)} testing pairs.")

        # STEP 3: Create the tf.data pipelines for training and testing
        train_dataset = create_dataset_pipeline(train_src_paths, train_tar_paths, BATCH_SIZE, IMAGE_SIZE)
        test_dataset = create_dataset_pipeline(test_src_paths, test_tar_paths, BATCH_SIZE, IMAGE_SIZE)

        # Define models
        image_shape = IMAGE_SIZE + (3,)
        g_model = define_generator(image_shape)
        d_model = define_discriminator(image_shape)

        print("\n--- Generator Summary ---")
        g_model.summary()

        # STEP 4: Train the model using the efficient data pipelines
        train(g_model, d_model, train_dataset, test_dataset, n_epochs=EPOCHS)

        print("\n--- Training Complete ---")
    else:
        print("\n--- No image paths were found. Please check that the DATA_PATH is correct. ---")