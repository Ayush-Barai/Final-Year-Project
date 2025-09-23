import os
import glob
import shutil
from tqdm import tqdm
from collections import defaultdict

# --- Run these prerequisite commands in a separate cell first ---
#
from google.colab import drive
drive.mount('/content/drive')

# !pip install kaggle tqdm
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# # # print("--- Downloading Multi-PIE dataset from Kaggle... ---")
# # # # Download the dataset to the Colab temporary storage
# !kaggle datasets download -d aliates/multi-pie -p /content/multi-pie-raw --unzip

# --- Input path where the raw Multi-PIE images were downloaded ---
# This path should point to the folder containing the subject folders (001, 002, etc.)
# Note: The Kaggle dataset might have a nested structure, so adjust if needed.
# Example: /content/multi-pie-raw/Images
RAW_DATA_PATH = '/content/multi-pie-raw/Multi_Pie/HR_128'

# --- Output path on your Google Drive where the formatted dataset will be saved ---
# The script will create this directory.
PROCESSED_DATA_PATH = '/content/drive/MyDrive/Multi-PIE_Formatted_for_GAN'

# --- Multi-PIE Pose IDs ---
# You can customize which poses to use as "profile" views.
FRONTAL_POSE_ID = '051'
# These are good profile poses: -45, +45, -90, +90 degrees
PROFILE_POSE_IDS = ['041', '140', '130', '010']

print(f"Input Path: {RAW_DATA_PATH}")
print(f"Output Path: {PROCESSED_DATA_PATH}")

def process_and_reorganize_multipie(raw_path, processed_path):
    """
    Finds all Multi-PIE images, sorts them by person, and copies them into a
    CFPW-like folder structure (person_id/profile/ and person_id/frontal/).
    """
    if not os.path.exists(raw_path):
        print(f"Error: Raw data path does not exist: {raw_path}")
        print("Please ensure the dataset was downloaded and the path is correct.")
        return

    # Create the main output directory
    os.makedirs(processed_path, exist_ok=True)
    print(f"--- Created output directory at: {processed_path} ---")

    # Find all image files recursively
    all_images = glob.glob(os.path.join(raw_path, '**', '*.png'), recursive=True)
    if not all_images:
        print("Error: No images found in the raw data path. Check the directory structure.")
        return

    # Group images by subject ID
    images_by_subject = defaultdict(list)
    for img_path in all_images:
        try:
            subject_id = os.path.basename(img_path).split('_')[0]
            images_by_subject[subject_id].append(img_path)
        except IndexError:
            print(f"Warning: Skipping file with unexpected name format: {img_path}")

    print(f"--- Found {len(all_images)} images for {len(images_by_subject)} subjects ---")
    print("--- Reorganizing into CFPW format... ---")

    subjects_processed = 0
    # Iterate through each person and their images
    for subject_id, paths in tqdm(images_by_subject.items()):

        # Create the person-specific output directories
        subject_out_dir = os.path.join(processed_path, subject_id)
        frontal_out_dir = os.path.join(subject_out_dir, 'frontal')
        profile_out_dir = os.path.join(subject_out_dir, 'profile')

        os.makedirs(frontal_out_dir, exist_ok=True)
        os.makedirs(profile_out_dir, exist_ok=True)

        frontal_count = 0
        profile_count = 0

        # Copy and rename the images
        for path in paths:
            pose_id = os.path.basename(path).split('_')[3]

            if pose_id == FRONTAL_POSE_ID:
                frontal_count += 1
                # Format the new filename, e.g., 01.jpg
                new_filename = f"{frontal_count:02d}.jpg"
                dest_path = os.path.join(frontal_out_dir, new_filename)
                # Use shutil.copy to handle file operations
                shutil.copy(path, dest_path)

            elif pose_id in PROFILE_POSE_IDS:
                profile_count += 1
                new_filename = f"{profile_count:02d}.jpg"
                dest_path = os.path.join(profile_out_dir, new_filename)
                shutil.copy(path, dest_path)

        # Only count subjects for whom we actually found and copied images
        if frontal_count > 0 or profile_count > 0:
            subjects_processed += 1

    print(f"\n--- Processing Complete! ---")
    print(f"Formatted dataset for {subjects_processed} subjects saved to your Google Drive.")
    print(f"Location: {PROCESSED_DATA_PATH}")

if __name__ == '__main__':
    # Run the main processing function
    process_and_reorganize_multipie(RAW_DATA_PATH, PROCESSED_DATA_PATH)