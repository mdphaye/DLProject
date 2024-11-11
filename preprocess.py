import cv2
import os
import numpy as np

# Function to preprocess images
def preprocess_image(image_path, target_size=(256, 256)):
    # Load the image from file
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    # Resize the image
    resized_image = cv2.resize(image, target_size)

    # Normalize the image (pixel values between 0 and 1)
    normalized_image = resized_image / 255.0

    return normalized_image

# Example usage for processing images in subdirectories
def process_images_in_directory(input_dir, output_dir, target_size=(1920, 1080)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Traverse through all subdirectories
    for subdir, _, files in os.walk(input_dir):
        # Create corresponding output subdirectory
        output_subdir = subdir.replace(input_dir, output_dir, 1)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(subdir, filename)
                
                # Preprocess the image
                preprocessed_image = preprocess_image(image_path, target_size)
                
                if preprocessed_image is not None:  # Check if image was successfully preprocessed
                    # Save the processed image to the corresponding subdirectory
                    output_path = os.path.join(output_subdir, filename)
                    # Convert back to uint8 for saving the image
                    cv2.imwrite(output_path, (preprocessed_image * 255).astype(np.uint8))
                    print(f"Processed and saved: {output_path}")
                else:
                    print(f"Skipping saving for image: {image_path}")

# **Update this with the path to your input dataset folder**
input_directory = '/Users/mangalyaphaye/Desktop/DLProject/dataset'  # <-- Update this with your project path

# **Update this with the path where you want to save processed images**
output_directory = '/Users/mangalyaphaye/Desktop/DLProject/preprocessed_data'  # <-- Update this with your project path

# Process the dataset
process_images_in_directory(input_directory, output_directory)