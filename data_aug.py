import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance

# Function to augment and save images
def augment_images(input_dir, target_size=(1920, 1080), num_augmented_images=10):
    # Walk through the input directory to get images
    for subdir, _, files in os.walk(input_dir):
        # No need to create a separate output directory
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(subdir, filename)
                
                # Open the image using OpenCV and resize it
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error loading image: {image_path}")
                    continue
                image = cv2.resize(image, target_size)  # Resize the image
                
                # Convert OpenCV image (BGR) to PIL format (RGB) for further augmentations
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                # Generate augmented images
                for i in range(num_augmented_images):
                    augmented_image = image_pil
                    
                    # Random rotation
                    angle = np.random.randint(-20, 20)
                    augmented_image = augmented_image.rotate(angle)
                    
                    # Random brightness adjustment
                    enhancer = ImageEnhance.Brightness(augmented_image)
                    augmented_image = enhancer.enhance(np.random.uniform(0.8, 1.2))
                    
                    # Random horizontal flip
                    if np.random.rand() > 0.5:
                        augmented_image = augmented_image.transpose(Image.FLIP_LEFT_RIGHT)
                    
                    # Convert back to OpenCV format (BGR) before saving
                    augmented_image_cv = cv2.cvtColor(np.array(augmented_image), cv2.COLOR_RGB2BGR)
                    
                    # Save augmented image in the original input directory
                    aug_filename = f"aug_{i}_{filename}"
                    cv2.imwrite(os.path.join(subdir, aug_filename), augmented_image_cv)

# Example usage
input_directory = '/Users/mangalyaphaye/Desktop/DLProject/dataset'  # <-- Update with your input directory

# Run the augmentation
augment_images(input_directory, num_augmented_images=4)