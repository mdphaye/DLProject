import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Initialize MTCNN and Inception Resnet Model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)  # keep_all=False to handle single face
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_face_embeddings(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None

    # Convert to RGB and detect face
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb_pil = Image.fromarray(img_rgb)
    box, _ = mtcnn.detect(img_rgb_pil)

    if box is None:
        print(f"No faces detected in image: {image_path}")
        return None

    # Get aligned face and embedding
    face_aligned = mtcnn(img_rgb_pil)
    if face_aligned is None:
        print(f"Face alignment failed for image: {image_path}")
        return None

    # Pass through model and return embedding
    face_aligned = face_aligned.to(device)
    embeddings = model(face_aligned.unsqueeze(0))
    return embeddings.squeeze(0)  # Remove batch dimension for a single embedding

def process_images():
    student_embeddings = {}
    base_dir = '/Users/mangalyaphaye/Desktop/DLProject/preprocessed_data'  # Base directory where students' data is stored

    for student_name in os.listdir(base_dir):  # Loop through student folders
        student_dir = os.path.join(base_dir, student_name)  # Get the full directory path for the student
        
        if os.path.isdir(student_dir):  # Check if it's a valid directory
            embeddings_list = []

            for img_name in os.listdir(student_dir):  # Loop through images within the student's folder
                img_path = os.path.join(student_dir, img_name)  # Get the image file path
                emb = get_face_embeddings(img_path)  # Get embeddings for each image
                
                if emb is not None:
                    embeddings_list.append(emb)

            if embeddings_list:
                student_embeddings[student_name] = torch.stack(embeddings_list).mean(dim=0)
            else:
                print(f"No valid images found for student: {student_name}")  # Handle case when no images have embeddings

    return student_embeddings

if __name__ == "__main__":
    student_embeddings = process_images()
    print("Processed student embeddings:", student_embeddings)
