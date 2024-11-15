import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import mysql.connector
from mysql.connector import Error

# Initialize MTCNN and Inception Resnet Model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Function to get face embeddings from the image
def get_face_embeddings(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb_pil = Image.fromarray(img_rgb)
    box, _ = mtcnn.detect(img_rgb_pil)

    if box is None:
        print(f"No faces detected in image: {image_path}")
        return None

    face_aligned = mtcnn(img_rgb_pil)
    if face_aligned is None:
        print(f"Face alignment failed for image: {image_path}")
        return None

    face_aligned = face_aligned.to(device)
    embeddings = model(face_aligned.unsqueeze(0))
    return embeddings.squeeze(0)

# Process all images and calculate embeddings for each student
def process_images(base_dir):
    student_embeddings = {}

    for student_name in os.listdir(base_dir):
        student_dir = os.path.join(base_dir, student_name)
        if os.path.isdir(student_dir):
            embeddings_list = []

            for img_name in os.listdir(student_dir):
                img_path = os.path.join(student_dir, img_name)
                emb = get_face_embeddings(img_path)

                if emb is not None:
                    embeddings_list.append(emb)

            if embeddings_list:
                student_embeddings[student_name] = torch.stack(embeddings_list).mean(dim=0)
            else:
                print(f"No valid images found for student: {student_name}")

    return student_embeddings

# Save embeddings to MySQL database
def save_embeddings_to_db(student_embeddings):
    try:
        db_connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="MPhaye0509*",
            database="Attendance_System_DB"
        )

        cursor = db_connection.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS StudentEmbeddings (
            student_name VARCHAR(255),
            embedding JSON
        )
        """)

        for student_name, embedding in student_embeddings.items():
            embedding_list = embedding.detach().cpu().numpy().tolist()
            cursor.execute("INSERT INTO StudentEmbeddings (student_name, embedding) VALUES (%s, %s)", 
                           (student_name, str(embedding_list)))

        db_connection.commit()
        cursor.close()
        db_connection.close()
        print("Embeddings successfully saved to the database.")

    except Error as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    base_dir = '/Users/mangalyaphaye/Desktop/DLProject/preprocessed_data'
    student_embeddings = process_images(base_dir)
    save_embeddings_to_db(student_embeddings)
    print("Processed student embeddings and saved to database.")