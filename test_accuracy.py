import numpy as np
import mysql.connector
from mysql.connector import Error
from scipy.spatial.distance import cosine
import torch
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Initialize MTCNN and FaceNet Model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load embeddings from the MySQL database
def load_embeddings_from_db():
    embeddings_dict = {}
    try:
        db_connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="MPhaye0509*",  # Replace with your actual MySQL password
            database="Attendance_System_DB"
        )
        cursor = db_connection.cursor()
        cursor.execute("SELECT student_name, embedding FROM StudentEmbeddings")
        
        for student_name, embedding_json in cursor.fetchall():
            embedding_array = np.array(eval(embedding_json))  # Convert JSON string to numpy array
            embeddings_dict[student_name] = torch.tensor(embedding_array)

        cursor.close()
        db_connection.close()
    except Error as e:
        print(f"Error: {e}")
    return embeddings_dict

# Function to get face embedding from an image frame
def get_face_embedding_from_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_rgb_pil = Image.fromarray(img_rgb)
    face_aligned = mtcnn(img_rgb_pil)
    
    if face_aligned is None:
        return None
    
    face_aligned = face_aligned.to(device)
    embedding = model(face_aligned.unsqueeze(0))
    return embedding.squeeze(0)

# Function to evaluate accuracy
def evaluate_accuracy(database_embeddings, test_images, true_labels, threshold=0.2):
    correct_matches = 0
    total_tests = len(test_images)

    for i, test_image in enumerate(test_images):
        test_embedding = get_face_embedding_from_frame(test_image)

        if test_embedding is not None:
            best_match = None
            min_distance = float("inf")

            for student_name, db_embedding in database_embeddings.items():
                distance = cosine(test_embedding.detach().cpu().numpy(), db_embedding.detach().cpu().numpy())

                if distance < min_distance:
                    min_distance = distance
                    best_match = student_name

            if min_distance < threshold and best_match == true_labels[i]:  # Check if prediction matches true label
                correct_matches += 1

    accuracy = (correct_matches / total_tests) * 100
    return accuracy

# Load database embeddings
database_embeddings = load_embeddings_from_db()  # Now this function exists

# Test Images
test_images = [
    cv2.imread("/Users/mangalyaphaye/Desktop/DLProject/preprocessed_data/Aditi_Priya/Aditi_Priya_9.jpg"),
    cv2.imread("/Users/mangalyaphaye/Desktop/DLProject/preprocessed_data/Aiman_Choudhri/Aiman_Choudhri_2.jpg"),
    cv2.imread("/Users/mangalyaphaye/Desktop/DLProject/preprocessed_data/Atharva_Jaiswal/Atharva_Jaiswal_0.jpg"),
    cv2.imread("/Users/mangalyaphaye/Desktop/DLProject/preprocessed_data/Aziz_Barwaniwala/Aziz_Barwaniwala_8.jpg"),
    cv2.imread("/Users/mangalyaphaye/Desktop/DLProject/preprocessed_data/Bhavika_Malik/Bhavika_Malik_4.jpg"),
    cv2.imread("/Users/mangalyaphaye/Desktop/DLProject/preprocessed_data/Devesh_Nahar/Devesh_Nahar_5.jpg"),
    cv2.imread("/Users/mangalyaphaye/Desktop/DLProject/preprocessed_data/Ishar_Singh/Ishar_Singh_7.jpg"),
    cv2.imread("/Users/mangalyaphaye/Desktop/DLProject/preprocessed_data/Khushi_Bansal/Khushi_Bansal_0.jpg"),
    cv2.imread("/Users/mangalyaphaye/Desktop/DLProject/preprocessed_data/Khwaish_Mankotia/Khwaish_Mankotia_0.jpg"),
    cv2.imread("/Users/mangalyaphaye/Desktop/DLProject/preprocessed_data/Mangalya_Phaye/Mangalya_Phaye_3.jpg"),
    cv2.imread("/Users/mangalyaphaye/Desktop/DLProject/preprocessed_data/Ragini_Grover/Ragini_Grover_5.jpg"),
    cv2.imread("/Users/mangalyaphaye/Desktop/DLProject/preprocessed_data/Ram_K/Ram_K_0.jpg"),
    cv2.imread("/Users/mangalyaphaye/Desktop/DLProject/preprocessed_data/Ramya_Mishra/Ramya_Mishra_10.jpg"),
    cv2.imread("/Users/mangalyaphaye/Desktop/DLProject/preprocessed_data/Ridima_Garg/Ridima_Garg_14.jpg"),
    cv2.imread("/Users/mangalyaphaye/Desktop/DLProject/preprocessed_data/Rashika_Ranjan/Rashika_Ranjan_11.jpg"),
]

# True Labels
true_labels = ["Aditi_Priya","Aiman_Choudhri","Atharva_Jaiswal","Aziz_Barwaniwala","Bhavika_Malik","Devesh_Nahar","Ishar_Singh","Khushi_Bansal","Khwaish_Mankotia","Mangalya_Phaye","Ragini_Grover","Ram_K","Ramya_Mishra","Ridima_Garg", "Rashika_Ranjan"]  # Replace with actual student names

# Evaluate accuracy
accuracy = evaluate_accuracy(database_embeddings, test_images, true_labels, threshold=0.6)
print(f"Face Recognition Accuracy: {accuracy:.2f}%")
