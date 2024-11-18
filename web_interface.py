import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import mysql.connector
from mysql.connector import Error
from scipy.spatial.distance import cosine

# Initialize MTCNN and Inception Resnet Model
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
            password="MPhaye0509*",
            database="Attendance_System_DB"
        )
        cursor = db_connection.cursor()
        cursor.execute("SELECT student_name, embedding FROM StudentEmbeddings")
        for student_name, embedding_json in cursor.fetchall():
            embedding_array = np.array(eval(embedding_json))  # Convert JSON string back to numpy array
            embeddings_dict[student_name] = torch.tensor(embedding_array).to(device)
        cursor.close()
        db_connection.close()
    except Error as e:
        print(f"Error: {e}")
    return embeddings_dict

# Get embedding for the detected face
def get_face_embedding_from_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_rgb_pil = Image.fromarray(img_rgb)
    face_aligned = mtcnn(img_rgb_pil)
    if face_aligned is None:
        return None
    face_aligned = face_aligned.to(device)
    embedding = model(face_aligned.unsqueeze(0))
    return embedding.squeeze(0)

# Match detected face embedding with database embeddings
def match_face(embedding, database_embeddings, threshold=0.6):
    min_distance = float("inf")
    best_match = None
    for student_name, db_embedding in database_embeddings.items():
        distance = cosine(embedding.detach().cpu().numpy(), db_embedding.detach().cpu().numpy())
        if distance < min_distance:
            min_distance = distance
            best_match = student_name
    if min_distance < threshold:
        return best_match
    return None

# Real-time face recognition integrated with background
def real_time_face_recognition_with_background(database_embeddings):
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 627)  # Set width
    cap.set(4, 466)  # Set height

    # Load the background image
    imgBackground = cv2.imread(r'/Users/mangalyaphaye/Desktop/DLProject/frame.jpg')
    if imgBackground is None:
        print("Error: Could not load the background image. Check the file path.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    # Define box dimensions
    box_x, box_y = 60, 160
    box_width, box_height = 630, 482

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture image from webcam.")
            break

        # Resize webcam frame to fit the background box
        img_resized = cv2.resize(frame, (box_width, box_height))
        imgBackground[box_y:box_y + box_height, box_x:box_x + box_width] = img_resized

        # Perform face recognition
        embedding = get_face_embedding_from_frame(frame)
        if embedding is not None:
            matched_name = match_face(embedding, database_embeddings)
            # n = len(matched_name)
            if matched_name:
                # for n in range(0,30):
                # Display matched name on the background
                cv2.putText(imgBackground, f"Matched: {matched_name}", (box_x + 10, box_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f"Attendance marked for: {matched_name}")
            else:
                cv2.putText(imgBackground, "No match found", (box_x + 10, box_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the final output
        cv2.imshow("Face Attendance", imgBackground)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main script
if __name__ == "__main__":
    # Load database embeddings
    database_embeddings = load_embeddings_from_db()
    # Start real-time face recognition with background
    real_time_face_recognition_with_background(database_embeddings)