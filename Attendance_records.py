import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import mysql.connector
from mysql.connector import Error
from scipy.spatial.distance import cosine
import time
from datetime import datetime

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
            password="MPhaye0509*",  # Replace with your MySQL password
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

# Store attendance record with date and time in separate columns
def store_attendance(student_name):
    try:
        db_connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="MPhaye0509*",  # Replace with your MySQL password
            database="Attendance_System_DB"
        )
        cursor = db_connection.cursor()

        # Create the attendance table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS AttendanceRecords (
            student_name VARCHAR(255),
            attendance_date DATE,
            attendance_time TIME
        )
        """)

        # Get current date and time
        current_datetime = datetime.now()
        attendance_date = current_datetime.date()
        attendance_time = current_datetime.time()

        # Insert attendance record with separate date and time
        cursor.execute("INSERT INTO AttendanceRecords (student_name, attendance_date, attendance_time) VALUES (%s, %s, %s)", 
                       (student_name, attendance_date, attendance_time))

        db_connection.commit()
        cursor.close()
        db_connection.close()
        print(f"Attendance recorded for {student_name} on {attendance_date} at {attendance_time}.")
    except Error as e:
        print(f"Error: {e}")

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

# Real-time face recognition integrated with background and attendance recording
def real_time_face_recognition_with_attendance(database_embeddings, time_limit=60):  # Set time_limit in seconds
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

    start_time = time.time()  # Store the start time of the session
    attendance_marked = False  # Flag to track if attendance has been marked

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture image from webcam.")
            break

        # Check if time limit is reached
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:  # Limit the camera feed for the given time
            print("Time limit reached. Exiting...")
            break

        # Resize webcam frame to fit the background box
        img_resized = cv2.resize(frame, (box_width, box_height))
        imgBackground[box_y:box_y + box_height, box_x:box_x + box_width] = img_resized

        # Perform face recognition
        embedding = get_face_embedding_from_frame(frame)
        if embedding is not None:
            matched_name = match_face(embedding, database_embeddings)
            if matched_name and not attendance_marked:
                # Display matched name on the background
                cv2.putText(imgBackground, f"Present: {matched_name}", (box_x + 100, box_y + 535), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f"Attendance marked for: {matched_name}")
                store_attendance(matched_name)  # Save the attendance record in the database
                attendance_marked = True  # Set flag to True so attendance is not marked again

            elif not matched_name:
                cv2.putText(imgBackground, "No match found", (box_x + 100, box_y + 535), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the final output
        cv2.imshow("Face Attendance", imgBackground)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting program.")
            break

    cap.release()
    cv2.destroyAllWindows()

# Main script
if __name__ == "__main__":
    # Load database embeddings
    database_embeddings = load_embeddings_from_db()
    # Start real-time face recognition with attendance recording
    real_time_face_recognition_with_attendance(database_embeddings, time_limit=7) 